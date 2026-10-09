from __future__ import annotations

import csv
import gc
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import cv2
import numpy as np
import torch

from semantic_redetector import SemanticMotionRedetector as GlobalTargetRedetector, bbox_center, bbox_wh


ProgressCB = Optional[Callable[[str, int, int], None]]
BBox = Tuple[int, int, int, int]


def fit_frame(frame: np.ndarray, max_side: int = 960) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    nw, nh = max(2, int(w * scale)), max(2, int(h * scale))
    nw -= nw % 2
    nh -= nh % 2
    if (nw, nh) != (w, h):
        frame = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    return frame


def extract_rgb_frames(video_path, frames_dir, max_side=768, progress: ProgressCB = None):
    video_path, frames_dir = Path(video_path), Path(frames_dir)

    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if not np.isfinite(fps) or fps <= 0:
        fps = 25.0

    idx = 0
    out_w = out_h = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = fit_frame(frame, max_side=max_side)

        if out_w is None:
            out_h, out_w = frame.shape[:2]

        cv2.imwrite(str(frames_dir / f"frame_{idx:06d}.png"), frame)
        idx += 1

        if progress:
            progress("extract", idx, max(total, idx))

    cap.release()

    if idx == 0:
        raise RuntimeError("No video frames were decoded.")

    return {
        "fps": fps,
        "frames": idx,
        "width": out_w,
        "height": out_h,
        "source_width": src_w,
        "source_height": src_h,
    }


def first_frame(frames_dir):
    paths = sorted(Path(frames_dir).glob("frame_*.png"))
    if not paths:
        raise RuntimeError("No RGB frames found.")

    frame = cv2.imread(str(paths[0]))
    if frame is None:
        raise RuntimeError("Could not read first RGB frame.")

    return frame


def mask_from_logits(logits, width: int, height: int):
    if logits is None:
        return np.zeros((height, width), dtype=bool), 0.0

    tensor = logits[0]
    prob = torch.sigmoid(tensor).detach().float().cpu().numpy()

    prob = np.squeeze(prob)
    mask = prob > 0.5

    if mask.shape != (height, width):
        mask = cv2.resize(
            mask.astype(np.uint8),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

        prob = cv2.resize(
            prob.astype(np.float32),
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )

    if mask.any():
        confidence = float(np.clip(prob[mask].mean(), 0.0, 1.0))
    else:
        confidence = float(np.clip(prob.max() if prob.size else 0.0, 0.0, 1.0))

    return mask, confidence


def bbox_from_mask(mask: np.ndarray):
    if not mask.any():
        return None

    ys, xs = np.where(mask)

    return (
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1,
    )


def mask_quality(
    mask: np.ndarray,
    confidence: float,
    previous_area: Optional[int],
    initial_area: int,
):
    area = int(mask.sum())

    if area <= 0:
        return False, area

    if confidence < 0.50:
        return False, area

    min_area = max(4, int(initial_area * 0.025))
    if area < min_area:
        return False, area

    if previous_area is not None and previous_area > 0:
        ratio = area / float(previous_area)
        if ratio < 0.15 or ratio > 6.0:
            return False, area

    return True, area


def build_camera_predictor(sam_workdir, checkpoint):
    sam_workdir = Path(sam_workdir)

    if str(sam_workdir) not in sys.path:
        sys.path.insert(0, str(sam_workdir))

    from sam2.build_sam import build_sam2_camera_predictor

    predictor = build_sam2_camera_predictor(
        "sam2.1/sam2.1_hiera_t.yaml",
        str(checkpoint),
        device="cuda",
    )

    predictor.fill_hole_area = 0
    return predictor


def initialize_predictor(
    predictor,
    frame: np.ndarray,
    point: Optional[Tuple[int, int]] = None,
    bbox: Optional[BBox] = None,
):
    h, w = frame.shape[:2]

    predictor.frame_idx = 0
    predictor.load_first_frame(frame)

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        box = np.array(
            [
                np.clip(x1, 0, w - 1),
                np.clip(y1, 0, h - 1),
                np.clip(x2, 1, w),
                np.clip(y2, 1, h),
            ],
            dtype=np.float32,
        )

        _, _, logits = predictor.add_new_prompt(
            frame_idx=0,
            obj_id=1,
            bbox=box,
        )
        return logits

    if point is None:
        raise ValueError("Either point or bbox must be provided.")

    x = int(np.clip(point[0], 0, w - 1))
    y = int(np.clip(point[1], 0, h - 1))

    _, _, logits = predictor.add_new_prompt(
        frame_idx=0,
        obj_id=1,
        points=np.array([[x, y]], dtype=np.float32),
        labels=np.array([1], dtype=np.int32),
    )

    return logits


def release_predictor(predictor):
    if predictor is None:
        return

    try:
        predictor.condition_state = {}
        predictor.frame_idx = 0
    except Exception:
        pass

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def write_metrics(metrics: Dict, json_path: Path, csv_path: Path):
    json_path.write_text(json.dumps(metrics, indent=2))

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])

        for key, value in metrics.items():
            writer.writerow([key, value])


def _bbox_area(bbox: BBox):
    bw, bh = bbox_wh(bbox)
    return max(1, bw * bh)


def _candidate_consistent(candidate: BBox, sam_bbox: BBox):
    ccx, ccy = bbox_center(candidate)
    scx, scy = bbox_center(sam_bbox)

    cw, ch = bbox_wh(candidate)
    sw, sh = bbox_wh(sam_bbox)

    center_distance = np.hypot(ccx - scx, ccy - scy)
    center_ok = center_distance <= max(20.0, 1.5 * np.hypot(cw, ch))

    area_ratio = (sw * sh) / max(1.0, cw * ch)
    size_ok = 0.20 <= area_ratio <= 5.0

    return center_ok and size_ok


def track_with_semantic_reacquisition(
    frames_dir,
    target_point: Tuple[int, int],
    fps,
    output_path,
    metrics_json,
    metrics_csv,
    sam_workdir,
    checkpoint,
    max_side=960,
    progress: ProgressCB = None,
):
    """
    Persistent single-object tracking:

    1. SAM2 performs local mask tracking while the lock is reliable.
    2. A deep target verifier checks identity independently of SAM2 confidence.
    3. If identity/mask quality fails, SAM2 is considered lost.
    4. A ResNet feature-map re-detector scans the entire frame.
    5. A candidate must survive temporal confirmation.
    6. SAM2 is reinitialized from the recovered bounding box.

    This follows the local-tracker + global-redetector design used by
    long-term tracking systems, rather than asking SAM2 memory to recover itself.
    """

    paths = sorted(Path(frames_dir).glob("frame_*.png"))

    if not paths:
        raise RuntimeError("No RGB frames to track.")

    frames = []

    for path in paths:
        frame = cv2.imread(str(path))

        if frame is not None:
            frames.append(fit_frame(frame, max_side=max_side))

    if not frames:
        raise RuntimeError("No valid RGB frames.")

    h, w = frames[0].shape[:2]

    frames = [
        frame
        if frame.shape[:2] == (h, w)
        else cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        for frame in frames
    ]

    output_path = Path(output_path)
    raw_path = output_path.with_name(output_path.stem + "_raw.mp4")

    writer = cv2.VideoWriter(
        str(raw_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (w, h),
    )

    if not writer.isOpened():
        raise RuntimeError("Could not create output video.")

    redetector = GlobalTargetRedetector(
        device="cuda",
        search_max_side=min(768, max_side),
        topk=24,
        confirm_score=0.52,
        confirm_frames=2,
    )

    amp_dtype = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )

    predictor = None

    previous_area = None
    initial_mask_area = 1
    initial_bbox_area = 1

    locked = True
    lost_since = None

    loss_events = 0
    global_search_frames = 0
    confirmed_reacquisitions = 0
    failed_reacquisitions = 0

    verifier_scores = []
    sam_confidences = []
    global_best_scores = []
    center_jumps = []

    last_center = None

    start = time.perf_counter()

    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=amp_dtype):
            predictor = build_camera_predictor(sam_workdir, checkpoint)

            logits = initialize_predictor(
                predictor,
                frames[0],
                point=target_point,
            )

            mask, confidence = mask_from_logits(logits, w, h)
            bbox = bbox_from_mask(mask)

            if bbox is None:
                x, y = target_point
                bbox = (
                    max(0, int(x) - 12),
                    max(0, int(y) - 12),
                    min(w, int(x) + 12),
                    min(h, int(y) + 12),
                )

            initial_mask_area = max(1, int(mask.sum()))
            initial_bbox_area = _bbox_area(bbox)
            previous_area = initial_mask_area

            redetector.initialize(frames[0], bbox)

            for idx, frame in enumerate(frames):
                reacquired_now = False
                search_detections = []

                if idx == 0:
                    current_mask = mask
                    current_confidence = confidence
                    current_bbox = bbox

                elif locked:
                    _, logits = predictor.track(frame)
                    current_mask, current_confidence = mask_from_logits(
                        logits,
                        w,
                        h,
                    )

                    good_mask, current_area = mask_quality(
                        current_mask,
                        current_confidence,
                        previous_area,
                        initial_mask_area,
                    )

                    current_bbox = (
                        bbox_from_mask(current_mask)
                        if good_mask
                        else None
                    )

                    verifier_score = 0.0

                    if current_bbox is not None:
                        verifier_score = redetector.verify_bbox(
                            frame,
                            current_bbox,
                        )

                        verifier_scores.append(verifier_score)

                        bbox_area = _bbox_area(current_bbox)
                        bbox_growth = bbox_area / float(initial_bbox_area)
                        frame_fraction = bbox_area / float(max(1, w * h))

                        identity_failed = verifier_score < 0.28
                        geometry_failed = (
                            bbox_growth > 7.0
                            or frame_fraction > 0.10
                        )

                        if identity_failed or geometry_failed:
                            good_mask = False
                            current_bbox = None

                    if good_mask and current_bbox is not None:
                        previous_area = current_area
                        sam_confidences.append(current_confidence)

                        redetector.update_trusted(
                            frame,
                            current_bbox,
                            verifier_score,
                        )

                    else:
                        locked = False
                        lost_since = idx
                        loss_events += 1
                        redetector.reset_confirmation()

                        current_mask = np.zeros(
                            (h, w),
                            dtype=bool,
                        )
                        current_bbox = None
                        current_confidence = 0.0

                else:
                    current_mask = np.zeros(
                        (h, w),
                        dtype=bool,
                    )
                    current_bbox = None
                    current_confidence = 0.0

                if not locked:
                    global_search_frames += 1

                    search_detections = redetector.search(frame)

                    if search_detections:
                        global_best_scores.append(
                            search_detections[0].score
                        )

                    confirmed = redetector.confirm(
                        search_detections
                    )

                    if confirmed is not None:
                        release_predictor(predictor)

                        predictor = build_camera_predictor(
                            sam_workdir,
                            checkpoint,
                        )

                        logits2 = initialize_predictor(
                            predictor,
                            frame,
                            bbox=confirmed.bbox,
                        )

                        mask2, confidence2 = mask_from_logits(
                            logits2,
                            w,
                            h,
                        )

                        good2, area2 = mask_quality(
                            mask2,
                            confidence2,
                            None,
                            initial_mask_area,
                        )

                        bbox2 = (
                            bbox_from_mask(mask2)
                            if good2
                            else None
                        )

                        verifier2 = (
                            redetector.verify_bbox(frame, bbox2)
                            if bbox2 is not None
                            else -1.0
                        )

                        confirmed_ok = (
                            good2
                            and bbox2 is not None
                            and verifier2 >= 0.25
                            and _candidate_consistent(
                                confirmed.bbox,
                                bbox2,
                            )
                        )

                        if confirmed_ok:
                            locked = True
                            reacquired_now = True
                            confirmed_reacquisitions += 1

                            current_mask = mask2
                            current_bbox = bbox2
                            current_confidence = confidence2
                            previous_area = area2

                            sam_confidences.append(confidence2)
                            verifier_scores.append(verifier2)

                            redetector.update_trusted(
                                frame,
                                bbox2,
                                max(verifier2, 0.70),
                            )
                            redetector.last_good_bbox = bbox2
                            redetector.reset_confirmation()

                        else:
                            failed_reacquisitions += 1
                            release_predictor(predictor)
                            predictor = None

                result = frame.copy()

                if locked and current_bbox is not None:
                    overlay = result.copy()
                    overlay[current_mask] = (255, 255, 255)

                    result = cv2.addWeighted(
                        result,
                        0.78,
                        overlay,
                        0.22,
                        0,
                    )

                    x1, y1, x2, y2 = current_bbox

                    cv2.rectangle(
                        result,
                        (x1, y1),
                        (x2, y2),
                        (255, 255, 255),
                        2,
                    )

                    status = (
                        "RE-ACQUIRED"
                        if reacquired_now
                        else "LOCKED"
                    )

                    cv2.putText(
                        result,
                        f"MODEL 4 | {status}",
                        (x1, max(24, y1 - 8)),
                        cv2.FONT_HERSHEY_DUPLEX,
                        0.55,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                    center = bbox_center(current_bbox)

                    if last_center is not None:
                        diag = max(1.0, np.hypot(w, h))

                        center_jumps.append(
                            float(
                                np.hypot(
                                    center[0] - last_center[0],
                                    center[1] - last_center[1],
                                )
                                / diag
                            )
                        )

                    last_center = center

                else:
                    cv2.putText(
                        result,
                        "MODEL 4 | GLOBAL RE-DETECTION",
                        (18, 30),
                        cv2.FONT_HERSHEY_DUPLEX,
                        0.65,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                    if lost_since is not None:
                        cv2.putText(
                            result,
                            f"lost for {idx - lost_since + 1} frame(s)",
                            (18, 55),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.48,
                            (220, 220, 220),
                            1,
                            cv2.LINE_AA,
                        )

                    for rank, det in enumerate(search_detections[:5]):
                        x1, y1, x2, y2 = det.bbox

                        cv2.rectangle(
                            result,
                            (x1, y1),
                            (x2, y2),
                            (255, 255, 255),
                            2 if rank == 0 else 1,
                        )

                        cv2.putText(
                            result,
                            f"G{rank + 1}:{det.score:.2f}",
                            (x1, max(16, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.42,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA,
                        )

                writer.write(result)

                if progress:
                    progress(
                        "track",
                        idx + 1,
                        len(frames),
                    )

    finally:
        writer.release()
        release_predictor(predictor)

    elapsed = max(
        1e-6,
        time.perf_counter() - start,
    )

    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-i",
            str(raw_path),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "22",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    raw_path.unlink(missing_ok=True)

    total = len(frames)

    metrics = {
        "tracker": "SAM2 + SigLIP 2 + Motion-Aware Memory",
        "total_frames": total,
        "loss_events": loss_events,
        "global_search_frames": global_search_frames,
        "confirmed_reacquisitions": confirmed_reacquisitions,
        "failed_reacquisition_checks": failed_reacquisitions,
        "mean_global_best_score": round(
            float(np.mean(global_best_scores))
            if global_best_scores
            else 0.0,
            4,
        ),
        "mean_identity_verifier_score": round(
            float(np.mean(verifier_scores))
            if verifier_scores
            else 0.0,
            4,
        ),
        "mean_sam_confidence_locked_frames": round(
            float(np.mean(sam_confidences))
            if sam_confidences
            else 0.0,
            4,
        ),
        "mean_normalized_center_jump": round(
            float(np.mean(center_jumps))
            if center_jumps
            else 0.0,
            6,
        ),
        "tracking_fps": round(
            total / elapsed,
            3,
        ),
        "evaluation_type": (
            "No-GT diagnostics only. "
            "These values are not tracking accuracy metrics."
        ),
    }

    write_metrics(
        metrics,
        Path(metrics_json),
        Path(metrics_csv),
    )

    return output_path, metrics
