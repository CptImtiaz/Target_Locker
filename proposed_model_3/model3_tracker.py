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

from memory_tree import MemoryTreeReacquirer, bbox_center


ProgressCB = Optional[Callable[[str, int, int], None]]


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
    mask = np.squeeze(prob > 0.5)
    prob = np.squeeze(prob)

    if mask.shape != (height, width):
        mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
        prob = cv2.resize(prob.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)

    if mask.any():
        conf = float(np.clip(prob[mask].mean(), 0.0, 1.0))
    else:
        conf = float(np.clip(prob.max() if prob.size else 0.0, 0.0, 1.0))

    return mask, conf


def bbox_from_mask(mask: np.ndarray):
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def mask_quality(mask: np.ndarray, confidence: float, previous_area: Optional[int], initial_area: int):
    area = int(mask.sum())
    if area <= 0:
        return False, area

    min_area = max(4, int(initial_area * 0.03))
    if area < min_area:
        return False, area

    if previous_area and previous_area > 0:
        ratio = area / float(previous_area)
        if ratio < 0.12 or ratio > 8.0:
            return False, area

    if confidence < 0.50:
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


def initialize_predictor(predictor, frame: np.ndarray, point: Tuple[int, int]):
    h, w = frame.shape[:2]
    x = int(np.clip(point[0], 0, w - 1))
    y = int(np.clip(point[1], 0, h - 1))

    predictor.frame_idx = 0
    predictor.load_first_frame(frame)

    _, _, logits = predictor.add_new_prompt(
        frame_idx=0,
        obj_id=(1,),
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


def track_with_memory_tree(
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
        f if f.shape[:2] == (h, w) else cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA)
        for f in frames
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
        raise RuntimeError("Could not create Model 3 output.")

    tree = MemoryTreeReacquirer(
        max_memories=12,
        beam_width=5,
        candidates_per_template=4,
        confirm_frames=2,
        confirm_score=0.62,
    )

    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    predictor = None
    previous_area = None
    initial_area = 1
    last_center = None
    confidence_values = []
    center_jumps = []
    tree_scores = []

    loss_events = 0
    search_frames = 0
    confirmed_reacquisitions = 0
    was_locked = True

    start = time.perf_counter()

    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=amp_dtype):
            predictor = build_camera_predictor(sam_workdir, checkpoint)

            logits = initialize_predictor(predictor, frames[0], target_point)
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

            initial_area = max(1, int(mask.sum()))
            previous_area = initial_area
            tree.update_locked(frames[0], bbox)

            for idx, frame in enumerate(frames):
                reacquired = False
                branches = []

                if idx > 0 and was_locked:
                    _, logits = predictor.track(frame)
                    mask, confidence = mask_from_logits(logits, w, h)
                elif idx == 0:
                    pass
                else:
                    mask = np.zeros((h, w), dtype=bool)
                    confidence = 0.0

                good, area = mask_quality(mask, confidence, previous_area, initial_area)
                bbox = bbox_from_mask(mask) if good else None

                if good and bbox is not None:
                    if not was_locked:
                        was_locked = True
                    previous_area = area
                    confidence_values.append(confidence)
                    tree.update_locked(frame, bbox)

                else:
                    if was_locked:
                        loss_events += 1
                        was_locked = False
                        tree.reset_search()

                    search_frames += 1
                    candidate, branches = tree.step(frame)

                    if branches:
                        tree_scores.append(branches[0].cumulative_score)

                    if candidate is not None:
                        release_predictor(predictor)
                        predictor = build_camera_predictor(sam_workdir, checkpoint)
                        logits = initialize_predictor(predictor, frame, candidate.center)
                        mask2, confidence2 = mask_from_logits(logits, w, h)
                        good2, area2 = mask_quality(mask2, confidence2, None, initial_area)
                        bbox2 = bbox_from_mask(mask2) if good2 else None

                        if good2 and bbox2 is not None:
                            mask = mask2
                            confidence = confidence2
                            area = area2
                            bbox = bbox2
                            was_locked = True
                            reacquired = True
                            confirmed_reacquisitions += 1
                            previous_area = area2
                            confidence_values.append(confidence2)
                            tree.update_locked(frame, bbox2)

                result = frame.copy()

                if was_locked and bbox is not None:
                    overlay = result.copy()
                    overlay[mask] = (255, 255, 255)
                    result = cv2.addWeighted(result, 0.76, overlay, 0.24, 0)

                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(result, (x1, y1), (x2, y2), (255, 255, 255), 2)
                    status = "RE-ACQUIRED" if reacquired else "LOCKED"
                    cv2.putText(
                        result,
                        f"MODEL 3 | {status}",
                        (x1, max(24, y1 - 8)),
                        cv2.FONT_HERSHEY_DUPLEX,
                        0.55,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                    center = bbox_center(bbox)
                    if last_center is not None:
                        diag = max(1.0, np.hypot(w, h))
                        center_jumps.append(
                            float(
                                np.hypot(
                                    center[0] - last_center[0],
                                    center[1] - last_center[1],
                                ) / diag
                            )
                        )
                    last_center = center

                else:
                    cv2.putText(
                        result,
                        "MODEL 3 | MEMORY TREE SEARCH",
                        (18, 30),
                        cv2.FONT_HERSHEY_DUPLEX,
                        0.65,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                    for rank, node in enumerate(branches[:5]):
                        x1, y1, x2, y2 = node.bbox
                        thickness = 2 if rank == 0 else 1
                        cv2.rectangle(result, (x1, y1), (x2, y2), (255, 255, 255), thickness)
                        cv2.putText(
                            result,
                            f"B{rank + 1}:{node.cumulative_score:.2f}",
                            (x1, max(16, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.40,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA,
                        )

                writer.write(result)

                if progress:
                    progress("track", idx + 1, len(frames))

    finally:
        writer.release()
        release_predictor(predictor)

    elapsed = max(1e-6, time.perf_counter() - start)

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
        "tracker": "SAM2 + Memory Tree Re-acquisition",
        "total_frames": total,
        "loss_events": loss_events,
        "memory_tree_search_frames": search_frames,
        "confirmed_reacquisitions": confirmed_reacquisitions,
        "mean_tree_best_score": round(float(np.mean(tree_scores)) if tree_scores else 0.0, 4),
        "mean_sam_confidence_locked_frames": round(float(np.mean(confidence_values)) if confidence_values else 0.0, 4),
        "mean_normalized_center_jump": round(float(np.mean(center_jumps)) if center_jumps else 0.0, 6),
        "tracking_fps": round(total / elapsed, 3),
        "evaluation_type": "No-GT diagnostics only. These are not tracking accuracy metrics.",
    }

    metrics_json = Path(metrics_json)
    metrics_csv = Path(metrics_csv)
    write_metrics(metrics, metrics_json, metrics_csv)

    return output_path, metrics
