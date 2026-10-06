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

from reacquisition_engine import ReacquisitionEngine, bbox_center


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


def run_thera(runtime_script, weights_dir, rgb_frames_dir, thermal_frames_dir, palette="SUNNY", steps=8):
    runtime_script = Path(runtime_script)
    weights_dir = Path(weights_dir)
    rgb_frames_dir = Path(rgb_frames_dir)
    thermal_frames_dir = Path(thermal_frames_dir)

    if thermal_frames_dir.exists():
        shutil.rmtree(thermal_frames_dir)
    thermal_frames_dir.mkdir(parents=True, exist_ok=True)

    cache = weights_dir / "palettes" / f"{palette.upper()}.pt"
    if not cache.exists():
        raise FileNotFoundError(f"Thermal condition file not found: {cache}")

    cmd = [
        sys.executable,
        str(runtime_script),
        "--weights-dir", str(weights_dir),
        "--rgb-dir", str(rgb_frames_dir),
        "--output-dir", str(thermal_frames_dir),
        "--reference-cache", str(cache),
        "--num-steps", str(int(steps)),
        "--device", "cuda",
    ]
    proc = subprocess.run(cmd, cwd=str(runtime_script.parent), text=True, capture_output=True)
    if proc.stdout:
        print(proc.stdout)
    if proc.returncode != 0:
        if proc.stderr:
            print(proc.stderr)
        raise RuntimeError(f"Thermal conversion failed.\n\n{proc.stderr or '(no stderr)'}")

    produced = sorted(thermal_frames_dir.glob("frame_*.png"))
    expected = sorted(rgb_frames_dir.glob("frame_*.png"))
    if len(produced) != len(expected):
        raise RuntimeError(f"Thermal output mismatch: expected {len(expected)}, got {len(produced)}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def frames_to_video(frames_dir, output_path, fps):
    frames = sorted(Path(frames_dir).glob("frame_*.png"))
    if not frames:
        raise RuntimeError(f"No frames found in {frames_dir}")

    first = cv2.imread(str(frames[0]))
    if first is None:
        raise RuntimeError(f"Could not read {frames[0]}")
    h, w = first.shape[:2]

    output_path = Path(output_path)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError("Could not create thermal video.")

    for path in frames:
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        writer.write(frame)
    writer.release()
    return output_path


def first_frame(frames_dir):
    paths = sorted(Path(frames_dir).glob("frame_*.png"))
    if not paths:
        raise RuntimeError("No thermal frames found.")
    frame = cv2.imread(str(paths[0]))
    if frame is None:
        raise RuntimeError("Could not read first thermal frame.")
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
    del predictor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def write_metrics(metrics: Dict, json_path: Path, csv_path: Path):
    json_path.write_text(json.dumps(metrics, indent=2))

    rows = [
        ("algorithm", metrics["algorithm"]),
        ("total_frames", metrics["total_frames"]),
        ("locked_frames", metrics["locked_frames"]),
        ("lock_retention_percent", metrics["lock_retention_percent"]),
        ("loss_events", metrics["loss_events"]),
        ("reacquisition_attempts", metrics["reacquisition_attempts"]),
        ("successful_reacquisitions", metrics["successful_reacquisitions"]),
        ("reacquisition_success_percent", metrics["reacquisition_success_percent"]),
        ("mean_reacquisition_latency_frames", metrics["mean_reacquisition_latency_frames"]),
        ("mean_sam_confidence", metrics["mean_sam_confidence"]),
        ("mean_normalized_center_jump", metrics["mean_normalized_center_jump"]),
        ("tracking_fps", metrics["tracking_fps"]),
        ("evaluation_type", metrics["evaluation_type"]),
    ]

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerows(rows)


def track_with_reacquisition(
    frames_dir,
    target_point: Tuple[int, int],
    fps,
    output_path,
    metrics_json,
    metrics_csv,
    sam_workdir,
    checkpoint,
    algorithm="auto_ensemble",
    max_side=960,
    recovery_threshold=0.38,
    progress: ProgressCB = None,
):
    paths = sorted(Path(frames_dir).glob("frame_*.png"))
    if not paths:
        raise RuntimeError("No thermal frames to track.")

    frames = []
    for path in paths:
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        frame = fit_frame(frame, max_side=max_side)
        frames.append(frame)

    if not frames:
        raise RuntimeError("No valid thermal frames.")

    h, w = frames[0].shape[:2]
    frames = [
        f if f.shape[:2] == (h, w) else cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA)
        for f in frames
    ]

    output_path = Path(output_path)
    raw_path = output_path.with_name(output_path.stem + "_raw.mp4")
    writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError("Could not create Model 3 output.")

    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    predictor = None

    locked_frames = 0
    loss_events = 0
    reacq_attempts = 0
    reacq_success = 0
    reacq_latencies = []
    confidence_values = []
    normalized_jumps = []
    last_center = None
    previous_area = None
    initial_area = 1
    was_locked = True
    current_loss_length = 0
    engine = None
    initial_bbox = None
    start = time.perf_counter()

    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=amp_dtype):
            predictor = build_camera_predictor(sam_workdir, checkpoint)
            logits = initialize_predictor(predictor, frames[0], target_point)
            mask, confidence = mask_from_logits(logits, w, h)
            initial_bbox = bbox_from_mask(mask)

            if initial_bbox is None:
                x, y = target_point
                initial_bbox = (
                    max(0, int(x) - 12),
                    max(0, int(y) - 12),
                    min(w, int(x) + 12),
                    min(h, int(y) + 12),
                )

            initial_area = max(1, int(mask.sum()))
            previous_area = initial_area
            engine = ReacquisitionEngine(frames[0], initial_bbox)

            for idx, frame in enumerate(frames):
                recovered_this_frame = False

                if idx > 0:
                    _, logits = predictor.track(frame)
                    mask, confidence = mask_from_logits(logits, w, h)

                good, area = mask_quality(mask, confidence, previous_area, initial_area)
                bbox = bbox_from_mask(mask) if good else None

                if not good:
                    if was_locked:
                        loss_events += 1
                        current_loss_length = 0
                    was_locked = False
                    current_loss_length += 1
                    reacq_attempts += 1

                    candidate = engine.reacquire(algorithm, idx, frames)

                    if candidate is not None and candidate.score >= recovery_threshold:
                        release_predictor(predictor)
                        predictor = build_camera_predictor(sam_workdir, checkpoint)
                        logits = initialize_predictor(predictor, frame, candidate.center)
                        mask, confidence = mask_from_logits(logits, w, h)
                        good2, area2 = mask_quality(mask, confidence, None, initial_area)
                        bbox2 = bbox_from_mask(mask) if good2 else None

                        if good2 and bbox2 is not None:
                            good = True
                            area = area2
                            bbox = bbox2
                            recovered_this_frame = True
                            reacq_success += 1
                            reacq_latencies.append(current_loss_length)
                            current_loss_length = 0
                            was_locked = True
                            engine.update_lock(frame, bbox)

                if good and bbox is not None:
                    locked_frames += 1
                    confidence_values.append(confidence)
                    previous_area = area

                    if not recovered_this_frame:
                        engine.update_lock(frame, bbox)

                    center = bbox_center(bbox)
                    if last_center is not None:
                        diag = max(1.0, np.hypot(w, h))
                        normalized_jumps.append(float(np.hypot(center[0] - last_center[0], center[1] - last_center[1]) / diag))
                    last_center = center

                result = frame.copy()

                if good and bbox is not None:
                    overlay = result.copy()
                    overlay[mask] = (255, 255, 255)
                    result = cv2.addWeighted(result, 0.74, overlay, 0.26, 0)

                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(result, (x1, y1), (x2, y2), (255, 255, 255), 2)
                    status = "RE-ACQUIRED" if recovered_this_frame else "LOCKED"
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
                else:
                    cv2.putText(
                        result,
                        "MODEL 3 | SEARCHING",
                        (18, 30),
                        cv2.FONT_HERSHEY_DUPLEX,
                        0.65,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                cv2.putText(
                    result,
                    f"{ReacquisitionEngine.METHODS.get(algorithm, algorithm)} | conf {confidence:.2f}",
                    (18, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
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
            "ffmpeg", "-nostdin", "-y", "-i", str(raw_path),
            "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "22",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    raw_path.unlink(missing_ok=True)

    total = len(frames)
    metrics = {
        "algorithm": ReacquisitionEngine.METHODS.get(algorithm, algorithm),
        "algorithm_key": algorithm,
        "total_frames": total,
        "locked_frames": locked_frames,
        "lock_retention_percent": round(100.0 * locked_frames / max(total, 1), 2),
        "loss_events": loss_events,
        "reacquisition_attempts": reacq_attempts,
        "successful_reacquisitions": reacq_success,
        "reacquisition_success_percent": round(100.0 * reacq_success / max(reacq_attempts, 1), 2),
        "mean_reacquisition_latency_frames": round(float(np.mean(reacq_latencies)) if reacq_latencies else 0.0, 3),
        "mean_sam_confidence": round(float(np.mean(confidence_values)) if confidence_values else 0.0, 4),
        "mean_normalized_center_jump": round(float(np.mean(normalized_jumps)) if normalized_jumps else 0.0, 6),
        "tracking_fps": round(total / elapsed, 3),
        "evaluation_type": "No-GT proxy evaluation; use benchmark ground truth for accuracy/AUC/precision.",
    }

    metrics_json = Path(metrics_json)
    metrics_csv = Path(metrics_csv)
    write_metrics(metrics, metrics_json, metrics_csv)

    return output_path, metrics
