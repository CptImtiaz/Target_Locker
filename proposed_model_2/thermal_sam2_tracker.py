from __future__ import annotations

import gc
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional, Tuple

import cv2
import numpy as np
import torch

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
        "fps": fps, "frames": idx, "width": out_w, "height": out_h,
        "source_width": src_w, "source_height": src_h,
    }


def run_thera(thera_dir, rgb_frames_dir, thermal_frames_dir, palette="SUNNY", steps=20):
    """Run official TherA RGB->TIR in a child process, then release its GPU memory."""
    thera_dir = Path(thera_dir)
    rgb_frames_dir = Path(rgb_frames_dir)
    thermal_frames_dir = Path(thermal_frames_dir)

    if thermal_frames_dir.exists():
        shutil.rmtree(thermal_frames_dir)
    thermal_frames_dir.mkdir(parents=True, exist_ok=True)

    cache = thera_dir / "weights" / "reference_caches" / f"{palette.upper()}.pt"
    if not cache.exists():
        raise FileNotFoundError(f"TherA reference cache not found: {cache}")

    cmd = [
        sys.executable, str(thera_dir / "infer_custom.py"),
        "--rgb-dir", str(rgb_frames_dir),
        "--output-dir", str(thermal_frames_dir),
        "--reference-cache", str(cache),
        "--num-steps", str(int(steps)),
        "--batch-size", "1",
        "--device", "cuda",
    ]
    subprocess.run(cmd, cwd=str(thera_dir), check=True)

    produced = sorted(thermal_frames_dir.glob("frame_*.png"))
    expected = sorted(rgb_frames_dir.glob("frame_*.png"))
    if len(produced) != len(expected):
        raise RuntimeError(f"TherA output mismatch: expected {len(expected)}, got {len(produced)}")

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
    frames = sorted(Path(frames_dir).glob("frame_*.png"))
    if not frames:
        raise RuntimeError("No thermal frames found.")
    frame = cv2.imread(str(frames[0]))
    if frame is None:
        raise RuntimeError("Could not read first thermal frame.")
    return frame


def track_thermal_frames_sam2(
    frames_dir,
    target_point: Tuple[int, int],
    fps,
    output_path,
    sam_workdir,
    checkpoint,
    max_side=960,
    progress: ProgressCB = None,
):
    """Track a mouse-selected object through the synthetic thermal frames with SAM2.1."""
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

    paths = sorted(Path(frames_dir).glob("frame_*.png"))
    if not paths:
        raise RuntimeError("No thermal frames to track.")

    frame0 = cv2.imread(str(paths[0]))
    frame0 = fit_frame(frame0, max_side=max_side)
    h, w = frame0.shape[:2]

    x = int(np.clip(target_point[0], 0, w - 1))
    y = int(np.clip(target_point[1], 0, h - 1))

    output_path = Path(output_path)
    raw_path = output_path.with_name(output_path.stem + "_raw.mp4")
    writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError("Could not create tracking output.")

    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=amp_dtype):
            predictor.frame_idx = 0
            predictor.load_first_frame(frame0)
            _, _, logits = predictor.add_new_prompt(
                frame_idx=0,
                obj_id=(1,),
                points=np.array([[x, y]], dtype=np.float32),
                labels=np.array([1], dtype=np.int32),
            )

            for idx, path in enumerate(paths):
                if idx == 0:
                    frame = frame0
                else:
                    frame = cv2.imread(str(path))
                    if frame is None:
                        continue
                    frame = fit_frame(frame, max_side=max_side)
                    if frame.shape[:2] != (h, w):
                        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
                    _, logits = predictor.track(frame)

                mask = (logits[0] > 0.0).detach().cpu().numpy()
                mask = np.squeeze(mask).astype(bool)
                if mask.shape != frame.shape[:2]:
                    mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)

                result = frame.copy()
                if mask.any():
                    overlay = result.copy()
                    overlay[mask] = (255, 255, 255)
                    result = cv2.addWeighted(result, 0.72, overlay, 0.28, 0)

                    ys, xs = np.where(mask)
                    x1, x2 = int(xs.min()), int(xs.max())
                    y1, y2 = int(ys.min()), int(ys.max())
                    cv2.rectangle(result, (x1, y1), (x2, y2), (255, 255, 255), 2)
                    cv2.putText(
                        result, "THERMAL TARGET LOCK",
                        (x1, max(26, y1 - 8)),
                        cv2.FONT_HERSHEY_DUPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA,
                    )

                writer.write(result)
                if progress:
                    progress("track", idx + 1, len(paths))
    finally:
        writer.release()
        predictor.condition_state = {}
        predictor.frame_idx = 0
        del predictor
        gc.collect()
        torch.cuda.empty_cache()

    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-y", "-i", str(raw_path),
            "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "22",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output_path),
        ],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    raw_path.unlink(missing_ok=True)
    return output_path
