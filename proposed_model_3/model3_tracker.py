from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional, Tuple

import cv2
import numpy as np


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


def extract_rgb_frames(
    video_path,
    frames_dir,
    max_side=768,
    progress: ProgressCB = None,
):
    video_path = Path(video_path)
    frames_dir = Path(frames_dir)

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


def run_samurai(
    video_path,
    target_point: Tuple[int, int],
    output_path,
    metrics_json,
    samurai_root,
    checkpoint,
):
    """
    Run the official SAMURAI tracker in an isolated child process.

    This avoids Python module conflicts between the SAM2 copy bundled in
    Target_Locker and SAMURAI's modified SAM2 implementation.
    """
    video_path = Path(video_path)
    output_path = Path(output_path)
    metrics_json = Path(metrics_json)
    samurai_root = Path(samurai_root)
    checkpoint = Path(checkpoint)

    runtime = Path(__file__).with_name("samurai_runtime.py")

    if not runtime.exists():
        raise FileNotFoundError(f"SAMURAI runtime wrapper not found: {runtime}")

    if not samurai_root.exists():
        raise FileNotFoundError(
            f"SAMURAI repository not found at {samurai_root}. "
            "Run the Model 3 setup cell first."
        )

    if not checkpoint.exists():
        raise FileNotFoundError(f"SAM2.1 checkpoint not found: {checkpoint}")

    x, y = target_point

    cmd = [
        sys.executable,
        str(runtime),
        "--samurai-root", str(samurai_root),
        "--video-path", str(video_path),
        "--model-path", str(checkpoint),
        "--x", str(float(x)),
        "--y", str(float(y)),
        "--output-path", str(output_path),
        "--metrics-json", str(metrics_json),
    ]

    proc = subprocess.run(
        cmd,
        cwd=str(samurai_root),
        text=True,
        capture_output=True,
    )

    if proc.stdout:
        print(proc.stdout)

    if proc.returncode != 0:
        if proc.stderr:
            print(proc.stderr)
        raise RuntimeError(
            "SAMURAI tracking failed.\n\n"
            f"STDERR:\n{proc.stderr or '(no stderr)'}"
        )

    if not output_path.exists():
        raise RuntimeError("SAMURAI completed but no output video was created.")

    if not metrics_json.exists():
        raise RuntimeError("SAMURAI completed but diagnostics JSON was not created.")

    metrics = json.loads(metrics_json.read_text())
    return output_path, metrics
