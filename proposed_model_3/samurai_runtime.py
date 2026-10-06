from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch


def determine_model_cfg(model_path: str) -> str:
    name = Path(model_path).name.lower()
    if "large" in name:
        return "configs/samurai/sam2.1_hiera_l.yaml"
    if "base_plus" in name or "base+" in name:
        return "configs/samurai/sam2.1_hiera_b+.yaml"
    if "small" in name:
        return "configs/samurai/sam2.1_hiera_s.yaml"
    return "configs/samurai/sam2.1_hiera_t.yaml"


def bbox_from_mask(mask: np.ndarray):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samurai-root", required=True)
    ap.add_argument("--video-path", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--x", required=True, type=float)
    ap.add_argument("--y", required=True, type=float)
    ap.add_argument("--output-path", required=True)
    ap.add_argument("--metrics-json", required=True)
    args = ap.parse_args()

    samurai_root = Path(args.samurai_root)
    sam2_root = samurai_root / "sam2"
    sys.path.insert(0, str(sam2_root))
    os.chdir(samurai_root)

    from sam2.build_sam import build_sam2_video_predictor

    model_cfg = determine_model_cfg(args.model_path)
    predictor = build_sam2_video_predictor(
        model_cfg,
        args.model_path,
        device="cuda:0",
    )

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 25.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError("No frames decoded from input video.")

    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        args.output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    if not writer.isOpened():
        raise RuntimeError("Could not create SAMURAI output video.")

    visible_frames = 0
    zero_mask_frames = 0
    box_areas = []
    center_jumps = []
    prev_center = None
    start = time.perf_counter()

    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=amp_dtype):
            state = predictor.init_state(args.video_path, offload_video_to_cpu=True)

            point = np.array([[args.x, args.y]], dtype=np.float32)
            label = np.array([1], dtype=np.int32)

            predictor.add_new_points_or_box(
                state,
                frame_idx=0,
                obj_id=0,
                points=point,
                labels=label,
            )

            for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
                frame = frames[frame_idx].copy()
                mask = masks[0][0].detach().cpu().numpy() > 0.0
                bbox = bbox_from_mask(mask)

                if bbox is None:
                    zero_mask_frames += 1
                    cv2.putText(
                        frame,
                        "SAMURAI | NO MASK",
                        (18, 30),
                        cv2.FONT_HERSHEY_DUPLEX,
                        0.65,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                else:
                    visible_frames += 1
                    x1, y1, x2, y2 = bbox
                    box_areas.append((x2 - x1) * (y2 - y1))
                    center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                    if prev_center is not None:
                        diag = max(1.0, float(np.hypot(w, h)))
                        center_jumps.append(
                            float(np.hypot(center[0] - prev_center[0], center[1] - prev_center[1]) / diag)
                        )
                    prev_center = center

                    overlay = frame.copy()
                    overlay[mask] = (255, 255, 255)
                    frame = cv2.addWeighted(frame, 0.78, overlay, 0.22, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                    cv2.putText(
                        frame,
                        "SAMURAI | MOTION-AWARE MEMORY",
                        (x1, max(24, y1 - 8)),
                        cv2.FONT_HERSHEY_DUPLEX,
                        0.52,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                writer.write(frame)

    finally:
        writer.release()
        try:
            del predictor
            del state
        except Exception:
            pass
        gc.collect()
        torch.cuda.empty_cache()

    elapsed = max(1e-6, time.perf_counter() - start)
    total = len(frames)

    metrics = {
        "tracker": "SAMURAI (SAM2.1 motion-aware memory)",
        "total_frames": total,
        "frames_with_mask": visible_frames,
        "zero_mask_frames": zero_mask_frames,
        "mask_presence_percent": round(100.0 * visible_frames / max(total, 1), 2),
        "mean_bbox_area": round(float(np.mean(box_areas)) if box_areas else 0.0, 3),
        "mean_normalized_center_jump": round(float(np.mean(center_jumps)) if center_jumps else 0.0, 6),
        "tracking_fps": round(total / elapsed, 3),
        "evaluation_type": "No-GT diagnostics only. These values are not tracking accuracy metrics.",
    }
    Path(args.metrics_json).write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
