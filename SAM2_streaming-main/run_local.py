"""Local CPU adaptation of khw11044/SAM2_streaming for macOS.

Click on the frozen first frame, Enter to start, Q to quit.
"""
import argparse
from pathlib import Path
import urllib.request

import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2_camera_predictor

ROOT = Path(__file__).resolve().parent
WINDOW = "Target Locker"


def fit(frame):
    h, w = frame.shape[:2]
    ratio = min(1, 640 / max(h, w))
    # Even dimensions for the optional video writer.
    return cv2.resize(frame, (max(2, int(w * ratio) // 2 * 2),
                              max(2, int(h * ratio) // 2 * 2)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", help="Video file; omit to use webcam.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--output", help="Optional new .mp4 output path (silent video).")
    parser.add_argument("--max-frames", type=int, default=300)
    args = parser.parse_args()
    if args.max_frames < 1:
        parser.error("--max-frames must be positive")
    if args.output and Path(args.output).exists():
        parser.error("Output already exists; choose a new filename.")
    checkpoint = ROOT / "checkpoints" / "sam2.1_hiera_tiny.pt"
    checkpoint.parent.mkdir(exist_ok=True)
    if not checkpoint.exists():
        print("Downloading SAM2.1 Tiny checkpoint…", flush=True)
        temp = checkpoint.with_suffix(".download")
        urllib.request.urlretrieve(
            "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
            temp,
        )
        temp.replace(checkpoint)

    print("Loading model on CPU. Processing is not guaranteed to be real-time.", flush=True)
    predictor = build_sam2_camera_predictor(
        "sam2.1/sam2.1_hiera_t.yaml", str(checkpoint), device="cpu"
    )
    # Set AFTER building: builder defaults override Hydra's earlier settings.
    predictor.fill_hole_area = 0
    cap = cv2.VideoCapture(args.video if args.video else args.camera)
    writer = None
    point = []
    try:
        ok, raw = cap.read()
        if not ok:
            raise RuntimeError("Cannot open video/camera. For webcam, allow camera access in macOS Settings.")
        frame = fit(raw)
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)

        def click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                h, w = frame.shape[:2]
                if 0 <= x < w and 0 <= y < h:
                    point[:] = [x, y]

        cv2.setMouseCallback(WINDOW, click)
        print("Click your object on the frozen frame, then press Enter. Q quits.", flush=True)
        while True:
            preview = frame.copy()
            if point:
                cv2.drawMarker(preview, tuple(point), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
            cv2.imshow(WINDOW, preview)
            key = cv2.waitKey(20) & 0xFF
            if key == ord("q") or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                return
            if key in (10, 13) and point:
                break

        cv2.setMouseCallback(WINDOW, lambda *a: None)
        if args.output:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not np.isfinite(fps) or fps <= 0:
                fps = 25
            writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps, (frame.shape[1], frame.shape[0]))
            if not writer.isOpened():
                raise RuntimeError("Could not create output video.")
        with torch.inference_mode():
            predictor.load_first_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            _, _, logits = predictor.add_new_prompt(
                frame_idx=0, obj_id=1,
                points=np.array([point], dtype=np.float32),
                labels=np.array([1], dtype=np.int32),
            )
            for i in range(args.max_frames):
                if i:
                    ok, raw = cap.read()
                    if not ok:
                        break
                    frame = fit(raw)
                    _, logits = predictor.track(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                mask = (logits[0, 0] > 0).cpu().numpy()
                overlay = frame.copy()
                overlay[mask] = (0, 220, 255)
                output = cv2.addWeighted(frame, 0.65, overlay, 0.35, 0)
                ys, xs = np.where(mask)
                if len(xs):
                    cv2.rectangle(output, (int(xs.min()), int(ys.min())),
                                  (int(xs.max()), int(ys.max())), (0, 255, 255), 2)
                cv2.imshow(WINDOW, output)
                if writer is not None:
                    writer.write(output)
                print(f"Processed frame {i + 1}", flush=True)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
        print("Finished. Default limit is 300 frames; adjust --max-frames for longer clips.")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
