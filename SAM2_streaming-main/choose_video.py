"""Place next to run_local.py in SAM2-Mac; run: python choose_video.py.

Opens a macOS file picker. Uses existing SAM2 CPU runner and checkpoint.
Default: first 300 frames. Use --max-frames N to change this.
"""
import argparse
from pathlib import Path
import subprocess
import sys
import uuid


def main():
    parser = argparse.ArgumentParser(description="Choose a local video for SAM2 tracking.")
    parser.add_argument("--max-frames", type=int, default=300)
    args = parser.parse_args()
    if args.max_frames < 1:
        parser.error("--max-frames must be positive")
    root = Path(__file__).resolve().parent
    runner = root / "run_local.py"
    if not runner.is_file():
        parser.error("Place choose_video.py inside SAM2-Mac, beside run_local.py.")
    if sys.platform != "darwin":
        parser.error("This file picker is for macOS.")
    result = subprocess.run(
        ["osascript", "-e",
         'POSIX path of (choose file with prompt "Choose a video to track")'],
        capture_output=True, text=True,
    )
    if result.returncode:
        if "-128" in result.stderr:
            print("Selection cancelled.")
            return 0
        print("Could not open file picker: " + result.stderr.strip(), file=sys.stderr)
        return 1
    # Remove only the line terminator; preserve spaces in the filename.
    video = Path(result.stdout.rstrip("\n"))
    if not video.is_file():
        parser.error("Selected file is not available.")
    outputs = root / "outputs"
    outputs.mkdir(exist_ok=True)
    output = outputs / f"tracked-{uuid.uuid4().hex[:12]}.mp4"
    print(f"Video: {video}", flush=True)
    print(f"Processing up to {args.max_frames} frames on CPU.", flush=True)
    print("Click the target on the first frame and press Enter. Q stops early.", flush=True)
    status = subprocess.run(
        [sys.executable, str(runner), "--video", str(video),
         "--output", str(output), "--max-frames", str(args.max_frames)],
        cwd=root,
    )
    if status.returncode == 0 and output.exists() and output.stat().st_size > 0:
        print(f"Output (no audio): {output}", flush=True)
        subprocess.run(["open", "-R", str(output)], check=False)
    elif status.returncode:
        print("Tracking failed; see the error above. Any partial output may be incomplete.")
    return status.returncode


if __name__ == "__main__":
    sys.exit(main())
