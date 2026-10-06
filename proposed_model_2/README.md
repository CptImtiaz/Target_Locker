# Proposed Model 2 — TherA + SAM2 Thermal Target Locker

This model keeps the existing Target Locker interaction style but moves tracking into a **synthetic thermal/TIR domain**.

## Pipeline

```
RGB video
↓
extract all RGB frames
↓
TherA RGB → synthetic TIR conversion
↓
thermal_frames/frame_000000.png ...
↓
generate thermal.mp4
↓
display thermal frame 1
↓
mouse click target
↓
SAM2.1 Tiny gets the target mask
↓
SAM2 propagates the selected object across all thermal frames
↓
output_thermal_tracking.mp4
```

## Files

- `App Interface.ipynb` — Colab app with the same upload/select/lock/result workflow.
- `thermal_sam2_tracker.py` — frame extraction, TherA batch inference wrapper, thermal-video creation, and SAM2 thermal tracking.
- `requirements.txt` — lightweight Target Locker dependencies. The notebook additionally installs TherA's official requirements.

## TherA mode

The notebook uses the official TherA repository and **reference-cache inference** (default `SUNNY.pt`) so LLaVA does not need to stay loaded at runtime.

TherA runs in a child process and exits before SAM2 loads. This is deliberate to release GPU memory between RGB→TIR conversion and tracking.

Default notebook settings:
- TherA sampling steps: 20 (raise toward the official default 100 for quality)
- Max RGB frame side: 768
- SAM2.1 Tiny for tracking
- Thermal palette/reference cache: SUNNY

## Important

TherA produces **synthetic thermal infrared imagery** from RGB. It does not recover true radiometric temperature and does not replace a real thermal sensor for temperature measurement.

## Run

Open `App Interface.ipynb` in Google Colab, choose a GPU runtime, run the setup cell, then run the app cell.

Upload RGB video → wait for full thermal conversion → click the object on thermal frame 1 → press **LOCK & TRACK**.
