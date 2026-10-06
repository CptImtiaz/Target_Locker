# Proposed Model 3 — SAMURAI Motion-Aware Long-Term Tracker

Model 3 now uses the **official SAMURAI tracker** instead of custom template-search or re-acquisition logic.

## Pipeline

```
RGB video
↓
Click target on frame 1
↓
Point prompt is mapped back to original video resolution
↓
Official SAMURAI
  ├─ SAM2.1 segmentation
  ├─ motion-aware memory selection
  ├─ object/mask quality gating
  └─ long-term tracking
↓
Tracked output video
↓
No-GT diagnostics
```

## Why SAMURAI

Vanilla SAM2 can drift when incorrect masks are written into memory. SAMURAI modifies SAM2 for visual object tracking with motion-aware memory selection so unreliable observations are less likely to contaminate long-term memory.

The implementation used here comes from the official repository:

`https://github.com/yangchris11/samurai`

Model 3 runs SAMURAI in an isolated subprocess so its modified SAM2 package does not conflict with the SAM2 copy already included in Target_Locker.

## User workflow

1. Run `App Interface.ipynb`.
2. Upload an RGB video.
3. Click **SELECT TARGET**.
4. Click one point on the target in the first frame.
5. Press **LOCK & TRACK**.
6. SAMURAI tracks the selected object through the original-resolution video.

No ground-truth file is required.

## Diagnostics

Without ground truth, Model 3 does **not** report tracking accuracy.

The app shows only diagnostics:

- Mask presence %
- Frames with mask
- Zero-mask frames
- Mean bounding-box area
- Mean normalized center jump
- Tracking FPS

These are useful for debugging and comparing runtime behavior, but they are **not** Success AUC, Precision, Normalized Precision, or other true SOT accuracy metrics.

## Files

- `App Interface.ipynb` — Colab setup and launcher
- `app_interface.py` — upload / click / tracking UI
- `model3_tracker.py` — isolated SAMURAI launcher
- `samurai_runtime.py` — official SAMURAI inference wrapper
- `requirements.txt` — Target_Locker runtime dependencies

## External runtime dependency

The notebook clones:

```
https://github.com/yangchris11/samurai.git
```

to:

```
/content/samurai
```

and installs its modified SAM2 package before running Model 3.
