# Proposed Model 3 — Thermal Target Re-acquisition Lab

Model 3 keeps the TherA + SAM2 thermal tracking pipeline and adds **automatic lock-loss detection, selectable re-acquisition algorithms, SAM2 re-initialization, and comparison metrics**.

## Pipeline

```
RGB video
↓
TherA RGB → synthetic thermal
↓
Sunny / Cloudy / Rainy / Night
↓
click target on thermal frame 1
↓
SAM2 tracking
↓
lock-quality check
├── good → continue
└── lost → selected re-acquisition method
            ↓
         candidate target
            ↓
         re-prompt / re-initialize SAM2
            ↓
         continue tracking
```

## Re-acquisition methods

1. **Adaptive Zoom** — progressively expands the search crop from local to full frame.
2. **Trajectory Tube** — estimates recent motion direction and prioritizes a widening path around the expected trajectory.
3. **Dual Resolution** — coarse low-resolution full-frame localization followed by high-resolution local refinement.
4. **Thermal Fingerprint** — combines template correlation with grayscale distribution, contrast, edges, and thermal-intensity statistics.
5. **Temporal Voting** — tests candidates over several future frames and rewards candidates that remain temporally consistent.
6. **Multi-Hypothesis + Backward Consistency** — retains several candidates and compares them against recent historical target templates plus motion consistency.
7. **Auto Ensemble** — runs the available recovery strategies and adds a cross-method consensus bonus.

## App controls

- Thermal condition: Sunny / Cloudy / Rainy / Night
- Re-acquisition algorithm
- TherA steps: 5 / 8 / 12 / 20
- Mouse target selection
- Lock & Track

The default TherA setting is **8 steps** for faster video experiments.

## Evaluation

Every run exports JSON and CSV metrics:

- Lock retention %
- Number of lock-loss events
- Re-acquisition attempts
- Successful re-acquisitions
- Re-acquisition success %
- Mean re-acquisition latency (frames)
- Mean SAM confidence
- Mean normalized center jump
- Tracking FPS

These are **no-ground-truth proxy metrics** intended for comparing recovery behavior on the same video. They are not replacements for standard SOT benchmark metrics.

For formal research evaluation with ground truth, use:

- Success / AUC
- Precision
- Normalized Precision
- SR@0.5
- Failure / robustness counts

## Files

- `App Interface.ipynb` — Colab setup and launcher
- `app_interface.py` — interactive Model 3 UI
- `model3_tracker.py` — SAM2 tracking, loss detection, recovery and metrics
- `reacquisition_engine.py` — all selectable search algorithms
- `evaluation.py` — multi-run comparison log and ranking helper
- `requirements.txt` — dependencies

TherA model assets come from `Imtiaz807/ThermalConversion`. The bundled TherA-compatible runtime remains in `proposed_model_2/thera_runtime.py`.

## Recommended experiment

Use the same video, same target click, same thermal condition, and same TherA step count. Run each re-acquisition algorithm separately and compare the exported metrics. For publication-quality comparison, repeat on a dataset with ground-truth target boxes.
