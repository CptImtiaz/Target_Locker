# Proposed Model 2 — TherA + SAM2 Thermal Target Locker

This model keeps the existing Target Locker interaction style but moves tracking into a **synthetic thermal/TIR domain**.

## Pipeline

```
RGB video
↓
extract all RGB frames
↓
bundled TherA-compatible RGB → synthetic TIR runtime
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

- `App Interface.ipynb` — Colab app with upload/select/lock/result workflow.
- `thera_runtime.py` — Target_Locker-owned TherA-compatible inference runtime.
- `thermal_sam2_tracker.py` — frame extraction, thermal conversion runner, video creation, and SAM2 tracking.
- `app_interface.py` — Model 2 interface.
- `MODEL_ASSETS.md` — exact model/config assets required for inference.
- `requirements.txt` — complete runtime dependencies.

## External dependency policy

The app **does not clone the TherA GitHub repository** anymore.

Only the trained TherA model assets are fetched from Hugging Face because the neural-network binaries are too large to store as normal GitHub files. GitHub blocks ordinary files larger than 100 MiB.

Expected TherA model layout:

```
/content/thera_weights/
├── checkpoint/model.pt
├── merged_models/
│   ├── unet/
│   └── adapter/
├── stable-diffusion/
│   ├── vae/
│   └── scheduler/
└── reference_caches/
    ├── SUNNY.pt
    ├── CLOUDY.pt
    ├── RAINY.pt
    └── NIGHT.pt
```

The setup notebook downloads all available documented condition caches.

## Runtime behavior

TherA conversion runs in a child Python process and exits before SAM2 loads. This releases GPU memory between RGB→TIR conversion and target tracking.

Default settings:

- TherA sampling steps: 20
- Max RGB frame side: 768
- SAM2.1 Tiny for tracking
- Default reference condition: `SUNNY`

## Important

TherA output is **synthetic thermal infrared imagery** generated from RGB. It is not radiometric temperature measurement and does not replace a real thermal sensor when actual temperature values are required.

## Run

Open `App Interface.ipynb` in Google Colab, select a GPU runtime, run the setup cell, then run the app cell.

Upload RGB video → full thermal conversion → click target on thermal frame 1 → **LOCK & TRACK**.
