# Proposed Model 2 — Model Asset Manifest

## Runtime code

All executable RGB→synthetic-TIR logic used by Proposed Model 2 is stored in this repository:

- `thera_runtime.py`
- `thermal_sam2_tracker.py`
- `app_interface.py`

The app no longer clones the upstream TherA GitHub repository at runtime.

## TherA model assets

The following trained assets are required from the your Hugging Face model repository `Imtiaz807/ThermalConversion` and are downloaded into `/content/thera_weights` by the Colab notebook:

```
thera_weights/
├── checkpoint/model.pt
├── merged_models/
│   ├── unet/
│   └── adapter/
├── stable-diffusion/
│   ├── vae/
│   └── scheduler/
└── palettes/
    ├── SUNNY.pt
    ├── CLOUDY.pt
    ├── RAINY.pt
    └── NIGHT.pt
```

The notebook also accepts distributions where `model.pt` is placed at the root and normalizes it to `checkpoint/model.pt`.

## SAM2

SAM2.1 Tiny is downloaded to:

```
/content/sam2.1_hiera_tiny.pt
```

and uses the SAM2 source already included in this Target_Locker repository.

## Why the large weights are not normal Git files

GitHub blocks ordinary Git objects larger than 100 MiB. TherA's neural-network weight files are large model binaries, so they should be distributed with Git LFS, GitHub Release assets, or a model host rather than committed directly into the repository history.

The current notebook therefore keeps all source/runtime logic in Target_Locker and downloads only the trained model binaries at setup time.
