# Proposed Model 3 — SAM2 + Deep Global Re-acquisition

This version uses a true two-stage long-term tracking design instead of asking SAM2 memory to recover itself.

## Pipeline

```
First-frame target selection
        ↓
SAM2 local tracking
        ↓
Independent target-identity verifier
        ↓
Reliable?
├── Yes → continue SAM2 and update only trusted target exemplars
└── No  → declare target lost
              ↓
        deep full-frame search
              ↓
        top target candidates
              ↓
        appearance verification
              ↓
        temporal confirmation
              ↓
        recovered bounding box
              ↓
        reinitialize SAM2 with bbox
              ↓
        local tracking resumes
```

## Why this is different

SAM2, SAMURAI and SAM2Long mainly improve continuity/memory. They are not dedicated full-frame re-detectors.

Model 3 separates the two jobs:

- **SAM2** = local mask tracker
- **GlobalTargetRedetector** = full-frame target search after loss
- **Verifier** = prevents SAM2 confidence alone from deciding identity
- **BBox reinitialization** = gives SAM2 a stronger recovery prompt than a single point

## Files

- `App Interface.ipynb` — Colab setup
- `app_interface.py` — upload / target selection / output UI
- `model3_tracker.py` — local/global state machine and SAM2 reinitialization
- `global_redetector.py` — pretrained ResNet-18 full-frame exemplar search and verification
- `requirements.txt` — dependencies

## Global re-detector

`global_redetector.py` uses pretrained ResNet-18 features.

The first target becomes an immutable anchor. Only high-confidence local-track appearances are added to a short trusted appearance bank.

When the local tracker is lost:

1. extract a deep feature map over the whole frame;
2. compare every spatial location with the target descriptor;
3. keep top candidate locations;
4. test multiple target scales;
5. verify candidates against the anchor + trusted appearance bank;
6. require temporal confirmation;
7. return the recovered bbox to SAM2.

This follows the local-tracker + global-redetector principle used by long-term trackers such as GlobalTrack/LTMU, but the implementation here is written specifically for this repository.

## No-GT diagnostics

The app reports:

- loss events
- global search frames
- confirmed re-acquisitions
- failed re-acquisition checks
- mean global candidate score
- mean identity-verifier score
- mean SAM confidence while locked
- normalized center jump
- tracking FPS

These are diagnostics, not tracking accuracy metrics.
