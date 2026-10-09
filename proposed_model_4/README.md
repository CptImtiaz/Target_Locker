# Proposed Model 4 — VLM-Guided Persistent Single-Object Tracking

## Goal
Keep the Model 3 Colab upload → first-frame mouse selection → playback interface while adding SigLIP 2 identity verification and soft motion-aware memory for re-acquisition.

## Modules
- `app_interface.py`: same Model 3 browser interface (video upload, mouse-click lock, metrics and output).
- `model4_tracker.py`: SAM2.1 Tiny RGB tracker and target-loss/re-acquisition state machine (adapted from Model 3).
- `global_redetector.py`: full-frame ResNet-18 candidate proposals from Model 3.
- `semantic_redetector.py`: SigLIP 2 visual-language image embeddings, trusted exemplar bank and velocity-based soft motion prior.
- `App Interface.ipynb`: one-click Colab setup and run.
- `requirements.txt`: dependencies.

## Method and limitations
SAM2 supplies mask tracking, ResNet-18 scans the full frame when a target is lost, SigLIP 2 compares initial target and candidate images, and a motion prior reranks proposals. Two-frame confirmation precedes SAM2 bounding-box reinitialization. The initial target anchor remains immutable.

**Important:** This is a SAMURAI-inspired *external* motion-aware memory implementation. It does NOT execute the upstream SAMURAI algorithm or load SAMURAI weights. The proposed fusion is a prototype; confidence thresholds are heuristics, not calibrated probabilities. SigLIP similarity cannot uniquely establish object identity, especially between near-identical objects. ResNet proposal quality bounds full-frame recovery. This is RGB-only, with no thermal conversion. It is not real-time on a T4 and has not been GPU-validated in this change.

## Run
Open `App Interface.ipynb` from the feature branch in Colab, select GPU, run first cell (dependencies and SAM2 checkpoint), then second cell (app). First frame appears for point selection. Upload an RGB MP4, select target, press LOCK & TRACK.

## Evaluation
Exported JSON/CSV are **no-ground-truth diagnostics**, not tracking accuracy. Validate separately with annotated UAV tracking benchmarks (Success AUC, normalized precision, reacquisition success/false recovery, GPU latency). Compare SAM2, Model 3, proposed Model 4, and official SAMURAI as an independent baseline.

Upstream research: https://github.com/yangchris11/samurai
SigLIP 2: https://huggingface.co/google/siglip2-base-patch16-512
