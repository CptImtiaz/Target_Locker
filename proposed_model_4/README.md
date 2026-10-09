# Proposed Model 4 — SAM2 + Global Re-detection + SigLIP 2

This is a **fresh replacement** of the previous Model 4 prototype, preserving its Colab video upload → first-frame click → LOCK & TRACK → preview/MP4/diagnostics interface. It does **not** use the previous ResNet-18 feature-map re-detector or the handmade velocity/motion-memory heuristic.

## Architecture

1. **SAM2.1 Tiny** from this repository's `SAM2_streaming-main`: tracks and segments the single clicked RGB target.
2. **Loss detection**: rejects missing, implausibly sized, or low-SigLIP-similarity masks; stops trusting the old SAM2 state.
3. **Grounding DINO Tiny**: searches the complete video frame for candidate objects of the selected category when lost.
4. **SigLIP 2 Base (224)**: stores immutable initial target image embedding, optionally a few high-trust examples; compares all candidate crops to original target.
5. **RecoveryGate**: rejects low target similarity and ambiguous near-ties, requires the same candidate on two separate *search frames*. Reinitializes SAM2 from bounding box only after confirmation plus mask/image verification. Otherwise remains SEARCHING (unknown identity).
6. **Output**: annotated MP4, per-frame CSV, JSON diagnostics.

### Why two models for reacquisition?
Grounding DINO is an open-vocabulary **detector**, not an individual identity matcher. SigLIP 2 is a **vision-language encoder** used to match candidates to a target reference, not a global detector. This design uses them together.

### Limitations
- The detector can miss very small UAV targets; reducing video resolution makes this worse. Increase `MAX_SIDE` if GPU permits.
- SigLIP 2 image similarity is **not unique-identity recognition**. Visually identical cars emerging from a tunnel cannot necessarily be distinguished. The UNKNOWN/SEARCHING state is intentional.
- Automatic class inference is based on a restricted class list. Type a target category in the optional field (e.g. `car`, `person`, `drone`) if automatic classification is incorrect.
- Similarities and `min_identity=.68`/`min_margin=.035` thresholds are **heuristics, not calibrated probabilities**. Validate and tune on separate annotated development videos.
- Full-frame detection every two lost frames and VLM scoring increase latency. No real-time or T4 GPU performance claims are made.
- Supports RGB video. No thermal conversion, official SAMURAI, or motion prediction is used.
- The code was structurally reviewed; Colab GPU inference and video benchmark performance are **not yet verified**.

## Run in Colab

Open `App Interface.ipynb` in Google Colab from branch `feature/proposed-model-4-vlm-memory`.
Choose **Runtime > Change runtime type > GPU**, then execute cells in order. First cell clones/updates the branch, installs requirements and downloads SAM2 Tiny. Second cell runs offline RecoveryGate unit tests. Third cell launches upload-and-click app. Choose the target with your mouse. Optionally type the object category to improve global detection.

**Downloads:** SAM2 Tiny checkpoint, Hugging Face `IDEA-Research/grounding-dino-tiny`, and `google/siglip2-base-patch16-224` are separate pretrained models.

## Files
- `App Interface.ipynb`: GPU notebook and app launcher
- `app_interface.py`: Model-3-style browser UI with optional target type input
- `model4_tracker.py`: SAM2 local tracking / loss / reinitialization / MP4/metrics
- `reacquisition.py`: Grounding DINO detection + SigLIP ranking
- `vlm_identity.py`: SigLIP 2 image and text embeddings + target gallery
- `confidence_policy.py`: conservative uncertainty and temporal matching policy
- `video_io.py`: RGB video preparation
- `tests/test_confidence_policy.py`: CPU-only tests
- `requirements.txt`: runtime dependencies

## Evaluation
CSV/JSON counts, loss events, and recovery counts are **no-GT diagnostics**. They are not ground-truth success, precision, or re-ID accuracy. For quantitative research, evaluate re-detection recall, true/false reacquisition, temporal recovery latency, unknown rejection rate, Success AUC and FPS against annotated UAV123/UAV20L or custom tunnel sequences.

## Pretrained model sources
- SAM2: https://github.com/facebookresearch/sam2
- Grounding DINO: https://huggingface.co/IDEA-Research/grounding-dino-tiny
- SigLIP 2: https://huggingface.co/google/siglip2-base-patch16-224

This project is not an implementation of MSRTrack/IGR or official SAMURAI; they are independent research baselines.


## Shadow / mask-geometry validation

Model 4 includes `mask_geometry.py`, a conservative geometry guard on SAM2
masks. It compares mask area, bounding-box elongation, disproportionate axis
growth, and mask fill to the **original selected-target mask** and the last
trusted mask. Masks that appear to expand into a person's long shadow are
rejected, and the existing Grounding DINO + SigLIP 2 reacquisition mechanism
takes over. Reinitialized SAM2 masks are checked with the same baseline.

This is a **mask rejection** strategy, not automatic person-shadow separation.
It works best if the first selected frame's mask excludes the shadow.
Perspective/scale change, pose change, and extremely small targets can also
trigger rejection; thresholds are experimental and need annotated evaluation.

The diagnostic JSON reports `geometry_rejections`. Run the geometry tests
with `python -m unittest discover -s tests -v`.
