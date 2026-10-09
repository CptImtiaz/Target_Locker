# Model 5: Universal Target Memory

A prototype tracking and reacquisition implementation adapted from Model 4, on its own branch. Run **App Interface.ipynb** in Google Colab GPU.

Components: SAM2.1 Tiny tracking; Grounding DINO category search; SigLIP 2 original anchor + trusted appearance embeddings; mask-geometry validation; 2-search recovery verification; `object_memory.py` trusted position history / image-plane velocity / aspect ratio / HSV color; `camera_motion.py` phase-correlation camera movement diagnostics; `distractor_memory.py` cache of rejected reinitializations; heuristic reranking and diagnostic JSON.

**Known limitations:** camera motion estimate is diagnostic only, not applied to motion prediction; no dense trajectories through long occlusion, real-world velocities, calibrated uncertainty, or accurate Re-ID of visually identical objects. Only confident masks update appearance/motion memory. Rejected SAM2 masks can be false negatives; negative cache must be treated conservatively. The system has not been end-to-end tested on Colab GPU in this change.

## Uncertainty-Adaptive Multi-Cue Memory Fusion

Implemented in `adaptive_fusion.py` and integrated into Model 5's global search. Evidence weights vary by recent image-appearance match, elapsed time since last trusted observation, camera-shift response, number of trusted geometric observations, and color gallery availability. The resulting candidate score uses visual appearance, motion, geometry, and color. A separate decision rejects insufficient appearance evidence, near-ties under either visual or fused scores, and excessive heuristic uncertainty. The existing two-observation recovery gate and SAM2 reinitialization checks remain in effect. Diagnostics include adaptive weights, uncertainty, reasons, and rejected-search counts.

**Important:** uncertainty is a heuristic ambiguity indicator, **not a calibrated probability** or learned model. Motion remains pixel-based and phase-correlation camera diagnostics are not yet applied as actual trajectory compensation. Distinguishing physically identical objects cannot be guaranteed. Calibrated confidence needs labeled benchmark data and held-out validation.
