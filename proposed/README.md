# Proposed Target Locker

This folder adds a **separate proposed-model app** without changing the existing Target Locker files.

## Proposed model

The runtime combines:

- fine-tuned DaSiamRPN
- Kalman bounding-box prediction
- lightweight multi-pathway memory
- SAM2.1 Tiny mask refinement
- trained confidence fusion

The trained files are downloaded automatically from the **Proposed_model** GitHub release:

- `dasiamrpn_uav_best.pth`
- `fusion_uav_best.pth`
- `target_locker_models.py`
- `target_locker_training.py`

SAM2.1 Tiny and the base DaSiamRPN checkpoint are prepared by the existing model setup code.

## Run

Open:

`proposed/App Interface.ipynb`

in Google Colab with a GPU runtime.

Workflow:

**Upload video → Analyze → Preview → Select Target → Click Object → Lock & Track → Result**

The visual interface follows the original Target Locker app, while the tracking backend uses the trained proposed hybrid model.
