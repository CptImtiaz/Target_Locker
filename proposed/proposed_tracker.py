"""
Proposed Target Locker inference helper.

Hybrid components:
- fine-tuned DaSiamRPN
- 8-state Kalman bbox filter
- lightweight multi-pathway memory tree
- SAM2.1 Tiny mask refinement
- trained confidence fusion

All trained assets are downloaded from the GitHub release:
https://github.com/CptImtiaz/Target_Locker/releases/tag/Proposed_model
"""

from __future__ import annotations

import importlib
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

BBox = Tuple[float, float, float, float]

REPO_URL = "https://github.com/CptImtiaz/Target_Locker.git"
REPO_ROOT = Path("/content/Target_Locker")
ASSET_DIR = Path("/content/proposed_model_assets")

RELEASE_BASE = (
    "https://github.com/CptImtiaz/Target_Locker/"
    "releases/download/Proposed_model"
)

ASSETS = {
    "target_locker_models.py": f"{RELEASE_BASE}/target_locker_models.py",
    "target_locker_training.py": f"{RELEASE_BASE}/target_locker_training.py",
    "dasiamrpn_uav_best.pth": f"{RELEASE_BASE}/dasiamrpn_uav_best.pth",
    "fusion_uav_best.pth": f"{RELEASE_BASE}/fusion_uav_best.pth",
}


def _download(url: str, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_file() and dst.stat().st_size > 0:
        return dst
    print(f"Downloading {dst.name} ...")
    urllib.request.urlretrieve(url, dst)
    if not dst.is_file() or dst.stat().st_size == 0:
        raise RuntimeError(f"Download failed: {url}")
    return dst


def ensure_assets() -> Path:
    """Prepare repository dependencies and proposed-model release assets."""
    if not REPO_ROOT.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", REPO_URL, str(REPO_ROOT)],
            check=True,
        )

    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    for name, url in ASSETS.items():
        _download(url, ASSET_DIR / name)

    if str(ASSET_DIR) not in sys.path:
        sys.path.insert(0, str(ASSET_DIR))

    return ASSET_DIR


def load_proposed_models(device: Optional[str] = None):
    """
    Build the base model stack, then replace DaSiamRPN and fusion weights
    with the trained UAV123 proposed-model checkpoints.
    """
    ensure_assets()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tlm = importlib.import_module("target_locker_models")
    tlt = importlib.import_module("target_locker_training")

    models = tlm.build_all_models(device=device)

    dasiam_ckpt = ASSET_DIR / "dasiamrpn_uav_best.pth"
    fusion_ckpt = ASSET_DIR / "fusion_uav_best.pth"

    dasiam_state = torch.load(
        dasiam_ckpt,
        map_location=device,
        weights_only=True,
    )
    models["siamese"].load_state_dict(dasiam_state, strict=True)
    models["siamese"].eval()

    fusion_state = torch.load(
        fusion_ckpt,
        map_location=device,
        weights_only=True,
    )
    models["fusion"].load_state_dict(fusion_state, strict=True)
    models["fusion"].eval()

    models["fusion_status"] = f"trained weights loaded: {fusion_ckpt}"
    models["proposed_dasiam_checkpoint"] = str(dasiam_ckpt)
    models["proposed_fusion_checkpoint"] = str(fusion_ckpt)

    return models, tlm, tlt


def valid_bbox(box: Optional[BBox]) -> bool:
    if box is None:
        return False
    x, y, w, h = box
    return bool(np.all(np.isfinite([x, y, w, h])) and w > 1 and h > 1)


def mask_to_bbox(mask) -> Optional[BBox]:
    if mask is None:
        return None

    mask = np.asarray(mask).astype(bool)
    ys, xs = np.where(mask)

    if len(xs) == 0:
        return None

    x1 = float(xs.min())
    y1 = float(ys.min())
    x2 = float(xs.max() + 1)
    y2 = float(ys.max() + 1)

    return (x1, y1, x2 - x1, y2 - y1)


def bbox_iou(a: Optional[BBox], b: Optional[BBox]) -> float:
    if not valid_bbox(a) or not valid_bbox(b):
        return 0.0

    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = aw * ah + bw * bh - inter + 1e-9
    return float(inter / union)


@torch.inference_mode()
def point_to_bbox(frame_bgr: np.ndarray, point_xy, models) -> BBox:
    """
    Convert the user's single mouse click into an initial target bbox using
    the proposed stack's SAM2.1 Tiny image predictor.
    """
    x, y = map(float, point_xy)
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    predictor = models["sam2"].predictor
    predictor.set_image(rgb)

    masks, scores, _ = predictor.predict(
        point_coords=np.array([[x, y]], dtype=np.float32),
        point_labels=np.array([1], dtype=np.int32),
        multimask_output=True,
    )

    best = int(np.argmax(scores))
    box = mask_to_bbox(masks[best])

    if not valid_bbox(box):
        # Small click-centered fallback, used only if SAM2 returns no mask.
        h, w = frame_bgr.shape[:2]
        bw = max(16.0, w * 0.04)
        bh = max(16.0, h * 0.06)
        box = (
            max(0.0, x - bw / 2),
            max(0.0, y - bh / 2),
            min(bw, float(w)),
            min(bh, float(h)),
        )

    return tuple(float(v) for v in box)


class ProposedTargetLocker:
    """Persistent single-object tracker for the trained proposed model."""

    def __init__(
        self,
        models,
        tlm,
        tlt,
        fusion_threshold: float = 0.30,
        sam2_refine_threshold: float = 0.30,
        top_k: int = 3,
    ):
        self.models = models
        self.tlm = tlm
        self.tlt = tlt

        self.device = models["device"]
        self.siamese = models["siamese"]
        self.sam2 = models["sam2"]
        self.fusion = models["fusion"]

        self.fusion_threshold = float(fusion_threshold)
        self.sam2_refine_threshold = float(sam2_refine_threshold)
        self.top_k = int(top_k)

        self.state = None
        self.kalman = None
        self.memory = None
        self.frame_size = None
        self.last_box = None
        self.last_signals = None

    @torch.inference_mode()
    def initialize(self, frame_bgr: np.ndarray, bbox: BBox) -> BBox:
        if not valid_bbox(bbox):
            raise ValueError(f"Invalid initialization bbox: {bbox}")

        h, w = frame_bgr.shape[:2]
        bbox = self.tlt.clip_bbox(bbox, w, h)

        self.state = self.tlt.init_dasiam_inference(
            frame_bgr,
            bbox,
            self.siamese,
            self.device,
        )

        self.kalman = self.tlm.KalmanBBoxFilter()
        self.kalman.initialize(bbox)

        self.memory = self.tlm.LightweightMemoryTree()
        self.memory.initialize(
            bbox,
            self.state["template_features"],
        )

        self.frame_size = (w, h)
        self.last_box = tuple(float(v) for v in bbox)
        self.last_signals = {
            "siamese": 1.0,
            "kalman": 1.0,
            "sam2": 1.0,
            "memory": 1.0,
            "fusion": 1.0,
        }

        return self.last_box

    @torch.inference_mode()
    def update(self, frame_bgr: np.ndarray) -> BBox:
        if self.state is None:
            raise RuntimeError("Call initialize() before update().")

        h, w = frame_bgr.shape[:2]

        kalman_box = self.kalman.predict()
        pred_pos, pred_sz = self.tlt.bbox_center_size(kalman_box)

        self.state["target_pos"] = pred_pos
        self.state["target_sz"] = pred_sz

        candidates = self.tlt.propose_candidates_inference(
            frame_bgr,
            self.state,
            self.siamese,
            self.device,
            top_k=self.top_k,
        )

        if candidates:
            tree_out = self.memory.step(candidates)

            siam_conf = max(float(c["sim_score"]) for c in candidates)
            memory_conf = float(tree_out["confidence"])

            visual_box = self.tlt.clip_bbox(
                tree_out["bbox"],
                w,
                h,
            )
        else:
            siam_conf = 0.0
            memory_conf = 0.0
            visual_box = self.tlt.clip_bbox(kalman_box, w, h)

        kalman_conf = float(
            np.clip(self.kalman.confidence(), 0.0, 1.0)
        )

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mask, sam2_conf, _ = self.sam2.segment_from_box(
            rgb,
            visual_box,
        )

        sam2_conf = float(np.clip(sam2_conf, 0.0, 1.0))
        sam2_box = mask_to_bbox(mask)

        if (
            valid_bbox(sam2_box)
            and sam2_conf >= self.sam2_refine_threshold
            and bbox_iou(sam2_box, visual_box) >= 0.05
        ):
            refined_box = self.tlt.clip_bbox(sam2_box, w, h)
        else:
            refined_box = visual_box

        fusion_input = torch.tensor(
            [[
                float(np.clip(siam_conf, 0.0, 1.0)),
                kalman_conf,
                sam2_conf,
                float(np.clip(memory_conf, 0.0, 1.0)),
            ]],
            dtype=torch.float32,
            device=self.device,
        )

        fused_conf = float(self.fusion(fusion_input).item())

        if candidates and fused_conf >= self.fusion_threshold:
            final_box = refined_box
            self.kalman.update(final_box)
        else:
            final_box = self.tlt.clip_bbox(kalman_box, w, h)

        pos, sz = self.tlt.bbox_center_size(final_box)
        self.state["target_pos"] = pos
        self.state["target_sz"] = sz

        self.last_box = tuple(float(v) for v in final_box)
        self.last_signals = {
            "siamese": float(siam_conf),
            "kalman": float(kalman_conf),
            "sam2": float(sam2_conf),
            "memory": float(memory_conf),
            "fusion": float(fused_conf),
        }

        return self.last_box
