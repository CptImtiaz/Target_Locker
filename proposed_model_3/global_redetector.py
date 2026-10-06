from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18


BBox = Tuple[int, int, int, int]


def clamp_bbox(bbox: BBox, width: int, height: int) -> BBox:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))
    return x1, y1, x2, y2


def bbox_center(bbox: BBox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def bbox_wh(bbox: BBox):
    x1, y1, x2, y2 = bbox
    return max(2, x2 - x1), max(2, y2 - y1)


def expand_bbox(bbox: BBox, width: int, height: int, factor: float = 1.8) -> BBox:
    cx, cy = bbox_center(bbox)
    bw, bh = bbox_wh(bbox)
    nw, nh = bw * factor, bh * factor
    return clamp_bbox(
        (
            int(cx - nw / 2),
            int(cy - nh / 2),
            int(cx + nw / 2),
            int(cy + nh / 2),
        ),
        width,
        height,
    )


def crop(frame: np.ndarray, bbox: BBox):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = clamp_bbox(bbox, w, h)
    patch = frame[y1:y2, x1:x2]
    return patch.copy() if patch.size else None


@dataclass
class Detection:
    bbox: BBox
    score: float
    feature_score: float
    map_score: float
    motion_score: float


class GlobalTargetRedetector:
    """
    Global exemplar re-detector.

    This is inspired by long-term tracking architectures such as GlobalTrack/LTMU:
    local tracking is separate from full-frame target search.

    The re-detector uses pretrained ResNet-18 features. It stores the first target
    as the immutable anchor and can add only highly trusted local-track exemplars.
    During loss it scans the full frame in feature space, proposes top candidate
    locations, verifies each candidate against the target feature bank, and only
    returns a candidate after temporal confirmation.
    """

    def __init__(
        self,
        device: str = "cuda",
        search_max_side: int = 768,
        topk: int = 20,
        confirm_score: float = 0.50,
        confirm_frames: int = 2,
    ):
        self.device = torch.device(device)
        self.search_max_side = int(search_max_side)
        self.topk = int(topk)
        self.confirm_score = float(confirm_score)
        self.confirm_frames = int(confirm_frames)

        net = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.encoder = torch.nn.Sequential(
            net.conv1,
            net.bn1,
            net.relu,
            net.maxpool,
            net.layer1,
            net.layer2,
        ).eval().to(self.device)

        for p in self.encoder.parameters():
            p.requires_grad_(False)

        self.mean = torch.tensor(
            [0.485, 0.456, 0.406], device=self.device
        ).view(1, 3, 1, 1)
        self.std = torch.tensor(
            [0.229, 0.224, 0.225], device=self.device
        ).view(1, 3, 1, 1)

        self.anchor_descriptor: Optional[torch.Tensor] = None
        self.descriptors: List[torch.Tensor] = []
        self.initial_bbox: Optional[BBox] = None
        self.last_good_bbox: Optional[BBox] = None

        self.pending_bbox: Optional[BBox] = None
        self.pending_count = 0
        self.pending_score = 0.0

    def _tensor(self, bgr: np.ndarray):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0)
        t = t.to(self.device) / 255.0
        return (t - self.mean) / self.std

    @torch.inference_mode()
    def _feature_map(self, bgr: np.ndarray):
        return self.encoder(self._tensor(bgr))

    @torch.inference_mode()
    def _descriptor(self, patch: np.ndarray):
        if patch is None or patch.size == 0:
            return None

        h, w = patch.shape[:2]
        if h < 8 or w < 8:
            patch = cv2.resize(
                patch,
                (max(16, w * 2), max(16, h * 2)),
                interpolation=cv2.INTER_CUBIC,
            )

        # Fixed input size makes descriptors comparable across target scales.
        patch = cv2.resize(patch, (128, 128), interpolation=cv2.INTER_CUBIC)
        fmap = self._feature_map(patch)
        desc = F.adaptive_avg_pool2d(fmap, 1).flatten(1)
        return F.normalize(desc, dim=1)

    def initialize(self, frame: np.ndarray, bbox: BBox):
        h, w = frame.shape[:2]
        bbox = clamp_bbox(bbox, w, h)
        context = expand_bbox(bbox, w, h, factor=1.6)
        desc = self._descriptor(crop(frame, context))
        if desc is None:
            raise RuntimeError("Could not create target descriptor.")

        self.anchor_descriptor = desc.detach()
        self.descriptors = [desc.detach()]
        self.initial_bbox = bbox
        self.last_good_bbox = bbox
        self.reset_confirmation()

    def reset_confirmation(self):
        self.pending_bbox = None
        self.pending_count = 0
        self.pending_score = 0.0

    def _reference_descriptor(self):
        if not self.descriptors:
            return self.anchor_descriptor
        bank = torch.cat(self.descriptors, dim=0)
        ref = bank.mean(dim=0, keepdim=True)
        return F.normalize(ref, dim=1)

    @torch.inference_mode()
    def verify_bbox(self, frame: np.ndarray, bbox: BBox) -> float:
        if self.anchor_descriptor is None:
            return 0.0

        h, w = frame.shape[:2]
        context = expand_bbox(bbox, w, h, factor=1.6)
        desc = self._descriptor(crop(frame, context))
        if desc is None:
            return 0.0

        # Anchor is always included, preventing model drift from replacing identity.
        anchor_score = float((desc @ self.anchor_descriptor.T).item())
        ref = self._reference_descriptor()
        ref_score = float((desc @ ref.T).item()) if ref is not None else anchor_score

        score = 0.65 * anchor_score + 0.35 * ref_score
        return float(np.clip(score, -1.0, 1.0))

    def update_trusted(self, frame: np.ndarray, bbox: BBox, verifier_score: float):
        if verifier_score < 0.68:
            return

        h, w = frame.shape[:2]
        context = expand_bbox(bbox, w, h, factor=1.6)
        desc = self._descriptor(crop(frame, context))
        if desc is None:
            return

        self.descriptors.append(desc.detach())
        # Keep immutable anchor + a short trusted appearance bank.
        if len(self.descriptors) > 8:
            self.descriptors = [self.descriptors[0]] + self.descriptors[-7:]

        self.last_good_bbox = bbox

    @torch.inference_mode()
    def search(self, frame: np.ndarray) -> List[Detection]:
        if self.anchor_descriptor is None or self.initial_bbox is None:
            return []

        fh, fw = frame.shape[:2]
        scale = min(1.0, self.search_max_side / max(fh, fw))
        sw = max(64, int(fw * scale))
        sh = max(64, int(fh * scale))
        search_img = (
            frame
            if scale == 1.0
            else cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)
        )

        fmap = self._feature_map(search_img)
        fmap_n = F.normalize(fmap, dim=1)
        ref = self._reference_descriptor()
        sim = (fmap_n * ref[:, :, None, None]).sum(dim=1)[0]

        # Suppress weak locations before top-k.
        flat = sim.flatten()
        k = min(self.topk, flat.numel())
        vals, inds = torch.topk(flat, k=k)

        fmap_h, fmap_w = sim.shape
        base_w, base_h = bbox_wh(self.initial_bbox)

        detections: List[Detection] = []
        seen = []

        for value, ind in zip(vals.detach().cpu().tolist(), inds.detach().cpu().tolist()):
            yy = ind // fmap_w
            xx = ind % fmap_w

            cx_s = (xx + 0.5) * (sw / fmap_w)
            cy_s = (yy + 0.5) * (sh / fmap_h)
            cx = cx_s / scale
            cy = cy_s / scale

            # Feature-map peaks near each other are the same proposal.
            if any(np.hypot(cx - sx, cy - sy) < 0.35 * max(base_w, base_h) for sx, sy in seen):
                continue
            seen.append((cx, cy))

            for size_scale in (0.70, 0.85, 1.0, 1.20, 1.45):
                bw = base_w * size_scale
                bh = base_h * size_scale
                bbox = clamp_bbox(
                    (
                        int(cx - bw / 2),
                        int(cy - bh / 2),
                        int(cx + bw / 2),
                        int(cy + bh / 2),
                    ),
                    fw,
                    fh,
                )

                feature_score = self.verify_bbox(frame, bbox)

                motion_score = 0.5
                if self.last_good_bbox is not None:
                    lx, ly = bbox_center(self.last_good_bbox)
                    diag = max(1.0, np.hypot(fw, fh))
                    d = np.hypot(cx - lx, cy - ly) / diag
                    # Weak prior only. Global re-detection must still permit large jumps.
                    motion_score = float(np.exp(-d / 0.65))

                map_score = float(np.clip((value + 1.0) / 2.0, 0.0, 1.0))
                fs01 = float(np.clip((feature_score + 1.0) / 2.0, 0.0, 1.0))

                score = 0.72 * fs01 + 0.18 * map_score + 0.10 * motion_score

                detections.append(
                    Detection(
                        bbox=bbox,
                        score=float(score),
                        feature_score=float(feature_score),
                        map_score=map_score,
                        motion_score=motion_score,
                    )
                )

            if len(seen) >= 8:
                break

        detections.sort(key=lambda d: d.score, reverse=True)
        return detections[:10]

    def confirm(self, detections: List[Detection]) -> Optional[Detection]:
        if not detections:
            self.pending_count = 0
            self.pending_bbox = None
            self.pending_score = 0.0
            return None

        best = detections[0]
        if best.score < self.confirm_score or best.feature_score < 0.15:
            self.pending_count = 0
            self.pending_bbox = None
            self.pending_score = 0.0
            return None

        if self.pending_bbox is None:
            self.pending_bbox = best.bbox
            self.pending_count = 1
            self.pending_score = best.score
            return None

        px, py = bbox_center(self.pending_bbox)
        bx, by = bbox_center(best.bbox)
        pw, ph = bbox_wh(self.pending_bbox)
        distance = np.hypot(bx - px, by - py)

        if distance <= max(24.0, 1.5 * np.hypot(pw, ph)):
            self.pending_count += 1
            self.pending_bbox = best.bbox
            self.pending_score = 0.5 * self.pending_score + 0.5 * best.score
        else:
            self.pending_bbox = best.bbox
            self.pending_count = 1
            self.pending_score = best.score

        if self.pending_count >= self.confirm_frames:
            confirmed = Detection(
                bbox=self.pending_bbox,
                score=self.pending_score,
                feature_score=best.feature_score,
                map_score=best.map_score,
                motion_score=best.motion_score,
            )
            self.reset_confirmation()
            return confirmed

        return None
