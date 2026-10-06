from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


BBox = Tuple[int, int, int, int]
Point = Tuple[int, int]


@dataclass
class Candidate:
    bbox: BBox
    center: Point
    score: float
    method: str


def clamp_bbox(bbox: BBox, w: int, h: int) -> BBox:
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(x1 + 1, min(int(x2), w))
    y2 = max(y1 + 1, min(int(y2), h))
    return x1, y1, x2, y2


def bbox_center(bbox: BBox) -> Point:
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def expand_bbox(bbox: BBox, scale: float, w: int, h: int) -> BBox:
    x1, y1, x2, y2 = bbox
    cx, cy = bbox_center(bbox)
    bw = max(8, int((x2 - x1) * scale))
    bh = max(8, int((y2 - y1) * scale))
    return clamp_bbox((cx - bw // 2, cy - bh // 2, cx + bw // 2, cy + bh // 2), w, h)


def crop(frame: np.ndarray, bbox: BBox) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    return frame[y1:y2, x1:x2]


def normalize_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(img)


def template_candidates(
    frame: np.ndarray,
    template: np.ndarray,
    search_bbox: Optional[BBox] = None,
    scales: Sequence[float] = (0.75, 0.9, 1.0, 1.1, 1.25),
    topk: int = 5,
    method_name: str = "template",
) -> List[Candidate]:
    h, w = frame.shape[:2]
    if search_bbox is None:
        search_bbox = (0, 0, w, h)
    search_bbox = clamp_bbox(search_bbox, w, h)
    sx1, sy1, sx2, sy2 = search_bbox
    search = normalize_gray(frame[sy1:sy2, sx1:sx2])
    base = normalize_gray(template)

    candidates: List[Candidate] = []
    if search.size == 0 or base.size == 0:
        return candidates

    for scale in scales:
        tw = max(6, int(base.shape[1] * scale))
        th = max(6, int(base.shape[0] * scale))
        if tw >= search.shape[1] or th >= search.shape[0]:
            continue
        t = cv2.resize(base, (tw, th), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
        result = cv2.matchTemplate(search, t, cv2.TM_CCOEFF_NORMED)

        work = result.copy()
        for _ in range(min(topk, work.size)):
            _, score, _, loc = cv2.minMaxLoc(work)
            if not np.isfinite(score):
                break
            x, y = loc
            bbox = clamp_bbox((sx1 + x, sy1 + y, sx1 + x + tw, sy1 + y + th), w, h)
            candidates.append(Candidate(bbox, bbox_center(bbox), float(score), method_name))
            rx1 = max(0, x - tw // 2)
            ry1 = max(0, y - th // 2)
            rx2 = min(work.shape[1], x + tw // 2 + 1)
            ry2 = min(work.shape[0], y + th // 2 + 1)
            work[ry1:ry2, rx1:rx2] = -1.0

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:topk]


def thermal_fingerprint(img: np.ndarray) -> np.ndarray:
    g = normalize_gray(img)
    hist = cv2.calcHist([g], [0], None, [16], [0, 256]).reshape(-1)
    hist = hist / max(hist.sum(), 1e-6)
    edges = cv2.Canny(g, 50, 150)
    edge_density = np.array([edges.mean() / 255.0], dtype=np.float32)
    stats = np.array([
        g.mean() / 255.0,
        g.std() / 128.0,
        np.percentile(g, 25) / 255.0,
        np.percentile(g, 50) / 255.0,
        np.percentile(g, 75) / 255.0,
    ], dtype=np.float32)
    return np.concatenate([hist.astype(np.float32), edge_density, stats])


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-8:
        return 0.0
    return float(np.dot(a, b) / denom)


class ReacquisitionEngine:
    """
    Modular lock-loss recovery for small thermal targets.

    Methods:
      adaptive_zoom
      trajectory_tube
      dual_resolution
      thermal_fingerprint
      temporal_voting
      multi_hypothesis_backward
      auto_ensemble
    """

    METHODS = {
        "adaptive_zoom": "Adaptive Zoom",
        "trajectory_tube": "Trajectory Tube",
        "dual_resolution": "Dual Resolution",
        "thermal_fingerprint": "Thermal Fingerprint",
        "temporal_voting": "Temporal Voting",
        "multi_hypothesis_backward": "Multi-Hypothesis + Backward Consistency",
        "auto_ensemble": "Auto Ensemble",
    }

    def __init__(self, initial_frame: np.ndarray, initial_bbox: BBox):
        h, w = initial_frame.shape[:2]
        self.frame_shape = (h, w)
        self.initial_bbox = clamp_bbox(initial_bbox, w, h)
        self.last_bbox = self.initial_bbox
        self.centers: List[Point] = [bbox_center(self.initial_bbox)]
        self.templates: List[np.ndarray] = []
        self.fingerprints: List[np.ndarray] = []
        self._remember(initial_frame, self.initial_bbox)

    def _remember(self, frame: np.ndarray, bbox: BBox):
        h, w = frame.shape[:2]
        bbox = clamp_bbox(bbox, w, h)
        c = crop(frame, bbox)
        if c.size == 0:
            return
        c = cv2.resize(c, (64, 64), interpolation=cv2.INTER_CUBIC)
        self.templates.append(c)
        self.templates = self.templates[-5:]
        self.fingerprints.append(thermal_fingerprint(c))
        self.fingerprints = self.fingerprints[-5:]
        self.last_bbox = bbox
        self.centers.append(bbox_center(bbox))
        self.centers = self.centers[-8:]

    def update_lock(self, frame: np.ndarray, bbox: BBox):
        self._remember(frame, bbox)

    def _template(self) -> np.ndarray:
        return self.templates[-1]

    def _reference_fp(self) -> np.ndarray:
        return np.mean(np.stack(self.fingerprints), axis=0)

    def _predicted_center(self) -> Point:
        if len(self.centers) < 3:
            return self.centers[-1]
        p0 = np.array(self.centers[-3], dtype=np.float32)
        p1 = np.array(self.centers[-2], dtype=np.float32)
        p2 = np.array(self.centers[-1], dtype=np.float32)
        v = 0.25 * (p1 - p0) + 0.75 * (p2 - p1)
        pred = p2 + v
        h, w = self.frame_shape
        return int(np.clip(pred[0], 0, w - 1)), int(np.clip(pred[1], 0, h - 1))

    def _candidate_fp_score(self, frame: np.ndarray, c: Candidate) -> float:
        patch = crop(frame, c.bbox)
        if patch.size == 0:
            return 0.0
        patch = cv2.resize(patch, (64, 64), interpolation=cv2.INTER_CUBIC)
        return max(0.0, cosine_similarity(self._reference_fp(), thermal_fingerprint(patch)))

    def adaptive_zoom(self, frame: np.ndarray) -> Optional[Candidate]:
        h, w = frame.shape[:2]
        for scale in (2.0, 3.5, 6.0, 10.0):
            region = expand_bbox(self.last_bbox, scale, w, h)
            cands = template_candidates(frame, self._template(), region, topk=3, method_name="adaptive_zoom")
            if cands and cands[0].score >= 0.38:
                return cands[0]
        cands = template_candidates(frame, self._template(), None, topk=3, method_name="adaptive_zoom_full")
        return cands[0] if cands else None

    def trajectory_tube(self, frame: np.ndarray) -> Optional[Candidate]:
        h, w = frame.shape[:2]
        px, py = self._predicted_center()
        bw = max(24, self.last_bbox[2] - self.last_bbox[0])
        bh = max(24, self.last_bbox[3] - self.last_bbox[1])

        for mult in (3.0, 5.0, 8.0):
            region = clamp_bbox(
                (px - int(bw * mult), py - int(bh * mult),
                 px + int(bw * mult), py + int(bh * mult)),
                w, h,
            )
            cands = template_candidates(frame, self._template(), region, topk=5, method_name="trajectory_tube")
            if cands:
                for c in cands:
                    d = np.hypot(c.center[0] - px, c.center[1] - py)
                    diag = max(1.0, np.hypot(region[2] - region[0], region[3] - region[1]))
                    c.score = 0.75 * c.score + 0.25 * max(0.0, 1.0 - d / diag)
                cands.sort(key=lambda x: x.score, reverse=True)
                if cands[0].score >= 0.38:
                    return cands[0]
        return None

    def dual_resolution(self, frame: np.ndarray) -> Optional[Candidate]:
        h, w = frame.shape[:2]
        small_w = min(384, w)
        scale = small_w / float(w)
        small = cv2.resize(frame, (small_w, max(2, int(h * scale))), interpolation=cv2.INTER_AREA)
        tmpl = self._template()
        tmpl_s = cv2.resize(tmpl, (max(6, int(tmpl.shape[1] * scale)), max(6, int(tmpl.shape[0] * scale))))
        coarse = template_candidates(small, tmpl_s, None, scales=(0.8, 1.0, 1.2), topk=3, method_name="dual_resolution")
        if not coarse:
            return None

        best = coarse[0]
        cx = int(best.center[0] / scale)
        cy = int(best.center[1] / scale)
        bw = max(32, (self.last_bbox[2] - self.last_bbox[0]) * 4)
        bh = max(32, (self.last_bbox[3] - self.last_bbox[1]) * 4)
        region = clamp_bbox((cx - bw, cy - bh, cx + bw, cy + bh), w, h)
        fine = template_candidates(frame, self._template(), region, topk=5, method_name="dual_resolution")
        return fine[0] if fine else None

    def thermal_fingerprint_search(self, frame: np.ndarray) -> Optional[Candidate]:
        cands = template_candidates(
            frame, self._template(), None,
            scales=(0.65, 0.8, 1.0, 1.2, 1.4),
            topk=12, method_name="thermal_fingerprint",
        )
        if not cands:
            return None
        for c in cands:
            fp = self._candidate_fp_score(frame, c)
            c.score = 0.55 * c.score + 0.45 * fp
        cands.sort(key=lambda x: x.score, reverse=True)
        return cands[0]

    def temporal_voting(
        self,
        frame_idx: int,
        frames: Sequence[np.ndarray],
        lookahead: int = 3,
    ) -> Optional[Candidate]:
        base_candidates = template_candidates(
            frames[frame_idx], self._template(), None,
            scales=(0.7, 0.85, 1.0, 1.15, 1.3),
            topk=6, method_name="temporal_voting",
        )
        if not base_candidates:
            return None

        for c in base_candidates:
            votes = [c.score]
            cx, cy = c.center
            tmpl = crop(frames[frame_idx], c.bbox)
            if tmpl.size == 0:
                continue
            for j in range(frame_idx + 1, min(len(frames), frame_idx + 1 + lookahead)):
                h, w = frames[j].shape[:2]
                bw = max(30, (c.bbox[2] - c.bbox[0]) * 3)
                bh = max(30, (c.bbox[3] - c.bbox[1]) * 3)
                region = clamp_bbox((cx - bw, cy - bh, cx + bw, cy + bh), w, h)
                nxt = template_candidates(frames[j], tmpl, region, topk=1, method_name="temporal_vote")
                if nxt:
                    votes.append(nxt[0].score)
                    cx, cy = nxt[0].center
            c.score = float(np.mean(votes)) * min(1.0, len(votes) / float(lookahead + 1))

        base_candidates.sort(key=lambda x: x.score, reverse=True)
        return base_candidates[0]

    def multi_hypothesis_backward(
        self,
        frame_idx: int,
        frames: Sequence[np.ndarray],
    ) -> Optional[Candidate]:
        cands = template_candidates(
            frames[frame_idx], self._template(), None,
            scales=(0.65, 0.8, 1.0, 1.2, 1.4),
            topk=8, method_name="multi_hypothesis_backward",
        )
        if not cands:
            return None

        history_templates = self.templates[-3:]
        pred = self._predicted_center()
        h, w = frames[frame_idx].shape[:2]
        diag = max(1.0, np.hypot(w, h))

        for c in cands:
            patch = crop(frames[frame_idx], c.bbox)
            if patch.size == 0:
                c.score = 0.0
                continue
            backward_scores = []
            for hist in history_templates:
                resized = cv2.resize(patch, (hist.shape[1], hist.shape[0]), interpolation=cv2.INTER_CUBIC)
                a = normalize_gray(resized).astype(np.float32).reshape(-1)
                b = normalize_gray(hist).astype(np.float32).reshape(-1)
                if a.std() > 1e-6 and b.std() > 1e-6:
                    corr = float(np.corrcoef(a, b)[0, 1])
                    backward_scores.append(max(0.0, corr))
            backward = float(np.mean(backward_scores)) if backward_scores else 0.0
            motion = max(0.0, 1.0 - np.hypot(c.center[0] - pred[0], c.center[1] - pred[1]) / diag)
            fp = self._candidate_fp_score(frames[frame_idx], c)
            c.score = 0.35 * c.score + 0.30 * backward + 0.20 * fp + 0.15 * motion

        cands.sort(key=lambda x: x.score, reverse=True)
        return cands[0]

    def auto_ensemble(self, frame_idx: int, frames: Sequence[np.ndarray]) -> Optional[Candidate]:
        proposals = [
            self.adaptive_zoom(frames[frame_idx]),
            self.trajectory_tube(frames[frame_idx]),
            self.dual_resolution(frames[frame_idx]),
            self.thermal_fingerprint_search(frames[frame_idx]),
            self.temporal_voting(frame_idx, frames),
            self.multi_hypothesis_backward(frame_idx, frames),
        ]
        proposals = [p for p in proposals if p is not None]
        if not proposals:
            return None

        # Consensus bonus: proposals close to other methods are favored.
        diag = max(1.0, np.hypot(frames[frame_idx].shape[1], frames[frame_idx].shape[0]))
        for p in proposals:
            agreement = []
            for q in proposals:
                if p is q:
                    continue
                d = np.hypot(p.center[0] - q.center[0], p.center[1] - q.center[1])
                agreement.append(max(0.0, 1.0 - d / (0.15 * diag)))
            consensus = float(np.mean(agreement)) if agreement else 0.0
            p.score = 0.75 * p.score + 0.25 * consensus

        proposals.sort(key=lambda x: x.score, reverse=True)
        proposals[0].method = "auto_ensemble"
        return proposals[0]

    def reacquire(
        self,
        method: str,
        frame_idx: int,
        frames: Sequence[np.ndarray],
    ) -> Optional[Candidate]:
        method = method.lower().strip()
        frame = frames[frame_idx]

        if method == "adaptive_zoom":
            return self.adaptive_zoom(frame)
        if method == "trajectory_tube":
            return self.trajectory_tube(frame)
        if method == "dual_resolution":
            return self.dual_resolution(frame)
        if method == "thermal_fingerprint":
            return self.thermal_fingerprint_search(frame)
        if method == "temporal_voting":
            return self.temporal_voting(frame_idx, frames)
        if method == "multi_hypothesis_backward":
            return self.multi_hypothesis_backward(frame_idx, frames)
        if method == "auto_ensemble":
            return self.auto_ensemble(frame_idx, frames)

        raise ValueError(f"Unknown reacquisition method: {method}")
