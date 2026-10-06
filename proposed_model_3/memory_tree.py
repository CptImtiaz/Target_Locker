from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


BBox = Tuple[int, int, int, int]
Point = Tuple[int, int]


def clamp_bbox(bbox: BBox, width: int, height: int) -> BBox:
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(int(x1), width - 1))
    y1 = max(0, min(int(y1), height - 1))
    x2 = max(x1 + 1, min(int(x2), width))
    y2 = max(y1 + 1, min(int(y2), height))
    return x1, y1, x2, y2


def bbox_center(bbox: BBox) -> Point:
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def bbox_size(bbox: BBox) -> Tuple[int, int]:
    x1, y1, x2, y2 = bbox
    return max(1, x2 - x1), max(1, y2 - y1)


def crop(frame: np.ndarray, bbox: BBox) -> Optional[np.ndarray]:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = clamp_bbox(bbox, w, h)
    patch = frame[y1:y2, x1:x2]
    return patch.copy() if patch.size else None


def gray_norm(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    return img


@dataclass
class MemoryNode:
    bbox: BBox
    center: Point
    local_score: float
    cumulative_score: float
    depth: int
    stable_count: int
    parent_center: Optional[Point] = None


class MemoryTreeReacquirer:
    """
    SAM2Long-inspired multi-hypothesis search tree for re-acquisition.

    This is an independent implementation of the memory-tree idea:
    - keep reliable appearance memories from locked frames
    - when lock is lost, generate several candidate branches
    - propagate several hypotheses across subsequent frames
    - score each branch by appearance + motion + size + history
    - prune weak branches
    - re-lock only after one branch stays strong for multiple frames
    """

    def __init__(
        self,
        max_memories: int = 12,
        beam_width: int = 5,
        candidates_per_template: int = 4,
        confirm_frames: int = 2,
        confirm_score: float = 0.62,
    ):
        self.max_memories = int(max_memories)
        self.beam_width = int(beam_width)
        self.candidates_per_template = int(candidates_per_template)
        self.confirm_frames = int(confirm_frames)
        self.confirm_score = float(confirm_score)

        self.memories: List[np.ndarray] = []
        self.memory_boxes: List[BBox] = []
        self.anchor: Optional[np.ndarray] = None
        self.anchor_bbox: Optional[BBox] = None

        self.last_locked_bbox: Optional[BBox] = None
        self.locked_centers: List[Point] = []
        self.branches: List[MemoryNode] = []
        self.loss_depth = 0

    def reset_search(self):
        self.branches = []
        self.loss_depth = 0

    def update_locked(self, frame: np.ndarray, bbox: BBox):
        patch = crop(frame, bbox)
        if patch is None:
            return

        ph, pw = patch.shape[:2]
        if ph < 5 or pw < 5:
            return

        if self.anchor is None:
            self.anchor = patch.copy()
            self.anchor_bbox = bbox

        self.memories.append(patch)
        self.memory_boxes.append(bbox)
        self.memories = self.memories[-self.max_memories :]
        self.memory_boxes = self.memory_boxes[-self.max_memories :]

        self.last_locked_bbox = bbox
        self.locked_centers.append(bbox_center(bbox))
        self.locked_centers = self.locked_centers[-8:]

        self.reset_search()


    def reference_similarity(self, frame: np.ndarray, bbox: BBox) -> float:
        """Compare a proposed locked box against trusted target memories."""
        patch = crop(frame, bbox)
        if patch is None:
            return 0.0

        refs = []
        if self.anchor is not None:
            refs.append(self.anchor)
        refs.extend(self.memories[-6:])

        if not refs:
            return 1.0

        target = gray_norm(patch)
        scores = []

        for ref in refs:
            ref_g = gray_norm(ref)

            # Compare at a common size so slow scale changes do not look like drift.
            size = (48, 48)
            a = cv2.resize(target, size, interpolation=cv2.INTER_AREA)
            b = cv2.resize(ref_g, size, interpolation=cv2.INTER_AREA)

            a = a.astype(np.float32)
            b = b.astype(np.float32)
            a -= a.mean()
            b -= b.mean()

            denom = float(np.linalg.norm(a) * np.linalg.norm(b))
            if denom <= 1e-6:
                continue

            score = float(np.clip((a * b).sum() / denom, -1.0, 1.0))
            scores.append(score)

        if not scores:
            return 0.0

        # Trust the best few reliable memories rather than a single old template.
        scores.sort(reverse=True)
        return float(np.mean(scores[: min(3, len(scores))]))

    def _predicted_center(self) -> Optional[Point]:
        if not self.locked_centers:
            return None
        if len(self.locked_centers) == 1:
            return self.locked_centers[-1]

        x1, y1 = self.locked_centers[-2]
        x2, y2 = self.locked_centers[-1]
        return int(x2 + (x2 - x1)), int(y2 + (y2 - y1))

    def _search_template(
        self,
        frame_gray: np.ndarray,
        template: np.ndarray,
        scales=(0.65, 0.8, 1.0, 1.2, 1.45),
    ):
        th0, tw0 = template.shape[:2]
        fh, fw = frame_gray.shape[:2]
        results = []

        for scale in scales:
            tw = max(5, int(tw0 * scale))
            th = max(5, int(th0 * scale))
            if tw >= fw or th >= fh:
                continue

            resized = cv2.resize(
                template,
                (tw, th),
                interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC,
            )
            response = cv2.matchTemplate(frame_gray, resized, cv2.TM_CCOEFF_NORMED)

            for _ in range(self.candidates_per_template):
                _, score, _, loc = cv2.minMaxLoc(response)
                if not np.isfinite(score):
                    break

                x, y = loc
                bbox = (x, y, x + tw, y + th)
                results.append((bbox, float(score)))

                radius_x = max(2, tw // 2)
                radius_y = max(2, th // 2)
                x1 = max(0, x - radius_x)
                y1 = max(0, y - radius_y)
                x2 = min(response.shape[1], x + radius_x)
                y2 = min(response.shape[0], y + radius_y)
                response[y1:y2, x1:x2] = -1.0

        return results

    def _appearance_candidates(self, frame: np.ndarray):
        frame_gray = gray_norm(frame)
        templates = []

        if self.anchor is not None:
            templates.append((gray_norm(self.anchor), 1.0))

        recent = self.memories[-6:]
        for i, mem in enumerate(recent):
            recency = 0.82 + 0.18 * ((i + 1) / max(len(recent), 1))
            templates.append((gray_norm(mem), recency))

        merged = []
        for template, weight in templates:
            for bbox, score in self._search_template(frame_gray, template):
                merged.append((bbox, float(score) * float(weight)))

        merged.sort(key=lambda x: x[1], reverse=True)

        kept = []
        for bbox, score in merged:
            c = bbox_center(bbox)
            duplicate = False
            for kb, _ in kept:
                kc = bbox_center(kb)
                if np.hypot(c[0] - kc[0], c[1] - kc[1]) < 0.35 * max(bbox_size(bbox)):
                    duplicate = True
                    break
            if not duplicate:
                kept.append((bbox, score))
            if len(kept) >= max(12, self.beam_width * 3):
                break

        return kept

    def _score_candidate(
        self,
        bbox: BBox,
        appearance_score: float,
        frame_shape,
        parent: Optional[MemoryNode],
    ) -> MemoryNode:
        h, w = frame_shape[:2]
        diag = max(1.0, float(np.hypot(w, h)))
        center = bbox_center(bbox)
        bw, bh = bbox_size(bbox)

        predicted = self._predicted_center()
        motion_score = 0.5
        if parent is not None:
            px, py = parent.center
            dist = np.hypot(center[0] - px, center[1] - py)
            motion_score = float(np.exp(-dist / (0.18 * diag)))
        elif predicted is not None:
            dist = np.hypot(center[0] - predicted[0], center[1] - predicted[1])
            motion_score = float(np.exp(-dist / (0.25 * diag)))

        size_score = 0.5
        ref_bbox = parent.bbox if parent is not None else self.last_locked_bbox
        if ref_bbox is not None:
            rw, rh = bbox_size(ref_bbox)
            ratio = (bw * bh) / max(1.0, rw * rh)
            size_score = float(np.exp(-abs(np.log(max(ratio, 1e-6)))))

        local = (
            0.62 * float(np.clip(appearance_score, 0.0, 1.0))
            + 0.23 * motion_score
            + 0.15 * size_score
        )

        if parent is None:
            cumulative = local
            stable_count = 1
            parent_center = None
        else:
            cumulative = 0.68 * parent.cumulative_score + 0.32 * local
            dist = np.hypot(center[0] - parent.center[0], center[1] - parent.center[1])
            stable_count = parent.stable_count + 1 if dist < 0.12 * diag else 1
            parent_center = parent.center

        return MemoryNode(
            bbox=bbox,
            center=center,
            local_score=local,
            cumulative_score=cumulative,
            depth=self.loss_depth,
            stable_count=stable_count,
            parent_center=parent_center,
        )

    def step(self, frame: np.ndarray):
        self.loss_depth += 1
        raw_candidates = self._appearance_candidates(frame)

        if not raw_candidates:
            self.branches = []
            return None, []

        new_nodes: List[MemoryNode] = []

        if not self.branches:
            for bbox, score in raw_candidates:
                new_nodes.append(
                    self._score_candidate(
                        bbox,
                        score,
                        frame.shape,
                        parent=None,
                    )
                )
        else:
            for parent in self.branches:
                px, py = parent.center
                pw, ph = bbox_size(parent.bbox)

                for bbox, score in raw_candidates:
                    cx, cy = bbox_center(bbox)
                    gate = np.hypot(cx - px, cy - py)
                    if gate <= max(40.0, 3.0 * np.hypot(pw, ph)):
                        new_nodes.append(
                            self._score_candidate(
                                bbox,
                                score,
                                frame.shape,
                                parent=parent,
                            )
                        )

            # Allow recovery from a previously bad branch by also creating fresh roots.
            for bbox, score in raw_candidates[: self.beam_width]:
                new_nodes.append(
                    self._score_candidate(
                        bbox,
                        score,
                        frame.shape,
                        parent=None,
                    )
                )

        new_nodes.sort(key=lambda n: n.cumulative_score, reverse=True)

        pruned: List[MemoryNode] = []
        for node in new_nodes:
            too_close = False
            for kept in pruned:
                if np.hypot(
                    node.center[0] - kept.center[0],
                    node.center[1] - kept.center[1],
                ) < 0.3 * max(bbox_size(node.bbox)):
                    too_close = True
                    break
            if not too_close:
                pruned.append(node)
            if len(pruned) >= self.beam_width:
                break

        self.branches = pruned
        best = self.branches[0] if self.branches else None

        confirmed = (
            best is not None
            and best.cumulative_score >= self.confirm_score
            and best.stable_count >= self.confirm_frames
        )

        return (best if confirmed else None), list(self.branches)
