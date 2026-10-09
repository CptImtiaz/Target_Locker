"""SigLIP 2 image-language target verifier + conservative velocity-based motion memory.
SAMURAI-inspired motion-aware scoring: this is NOT an import of official SAMURAI.
"""
from __future__ import annotations
from collections import deque
import inspect
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoProcessor
from global_redetector import GlobalTargetRedetector, bbox_center, bbox_wh, crop, expand_bbox, Detection

class SemanticMotionRedetector(GlobalTargetRedetector):
    def __init__(self, *args, vlm_name="google/siglip2-base-patch16-512", **kwargs):
        super().__init__(*args, **kwargs)
        self.vlm_name=vlm_name
        self.processor=AutoProcessor.from_pretrained(vlm_name)
        self.vlm=AutoModel.from_pretrained(vlm_name, torch_dtype=torch.float16 if self.device.type=="cuda" else torch.float32).eval().to(self.device)
        self.semantic_anchor=None
        self.semantic_bank=deque(maxlen=6)
        self.locations=deque(maxlen=8)
        self.search_ticks=0
        self.semantic_checks=0

    @torch.inference_mode()
    def _semantic(self, frame, bbox):
        h,w=frame.shape[:2]
        patch=crop(frame,expand_bbox(bbox,w,h,1.30))
        if patch is None or patch.size==0:
            return None
        rgb=cv2.cvtColor(patch,cv2.COLOR_BGR2RGB)
        image=Image.fromarray(rgb)
        inputs=self.processor(images=[image],return_tensors="pt")
        inputs={k:v.to(self.device) if hasattr(v,"to") else v for k,v in inputs.items()}
        sig=inspect.signature(self.vlm.get_image_features)
        accepted={k:v for k,v in inputs.items() if k in sig.parameters}
        features=self.vlm.get_image_features(**accepted)
        if not isinstance(features,torch.Tensor):
            features=features.pooler_output if hasattr(features,"pooler_output") else features[0]
        return F.normalize(features.float().reshape(1,-1),dim=-1).detach()

    def initialize(self,frame,bbox):
        super().initialize(frame,bbox)
        self.semantic_anchor=self._semantic(frame,bbox)
        if self.semantic_anchor is None: raise RuntimeError("SigLIP 2 could not encode selected target")
        self.semantic_bank.clear()
        self.locations.clear()
        self.locations.append(bbox_center(bbox))
        self.search_ticks=0

    @torch.inference_mode()
    def _semantic_score(self,frame,bbox):
        emb=self._semantic(frame,bbox)
        if emb is None:return 0.0
        self.semantic_checks+=1
        anchor=float((emb@self.semantic_anchor.T).item())
        if self.semantic_bank:
            other=torch.cat(list(self.semantic_bank),dim=0)
            bank=float((emb@other.T).max().item())
            return float(np.clip(.75*anchor+.25*bank,-1,1))
        return float(np.clip(anchor,-1,1))

    def verify_bbox(self,frame,bbox):
        # ResNet match is stable for local frames; use SigLIP selectively on candidates
        visual=super().verify_bbox(frame,bbox)
        semantic=self._semantic_score(frame,bbox)
        return float(np.clip(.55*visual+.45*semantic,-1,1))

    def update_trusted(self,frame,bbox,verifier_score):
        if verifier_score>=.72:
            vec=self._semantic(frame,bbox)
            if vec is not None:self.semantic_bank.append(vec)
            self.locations.append(bbox_center(bbox))
        super().update_trusted(frame,bbox,verifier_score)

    def reset_confirmation(self):
        super().reset_confirmation()
        self.search_ticks=0

    def _motion_prior(self,bbox,frame_shape):
        if len(self.locations)<2:return 0.5
        (x0,y0),(x1,y1)=list(self.locations)[-2:]
        # Soft prior weakens when target has been lost for longer.
        t=min(self.search_ticks,15)
        px=x1+(x1-x0)*t
        py=y1+(y1-y0)*t
        cx,cy=bbox_center(bbox)
        diagonal=max(1,np.hypot(frame_shape[1],frame_shape[0]))
        sigma=.12+.035*t
        return float(np.exp(-0.5*((np.hypot(cx-px,cy-py)/diagonal)/sigma)**2))

    def search(self,frame):
        self.search_ticks+=1
        proposals=super().search(frame)
        # Original ResNet proposals scan full frame. SigLIP candidate verification
        # is performed by overridden verify_bbox; motion acts as a soft prior.
        for p in proposals:
            p.motion_score=self._motion_prior(p.bbox,frame.shape)
            p.score=float(.85*p.score+.15*p.motion_score)
        proposals.sort(key=lambda d:d.score,reverse=True)
        return proposals

    def confirm(self,detections):
        result=super().confirm(detections)
        if result is not None:
            self.search_ticks=0
        return result
