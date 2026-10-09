"""Grounding DINO full-frame target-category detection followed by SigLIP 2 instance matching."""
import cv2
import torch
from PIL import Image
from transformers import AutoProcessor,AutoModelForZeroShotObjectDetection
from confidence_policy import Candidate,overlap

class GlobalReacquirer:
    def __init__(self,identity,device="cuda",model_id="IDEA-Research/grounding-dino-tiny",
                 detector_threshold=.24,topk=12):
        self.identity=identity
        self.device=torch.device(device)
        self.processor=AutoProcessor.from_pretrained(model_id)
        self.model=AutoModelForZeroShotObjectDetection.from_pretrained(model_id).eval().to(self.device)
        self.detector_threshold=float(detector_threshold)
        self.topk=int(topk)
        self.search_calls=0

    @torch.inference_mode()
    def search(self,frame):
        self.search_calls+=1
        h,w=frame.shape[:2]
        rgb=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        prompt=(self.identity.selected_label or "object").strip().rstrip(".")+"."
        inputs=self.processor(images=rgb,text=prompt,return_tensors="pt")
        inputs={k:v.to(self.device) if hasattr(v,"to") else v for k,v in inputs.items()}
        out=self.model(**inputs)
        results=self.processor.post_process_grounded_object_detection(
            out,inputs["input_ids"],threshold=self.detector_threshold,
            text_threshold=.20,target_sizes=[(h,w)])[0]
        proposals=[]
        for box,score in zip(results["boxes"].cpu().tolist(),results["scores"].cpu().tolist()):
            x1,y1,x2,y2=box
            b=(max(0,int(x1)),max(0,int(y1)),min(w,int(x2)),min(h,int(y2)))
            if b[2]-b[0]<4 or b[3]-b[1]<4:continue
            proposals.append((b,float(score)))
        proposals.sort(key=lambda item:item[1],reverse=True)
        unique=[]
        for candidate in proposals:
            if all(overlap(candidate[0],other[0])<.55 for other in unique):
                unique.append(candidate)
            if len(unique)>=self.topk:break
        if not unique:return []
        similarity=self.identity.compare(frame,[b for b,_ in unique])
        matches=[Candidate(bbox=b,detector_score=conf,identity_score=sim,
                           rank_score=.90*sim+.10*conf)
                 for (b,conf),sim in zip(unique,similarity)]
        matches.sort(key=lambda c:c.rank_score,reverse=True)
        return matches
