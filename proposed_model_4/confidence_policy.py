"""Pure-Python identity ambiguity and temporal confirmation policy. Cosine scores are not probabilities."""
from dataclasses import dataclass
from math import hypot
from typing import Optional,Sequence

Box=tuple[int,int,int,int]

@dataclass(frozen=True)
class Candidate:
    bbox:Box
    detector_score:float
    identity_score:float
    rank_score:float

def center(b):return ((b[0]+b[2])*.5,(b[1]+b[3])*.5)

def overlap(a,b):
    x1=max(a[0],b[0]);y1=max(a[1],b[1])
    x2=min(a[2],b[2]);y2=min(a[3],b[3])
    intersection=max(0,x2-x1)*max(0,y2-y1)
    aa=max(0,a[2]-a[0])*max(0,a[3]-a[1])
    bb=max(0,b[2]-b[0])*max(0,b[3]-b[1])
    return intersection/max(1,aa+bb-intersection)

class RecoveryGate:
    def __init__(self,min_identity=.68,min_margin=.035,confirmations=2):
        self.min_identity=float(min_identity)
        self.min_margin=float(min_margin)
        self.confirmations=max(2,int(confirmations))
        self.pending:Optional[Candidate]=None
        self.count=0
        self.rejected_ambiguous=0

    def reset(self):
        self.pending=None
        self.count=0

    def consider(self,ranked:Sequence[Candidate])->Optional[Candidate]:
        if not ranked:
            self.reset()
            return None
        top=ranked[0]
        runner=ranked[1] if len(ranked)>1 else None
        if runner and (top.identity_score-runner.identity_score)<self.min_margin:
            self.rejected_ambiguous+=1
            self.reset()
            return None
        if top.identity_score<self.min_identity:
            self.reset()
            return None
        if self.pending is None:
            self.pending=top
            self.count=1
            return None
        x,y=center(top.bbox)
        px,py=center(self.pending.bbox)
        size=max(1,top.bbox[2]-top.bbox[0],top.bbox[3]-top.bbox[1],
                 self.pending.bbox[2]-self.pending.bbox[0],
                 self.pending.bbox[3]-self.pending.bbox[1])
        consistent=overlap(self.pending.bbox,top.bbox)>.06 or hypot(x-px,y-py)<1.2*size
        if not consistent:
            self.pending=top
            self.count=1
            return None
        self.pending=top
        self.count+=1
        if self.count>=self.confirmations:
            result=top
            self.reset()
            return result
        return None
