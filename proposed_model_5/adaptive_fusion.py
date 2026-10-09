"""Uncertainty-adaptive fusion with explicit abstention.

Scores are heuristic evidence, NOT calibrated identity probabilities.
"""
from dataclasses import dataclass
from math import exp
import numpy as np
from confidence_policy import Candidate

@dataclass
class Decision:
    accepted: bool
    reason: str
    uncertainty: float
    margin: float
    weights: dict

class AdaptiveFusion:
    def __init__(self, min_appearance=.68, min_fused_margin=.045):
        self.min_appearance=float(min_appearance)
        self.min_fused_margin=float(min_fused_margin)
        self.history=[]
        self.last=None

    def evaluate(self, frame, candidates, memory, frame_index, camera_response=0.):
        if not candidates:
            self.last=Decision(False,"no_candidates",1.,0.,{})
            return [],self.last
        dt=0. if memory.last_reliable is None else max(0.,(frame_index-memory.last_reliable)/memory.fps)
        visual_quality=float(np.clip(max(c.identity_score for c in candidates),0.,1.))
        motion_quality=float(np.clip(exp(-dt/2.0)*(1.-.7*max(0.,1.-camera_response)),.02,1.))
        ref_count=len(memory.shapes)
        geometry_quality=float(np.clip(ref_count/5.,.2,1.))
        local_quality=float(np.clip(len(memory.colors)/4.,.2,1.))
        raw=np.array([.54*max(.15,visual_quality),.19*motion_quality,
                      .13*geometry_quality,.14*local_quality],dtype=float)
        weights=raw/np.sum(raw)
        keys=("appearance","motion","geometry","local")
        wd={k:round(float(v),4) for k,v in zip(keys,weights)}
        ranked=[]
        for c in candidates:
            g,l,m=memory.scores(frame,c.bbox,frame_index)
            appearance=float(np.clip((c.identity_score+1.)/2.,0,1))
            cue=np.array([appearance,m,g,l],dtype=float)
            score=float(np.dot(weights,cue))
            ranked.append(Candidate(c.bbox,c.detector_score,c.identity_score,score))
        ranked.sort(key=lambda c:c.rank_score,reverse=True)
        top=ranked[0]
        runner=ranked[1] if len(ranked)>1 else None
        fused_margin=(top.rank_score-runner.rank_score) if runner else 1.
        visual_margin=(top.identity_score-runner.identity_score) if runner else 1.
        # Conflict/ambiguity proxy is not a posterior probability.
        ambiguity=max(0.,1.-max(0.,fused_margin)/.12) if runner else 0.
        uncertainty=float(np.clip(.45*ambiguity+.25*(1.-visual_quality)+
                                  .20*(1.-motion_quality)+.10*(1.-geometry_quality),0,1))
        if top.identity_score<self.min_appearance:
            reason="weak_appearance"
        elif runner and visual_margin<.035:
            reason="visual_tie"
        elif runner and fused_margin<self.min_fused_margin:
            reason="fusion_tie"
        elif uncertainty>.56:
            reason="uncertain"
        else:
            reason="candidate_supported"
        decision=Decision(reason=="candidate_supported",reason,uncertainty,float(fused_margin),wd)
        self.last=decision
        self.history.append({"frame":int(frame_index),"reason":reason,
                             "uncertainty":round(uncertainty,4),
                             "fused_margin":round(float(fused_margin),4),"weights":wd})
        return ranked,decision

    def summary(self):
        return {"method":"heuristic reliability-weighted fusion; not calibrated",
                "last_decision":vars(self.last) if self.last else None,
                "recent_decisions":self.history[-30:]}
