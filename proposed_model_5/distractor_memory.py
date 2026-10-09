"""Conservative negative-candidate cache for rejected reinitializations."""
from collections import deque
class DistractorMemory:
    def __init__(self,maxlen=8):
        self.rejected=deque(maxlen=maxlen)
        self.rejections=0
    def observe(self,frame,box,identity):
        try:
            from vlm_identity import crop_rgb
            embedding=identity.images([crop_rgb(frame,box)]).detach()
            self.rejected.append(embedding)
            self.rejections+=1
        except Exception:
            pass
    def penalize(self,frame,candidates,identity,frame_index=0):
        if not self.rejected or not candidates:return candidates
        import torch
        from vlm_identity import crop_rgb
        from confidence_policy import Candidate
        vectors=identity.images([crop_rgb(frame,c.bbox) for c in candidates])
        bank=torch.cat(list(self.rejected),dim=0)
        negatives=(vectors@bank.T).max(dim=1).values.cpu().tolist()
        ranked=[]
        for c,neg in zip(candidates,negatives):
            # Only heavily penalize near-exact matches to a *rejected* candidate.
            # Rejected masks do NOT prove negative identity.
            penalty=.055 if neg>.98 else 0.
            ranked.append(Candidate(c.bbox,c.detector_score,c.identity_score,c.rank_score-penalty))
        return sorted(ranked,key=lambda c:c.rank_score,reverse=True)
    def summary(self):
        return {"stored_rejected_candidates":len(self.rejected),
                "rejected_reinitializations_observed":self.rejections,
                "warning":"Rejected SAM2 masks are not ground-truth negative identities"}
