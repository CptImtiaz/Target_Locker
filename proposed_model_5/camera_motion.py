"""Lightweight background feature displacement estimate (diagnostic only)."""
import cv2
import numpy as np
class CameraMotion:
    def __init__(self):
        self.previous=None
        self.last_shift=(0.,0.)
        self.confidence=0.
        self.frames=0
    def update(self,frame):
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        small=cv2.resize(gray,(320,180))
        if self.previous is not None:
            prev=np.float32(self.previous)
            now=np.float32(small)
            shift,response=cv2.phaseCorrelate(prev,now)
            self.last_shift=(float(shift[0])*frame.shape[1]/320,
                             float(shift[1])*frame.shape[0]/180)
            self.confidence=float(np.clip(response,0,1))
        self.previous=small
        self.frames+=1
        return self.last_shift
    def summary(self):
        return {"method":"phase correlation diagnostic","frames":self.frames,
                "last_displacement_pixels":self.last_shift,
                "last_response":self.confidence,
                "motion_compensation_applied_to_matching":False}
