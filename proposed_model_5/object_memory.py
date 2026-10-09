"""Universal object memory for Model 5."""
from collections import deque
import numpy as np
import cv2

class ObjectMemory:
    def __init__(self,fps):
        self.fps=max(float(fps),1.)
        self.positions=deque(maxlen=64)
        self.shapes=deque(maxlen=32)
        self.colors=deque(maxlen=5)
        self.last_reliable=None
        self.events=[]
        self.category="object"

    def color(self,frame,box):
        x1,y1,x2,y2=map(int,box)
        crop=frame[max(0,y1):max(0,y2),max(0,x1):max(0,x2)]
        if crop.size==0:return None
        hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
        h=cv2.calcHist([hsv],[0,1],None,[12,8],[0,180,0,256]).flatten()
        return h/max(float(np.linalg.norm(h)),1e-8)

    def observe(self,frame,index,box,mask,score=1.,force=False):
        if not force and (score is None or score<.82):return False
        x1,y1,x2,y2=box
        w=max(1,x2-x1);h=max(1,y2-y1)
        center=((x1+x2)/2,(y1+y2)/2)
        self.positions.append((int(index),center))
        self.shapes.append((int(index),int(mask.sum()),w/h))
        col=self.color(frame,box)
        if col is not None:self.colors.append(col)
        self.last_reliable=int(index)
        return True

    def velocity(self):
        if len(self.positions)<2:return (0.,0.)
        (t0,p0),(t1,p1)=self.positions[-2],self.positions[-1]
        dt=max(1,t1-t0)/self.fps
        return ((p1[0]-p0[0])/dt,(p1[1]-p0[1])/dt)

    def scores(self,frame,box,index):
        x1,y1,x2,y2=box
        w=max(1,x2-x1);h=max(1,y2-y1)
        geom=1.
        if self.shapes:
            ratio=self.shapes[0][2]
            geom=float(np.exp(-abs(np.log(max(1e-5,w/h)/max(1e-5,ratio)))))
        local=.5
        c=self.color(frame,box)
        if c is not None and self.colors:
            local=float(np.clip(max(np.dot(c,prev) for prev in self.colors),0,1))
        motion=.5
        if self.positions:
            t,p=self.positions[-1]
            vx,vy=self.velocity()
            dt=max(0,index-t)/self.fps
            predicted=(p[0]+vx*dt,p[1]+vy*dt)
            distance=np.hypot((x1+x2)/2-predicted[0],(y1+y2)/2-predicted[1])
            uncertainty=np.hypot(w,h)+20*dt+15*dt*dt
            motion=float(np.exp(-distance/max(1,uncertainty)))
        return geom,local,motion

    def rerank(self,frame,candidates,index):
        from confidence_policy import Candidate
        if not candidates:return []
        dt=0 if self.last_reliable is None else max(0,index-self.last_reliable)/self.fps
        motion_weight=float(.15*np.exp(-dt/2))
        result=[]
        for c in candidates:
            g,l,m=self.scores(frame,c.bbox,index)
            score=(.75-motion_weight)*((c.identity_score+1)/2)+.12*g+.13*l+motion_weight*m
            result.append(Candidate(c.bbox,c.detector_score,c.identity_score,float(score)))
        return sorted(result,key=lambda x:x.rank_score,reverse=True)

    def summary(self):
        return {"trusted_observations":len(self.positions),"latest_velocity_px_s":self.velocity(),
                "last_reliable_frame":self.last_reliable,"events":self.events}
