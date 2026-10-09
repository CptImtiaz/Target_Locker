import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from object_memory import ObjectMemory
from confidence_policy import Candidate
from camera_motion import CameraMotion
from distractor_memory import DistractorMemory
class Model5Tests(unittest.TestCase):
    def test_memory_and_search_ranking(self):
        f=np.zeros((100,100,3),dtype=np.uint8)
        m=ObjectMemory(20)
        mask=np.ones((10,10),dtype=bool)
        self.assertTrue(m.observe(f,0,(10,10,20,20),mask,force=True))
        self.assertFalse(m.observe(f,1,(10,10,20,20),mask,.2))
        self.assertTrue(m.observe(f,2,(12,10,22,20),mask,.91))
        cs=[Candidate((13,10,23,20),.9,.9,.9),Candidate((75,75,85,85),.9,.9,.9)]
        self.assertEqual(m.rerank(f,cs,3)[0].bbox,(13,10,23,20))
    def test_camera_and_distractor(self):
        f=np.zeros((100,100,3),dtype=np.uint8)
        cam=CameraMotion();cam.update(f);cam.update(f)
        self.assertEqual(cam.frames,2)
        self.assertEqual(DistractorMemory().summary()['stored_rejected_candidates'],0)
if __name__=='__main__':unittest.main()
