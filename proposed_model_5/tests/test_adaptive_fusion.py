import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from adaptive_fusion import AdaptiveFusion
from object_memory import ObjectMemory
from confidence_policy import Candidate

class FusionTests(unittest.TestCase):
    def setUp(self):
        self.frame=np.zeros((120,120,3),dtype=np.uint8)
        self.memory=ObjectMemory(20)
        self.mask=np.ones((15,15),dtype=bool)
        self.memory.observe(self.frame,0,(20,20,35,35),self.mask,force=True)
    def test_reject_visual_tie(self):
        candidates=[Candidate((20,20,35,35),.9,.93,.93),
                    Candidate((70,70,85,85),.9,.92,.92)]
        ranked,d=AdaptiveFusion().evaluate(self.frame,candidates,self.memory,1,1.)
        self.assertFalse(d.accepted)
        self.assertEqual(d.reason,"visual_tie")
    def test_reject_low_similarity(self):
        _,d=AdaptiveFusion().evaluate(self.frame,[Candidate((20,20,35,35),.9,.51,.51)],self.memory,1,1.)
        self.assertFalse(d.accepted)
        self.assertEqual(d.reason,"weak_appearance")
    def test_weights_change_with_occlusion(self):
        candidate=[Candidate((20,20,35,35),.9,.95,.95)]
        f=AdaptiveFusion()
        _,near=f.evaluate(self.frame,candidate,self.memory,1,1.)
        _,far=f.evaluate(self.frame,candidate,self.memory,300,1.)
        self.assertLess(far.weights["motion"],near.weights["motion"])
        self.assertAlmostEqual(sum(near.weights.values()),1,places=3)
    def test_candidate_single_can_pass(self):
        _,d=AdaptiveFusion().evaluate(self.frame,[Candidate((20,20,35,35),.9,.95,.95)],self.memory,1,1.)
        self.assertTrue(d.accepted)
if __name__=="__main__":unittest.main()
