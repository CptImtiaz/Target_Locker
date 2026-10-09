"""Run: python -m unittest discover -s tests -v. CPU only; no model downloads."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from confidence_policy import Candidate,RecoveryGate,overlap

def candidate(box,score,det=.7):
    return Candidate(box,det,score,.9*score+.1*det)

class TestRecoveryPolicy(unittest.TestCase):
    def test_iou(self):
        self.assertAlmostEqual(overlap((0,0,10,10),(0,0,10,10)),1.)
        self.assertAlmostEqual(overlap((0,0,10,10),(20,20,30,30)),0.)
    def test_no_candidates_remains_lost(self):
        gate=RecoveryGate()
        self.assertIsNone(gate.consider([]))
    def test_reject_ambiguous_identical_cars(self):
        gate=RecoveryGate(min_margin=.04)
        a=candidate((10,10,40,40),.90)
        b=candidate((90,10,120,40),.89)
        self.assertIsNone(gate.consider([a,b]))
        self.assertEqual(gate.rejected_ambiguous,1)
        self.assertEqual(gate.count,0)
    def test_low_identity_rejected(self):
        gate=RecoveryGate(min_identity=.68)
        self.assertIsNone(gate.consider([candidate((10,10,40,40),.5)]))
    def test_two_consistent_searches_confirm(self):
        gate=RecoveryGate(confirmations=2)
        self.assertIsNone(gate.consider([candidate((10,10,40,40),.9)]))
        confirmed=gate.consider([candidate((12,10,42,40),.91)])
        self.assertIsNotNone(confirmed)
        self.assertEqual(confirmed.bbox,(12,10,42,40))
    def test_spatially_distant_candidate_resets(self):
        gate=RecoveryGate(confirmations=2)
        gate.consider([candidate((10,10,40,40),.9)])
        self.assertIsNone(gate.consider([candidate((500,500,530,530),.9)]))
        self.assertEqual(gate.count,1)

if __name__=="__main__":unittest.main()
