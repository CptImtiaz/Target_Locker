import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mask_geometry import geometry,validate

def rectangle(y1,y2,x1,x2):
    mask=np.zeros((160,180),dtype=bool)
    mask[y1:y2,x1:x2]=True
    return mask

class TestMaskGeometry(unittest.TestCase):
    def test_normal_person_mask_accepted(self):
        ref=rectangle(30,60,70,82)
        candidate=rectangle(31,63,70,83)
        self.assertTrue(validate(candidate,geometry(ref),geometry(ref))[0])

    def test_long_shadow_rejected(self):
        ref=rectangle(30,60,70,82)
        connected_shadow=rectangle(30,115,70,82)
        self.assertFalse(validate(connected_shadow,geometry(ref),geometry(ref))[0])

    def test_progressive_shadow_rejected(self):
        ref=rectangle(30,60,70,82)
        shadow=rectangle(30,108,70,82)
        self.assertFalse(validate(shadow,geometry(ref),geometry(rectangle(30,100,70,82)))[0])

    def test_empty_mask_rejected(self):
        ref=rectangle(30,60,70,82)
        self.assertFalse(validate(np.zeros_like(ref),geometry(ref))[0])

if __name__=="__main__":unittest.main()
