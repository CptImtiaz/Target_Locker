"""Geometry guard against SAM2 mask expansion into connected shadows.

The first trusted mask is the stable reference. Scale-invariant shape checks
supplement existing temporal area checks. This rejects suspicious masks;
it does not independently remove or segment a shadow.
"""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Geometry:
    area: int
    width: int
    height: int
    fill: float
    elongation: float

def geometry(mask):
    ys,xs=np.nonzero(mask)
    if xs.size==0:return None
    w=int(xs.max()-xs.min()+1)
    h=int(ys.max()-ys.min()+1)
    area=int(xs.size)
    return Geometry(area,w,h,area/max(1,w*h),max(w,h)/max(1,min(w,h)))

def validate(mask,reference,previous=None,minimum_pixels=12):
    g=geometry(mask)
    if g is None or g.area<minimum_pixels:
        return False,"empty_or_tiny"
    if reference is None:
        return True,"ok"
    # Reject disproportionate expansion of one axis, typical of a long shadow.
    wx=g.width/max(1,reference.width)
    hy=g.height/max(1,reference.height)
    asym=max(wx/max(.01,hy),hy/max(.01,wx))
    if asym>2.05 and max(wx,hy)>1.65:
        return False,"axis_expansion"
    if g.elongation>reference.elongation*2.1 and g.elongation>2.8:
        return False,"elongated_mask"
    if g.fill<reference.fill*.48 and g.area>reference.area*1.3:
        return False,"low_mask_fill"
    # Bounds on progressive mask spread; slowly drifting memory is not accepted forever.
    if g.area>reference.area*4.5 and asym>1.45:
        return False,"progressive_growth"
    if previous is not None:
        if g.area>previous.area*2.6 and asym>1.3:
            return False,"sudden_growth"
    return True,"ok"
