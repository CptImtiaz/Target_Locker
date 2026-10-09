"""SigLIP 2 image-image matching with optional text-category inference and immutable target anchor."""
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import Siglip2Model,AutoProcessor

CATEGORIES=("car","truck","bus","motorcycle","bicycle","person","dog","cat",
            "bird","boat","airplane","drone","tractor","animal","vehicle","object")

def crop_rgb(frame,bbox,pad=.12):
    h,w=frame.shape[:2]
    x1,y1,x2,y2=map(int,bbox)
    dx=int((x2-x1)*pad);dy=int((y2-y1)*pad)
    x1=max(0,x1-dx);y1=max(0,y1-dy);x2=min(w,x2+dx);y2=min(h,y2+dy)
    if x2<=x1 or y2<=y1:raise ValueError("Invalid image crop")
    return Image.fromarray(cv2.cvtColor(frame[y1:y2,x1:x2],cv2.COLOR_BGR2RGB))

class SigLIPIdentity:
    def __init__(self,device="cuda",model_id="google/siglip2-base-patch16-224"):
        self.device=torch.device(device)
        dtype=torch.float16 if self.device.type=="cuda" else torch.float32
        self.processor=AutoProcessor.from_pretrained(model_id)
        self.model=Siglip2Model.from_pretrained(model_id,torch_dtype=dtype).eval().to(self.device)
        self.anchor=None
        self.trusted=[]
        self.selected_label=None
        self.verifications=0

    def _image_inputs(self,images):
        data=self.processor(images=images,return_tensors="pt")
        dtype=next(self.model.parameters()).dtype
        return {k:(v.to(self.device,dtype=dtype) if k=="pixel_values" else v.to(self.device)) for k,v in data.items()}

    @torch.inference_mode()
    def images(self,images):
        out=self.model.get_image_features(**self._image_inputs(images))
        features=out.pooler_output if hasattr(out,"pooler_output") else (out if torch.is_tensor(out) else out[1])
        return F.normalize(features.float(),dim=-1)

    @torch.inference_mode()
    def _text(self,labels):
        data=self.processor(text=["a photo of a "+x for x in labels],
                            padding="max_length",return_tensors="pt")
        out=self.model.get_text_features(**{k:v.to(self.device) for k,v in data.items()})
        features=out.pooler_output if hasattr(out,"pooler_output") else (out if torch.is_tensor(out) else out[1])
        return F.normalize(features.float(),dim=-1)

    def initialize(self,frame,bbox,target_description=""):
        self.anchor=self.images([crop_rgb(frame,bbox)]).detach()
        self.trusted.clear()
        if target_description.strip():
            self.selected_label=target_description.strip().rstrip(".")
        else:
            score=(self.anchor@self._text(CATEGORIES).T)[0]
            self.selected_label=CATEGORIES[int(score.argmax())]
        return self.selected_label

    @torch.inference_mode()
    def compare(self,frame,boxes):
        if self.anchor is None:raise RuntimeError("Initialize target first")
        if not boxes:return []
        features=self.images([crop_rgb(frame,b) for b in boxes])
        anchor=(features@self.anchor.T).flatten()
        if self.trusted:
            bank=torch.cat(self.trusted,dim=0)
            gallery=(features@bank.T).max(dim=1).values
            scores=.8*anchor+.2*gallery
        else:
            scores=anchor
        self.verifications+=len(boxes)
        return [float(n) for n in scores.cpu().tolist()]

    @torch.inference_mode()
    def update_trusted(self,frame,bbox,identity_score):
        if identity_score<.82:return
        self.trusted.append(self.images([crop_rgb(frame,bbox)]).detach())
        self.trusted=self.trusted[-5:]
