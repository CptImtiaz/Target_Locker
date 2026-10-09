"""Proposed Model 4: SAM2.1 local tracker + Grounding DINO global search + SigLIP2 target verifier.
RGB-only single-target prototype. Deliberately has an UNKNOWN/SEARCHING state to avoid forced switches.
"""
from __future__ import annotations
import csv
import gc
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from video_io import extract_rgb_frames, first_frame
from vlm_identity import SigLIPIdentity
from reacquisition import GlobalReacquirer
from confidence_policy import RecoveryGate, overlap
from mask_geometry import geometry, validate as validate_geometry
from object_memory import ObjectMemory
from camera_motion import CameraMotion
from distractor_memory import DistractorMemory

def bbox_from_mask(mask):
    ys,xs=np.where(mask)
    if len(xs)==0:return None
    return (int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1)

def mask_decode(logits,shape):
    h,w=shape
    if logits is None:return np.zeros((h,w),bool),0.
    tensor=logits[0] if isinstance(logits,torch.Tensor) else logits
    x=torch.sigmoid(tensor).detach().float().cpu().numpy().squeeze()
    if x.shape!=(h,w):
        x=cv2.resize(x,(w,h),interpolation=cv2.INTER_LINEAR)
    mask=x>.5
    quality=float(np.mean(x[mask])) if np.any(mask) else 0.
    return mask,quality

def valid_mask(mask,previous_area,initial_area):
    area=int(mask.sum())
    if area<max(6,int(initial_area*.025)):return False
    if area>.40*mask.size:return False
    if previous_area:
        ratio=area/max(1,previous_area)
        if ratio<.12 or ratio>6.:return False
    return True

class Sam2Session:
    """Adapter for the repository's SAM2.1 camera predictor."""
    def __init__(self,sam_workdir,checkpoint):
        root=str(Path(sam_workdir).resolve())
        if root not in sys.path:sys.path.insert(0,root)
        from sam2.build_sam import build_sam2_camera_predictor
        self.predictor=build_sam2_camera_predictor(
            "sam2.1/sam2.1_hiera_t.yaml",str(checkpoint),device="cuda")
        self.predictor.fill_hole_area=0

    def initialize(self,frame,point=None,bbox=None):
        p=self.predictor
        p.frame_idx=0
        p.load_first_frame(frame)
        if bbox is not None:
            coords=np.array(bbox,dtype=np.float32)
            _,_,logits=p.add_new_prompt(frame_idx=0,obj_id=1,bbox=coords)
        else:
            if point is None:raise ValueError("point or bbox is required")
            coords=np.array([[point[0],point[1]]],dtype=np.float32)
            _,_,logits=p.add_new_prompt(frame_idx=0,obj_id=1,points=coords,
                                         labels=np.array([1],dtype=np.int32))
        return logits

    def track(self,frame):
        _,logits=self.predictor.track(frame)
        return logits

    def close(self):
        if self.predictor is not None:
            try:self.predictor.condition_state={}
            except Exception:pass
            self.predictor=None
            gc.collect()
            if torch.cuda.is_available():torch.cuda.empty_cache()

def write_diagnostics(json_path,csv_path,summary,rows):
    Path(json_path).write_text(json.dumps(summary,indent=2),encoding="utf-8")
    with Path(csv_path).open("w",newline="",encoding="utf-8") as stream:
        out=csv.DictWriter(stream,fieldnames=[
            "frame","state","sam_mask_score","identity_similarity",
            "best_search_similarity","candidate_margin","candidates"])
        out.writeheader()
        out.writerows(rows)

def render(frame,mask,bbox,status,candidates):
    result=frame.copy()
    if bbox is not None:
        overlay=result.copy()
        overlay[mask]=(70,235,100)
        result=cv2.addWeighted(result,.77,overlay,.23,0)
        cv2.rectangle(result,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(40,235,100),2)
    else:
        for i,c in enumerate(candidates[:3]):
            x1,y1,x2,y2=c.bbox
            cv2.rectangle(result,(x1,y1),(x2,y2),(10,180,255),1)
            cv2.putText(result,f"candidate {i+1} {c.identity_score:.2f}",
                        (x1,max(14,y1-6)),cv2.FONT_HERSHEY_SIMPLEX,.43,
                        (10,180,255),1,cv2.LINE_AA)
    color=(40,235,100) if status in ("LOCKED","REACQUIRED") else (10,180,255)
    cv2.putText(result,"MODEL 4 | "+status,(16,30),cv2.FONT_HERSHEY_SIMPLEX,
                .74,color,2,cv2.LINE_AA)
    return result

def track_with_reacquisition(frames_dir,target_point,fps,output_path,
                            metrics_json,metrics_csv,sam_workdir,checkpoint,
                            target_description="",max_side=960,progress=None,
                            verify_every=4,search_every=2,
                            min_identity=.68,min_margin=.035,
                            detector_threshold=.24):
    """Returns (mp4 path, no-GT diagnostic summary). Absolute identity accuracy needs annotations."""
    paths=sorted(Path(frames_dir).glob("frame_*.png"))
    if not paths:raise ValueError("No input frames")
    first=cv2.imread(str(paths[0]))
    if first is None:raise RuntimeError("Unable to read first RGB frame")
    h,w=first.shape[:2]
    x=int(np.clip(target_point[0],0,w-1))
    y=int(np.clip(target_point[1],0,h-1))
    out_path=Path(output_path)
    out_path.parent.mkdir(parents=True,exist_ok=True)
    raw_path=out_path.with_name(out_path.stem+"_uncompressed.mp4")
    writer=cv2.VideoWriter(str(raw_path),cv2.VideoWriter_fourcc(*"mp4v"),float(fps),(w,h))
    if not writer.isOpened():raise RuntimeError("Cannot create output video")
    session=None
    rows=[]
    stats=dict(loss_events=0,global_search_frames=0,global_detector_calls=0,
               confirmed_reacquisitions=0,rejected_reacquisitions=0,
               identity_checks=0,locked_frames=0,geometry_rejections=0)
    identity_scores=[]
    memory=ObjectMemory(fps)
    camera=CameraMotion()
    distractors=DistractorMemory()
    start=time.perf_counter()
    search_candidates=[]
    gate=RecoveryGate(min_identity=min_identity,min_margin=min_margin,confirmations=2)
    last_bbox=None
    previous_area=None
    initial_area=1
    reference_geometry=None
    last_trusted_geometry=None
    state="LOCKED"
    first_mask=None
    first_quality=0.
    vlm=None;redetector=None
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU required; select Colab GPU runtime")
        amp=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.inference_mode(),torch.autocast("cuda",dtype=amp):
            session=Sam2Session(sam_workdir,checkpoint)
            logits=session.initialize(first,point=(x,y))
            first_mask,first_quality=mask_decode(logits,(h,w))
            first_bbox=bbox_from_mask(first_mask)
            if first_bbox is None or not valid_mask(first_mask,None,1):
                raise RuntimeError("SAM2 could not segment the selected target. Choose another point.")
            initial_area=int(first_mask.sum())
            reference_geometry=geometry(first_mask)
            last_trusted_geometry=reference_geometry
            previous_area=initial_area
            last_bbox=first_bbox

        # Load expensive models only after target selection succeeds.
        vlm=SigLIPIdentity(device="cuda")
        label=vlm.initialize(first,first_bbox,target_description)
        redetector=GlobalReacquirer(vlm,device="cuda",detector_threshold=detector_threshold)
        stats["target_label"]=label
        memory.category=label
        memory.observe(first,0,first_bbox,first_mask,force=True)

        for i,path in enumerate(paths):
            frame=first if i==0 else cv2.imread(str(path))
            if frame is None:raise RuntimeError(f"Unreadable frame: {path}")
            if frame.shape[:2]!=(h,w):frame=cv2.resize(frame,(w,h))
            camera.update(frame)
            box=None;mask=np.zeros((h,w),dtype=bool)
            sam_score=None;identity_score=None
            candidates=[]
            was_reacquired=False
            if i==0:
                mask=first_mask;sam_score=first_quality;box=first_bbox
            elif state=="LOCKED":
                with torch.inference_mode(),torch.autocast("cuda",dtype=amp):
                    logits=session.track(frame)
                    mask,sam_score=mask_decode(logits,(h,w))
                geometry_ok,geometry_reason=validate_geometry(mask,reference_geometry,last_trusted_geometry)
                if not geometry_ok:
                    stats['geometry_rejections']+=1
                if geometry_ok and valid_mask(mask,previous_area,initial_area):
                    box=bbox_from_mask(mask)
                if box is not None and (i%verify_every==0):
                    identity_score=vlm.compare(frame,[box])[0]
                    identity_scores.append(identity_score)
                    stats["identity_checks"]+=1
                    if identity_score<min_identity:
                        box=None
                if box is None:
                    state="SEARCHING"
                    stats["loss_events"]+=1
                    memory.events.append({"frame":i,"event":"lost"})
                    gate.reset()
                    mask=np.zeros((h,w),dtype=bool)
                    session.close()
                    session=None
                else:
                    previous_area=int(mask.sum())
                    last_trusted_geometry=geometry(mask)
                    last_bbox=box
                    if identity_score is not None:
                        vlm.update_trusted(frame,box,identity_score)
                        memory.observe(frame,i,box,mask,identity_score)
            if state=="SEARCHING":
                stats["global_search_frames"]+=1
                mask=np.zeros((h,w),dtype=bool)
                box=None
                if (i%search_every==0) or stats["global_search_frames"]==1:
                    candidates=memory.rerank(frame,redetector.search(frame),i)
                    candidates=distractors.penalize(frame,candidates,vlm,frame_index=i)
                    stats["global_detector_calls"]+=1
                    found=gate.consider(candidates)
                    if found is not None:
                        # A confirmed candidate still needs to initialize SAM2 and
                        # pass an independent mask + target-image verification.
                        proposed=Sam2Session(sam_workdir,checkpoint)
                        with torch.inference_mode(),torch.autocast("cuda",dtype=amp):
                            logits=proposed.initialize(frame,bbox=found.bbox)
                            new_mask,new_quality=mask_decode(logits,(h,w))
                        new_box=bbox_from_mask(new_mask)
                        new_geometry_ok,_=validate_geometry(new_mask,reference_geometry)
                        if not new_geometry_ok:
                            stats['geometry_rejections']+=1
                        okay=(new_box is not None and new_geometry_ok and
                              valid_mask(new_mask,None,initial_area) and
                              overlap(new_box,found.bbox)>=.10)
                        new_similarity=vlm.compare(frame,[new_box])[0] if okay else -1.
                        okay=okay and new_similarity>=min_identity
                        if okay:
                            session=proposed
                            state="LOCKED"
                            was_reacquired=True
                            stats["confirmed_reacquisitions"]+=1
                            memory.events.append({"frame":i,"event":"reacquired"})
                            memory.observe(frame,i,new_box,new_mask,new_similarity)
                            mask=new_mask;sam_score=new_quality;box=new_box
                            identity_score=new_similarity
                            previous_area=int(mask.sum())
                            last_trusted_geometry=geometry(mask)
                            last_bbox=new_box
                            gate.reset()
                        else:
                            proposed.close()
                            stats["rejected_reacquisitions"]+=1
                            distractors.observe(frame,found.bbox,vlm)
                            gate.reset()
            shown="REACQUIRED" if was_reacquired else state
            if box is not None:
                stats["locked_frames"]+=1
            row=dict(frame=i,state=shown,
                     sam_mask_score=round(sam_score,4) if sam_score is not None else "",
                     identity_similarity=round(identity_score,4) if identity_score is not None else "",
                     best_search_similarity=round(candidates[0].identity_score,4) if candidates else "",
                     candidate_margin=round(candidates[0].identity_score-candidates[1].identity_score,4) if len(candidates)>1 else "",
                     candidates=len(candidates))
            rows.append(row)
            writer.write(render(frame,mask,box,shown,candidates))
            if progress and (i%4==0 or i==len(paths)-1):progress("track",i+1,len(paths))
    finally:
        writer.release()
        if session is not None:session.close()

    elapsed=max(time.perf_counter()-start,1e-6)
    if shutil.which("ffmpeg"):
        try:
            subprocess.run(["ffmpeg","-nostdin","-y","-loglevel","error","-i",
                            str(raw_path),"-an","-c:v","libx264","-preset","fast",
                            "-crf","23","-pix_fmt","yuv420p","-movflags","+faststart",
                            str(out_path)],check=True)
            raw_path.unlink(missing_ok=True)
        except (subprocess.CalledProcessError,OSError):
            shutil.move(str(raw_path),str(out_path))
    else:
        shutil.move(str(raw_path),str(out_path))
    summary={
        "model":"Model 5: SAM2.1 + Grounding DINO + SigLIP 2 + universal memory",
        "target_label":stats["target_label"],
        "total_frames":len(rows),
        "loss_events":stats["loss_events"],
        "geometry_rejections":stats["geometry_rejections"],
        "global_search_frames":stats["global_search_frames"],
        "global_detector_calls":stats["global_detector_calls"],
        "confirmed_reacquisitions":stats["confirmed_reacquisitions"],
        "rejected_reacquisitions":stats["rejected_reacquisitions"],
        "ambiguous_rejections":gate.rejected_ambiguous,
        "identity_checks":stats["identity_checks"],
        "mean_vlm_similarity":round(float(np.mean(identity_scores)),4) if identity_scores else None,
        "locked_frame_fraction":round(stats["locked_frames"]/max(1,len(rows)),4),
        "processing_fps":round(len(rows)/elapsed,3),
        "ground_truth_accuracy":"not measured; diagnostics are not identity accuracy",
        "final_state":rows[-1]["state"],
        "universal_memory":memory.summary(),
        "camera_motion":camera.summary(),
        "distractor_memory":distractors.summary(),
    }
    write_diagnostics(metrics_json,metrics_csv,summary,rows)
    return out_path,summary
