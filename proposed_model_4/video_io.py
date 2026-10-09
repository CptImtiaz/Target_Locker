"""Video decoding and scaled RGB frame storage for Colab app."""
from pathlib import Path
import shutil
import cv2
import numpy as np

def resize_even(frame,max_side=960):
    h,w=frame.shape[:2];scale=min(1.,max_side/max(h,w))
    new_w=max(2,(int(w*scale)//2)*2)
    new_h=max(2,(int(h*scale)//2)*2)
    if (new_w,new_h)!=(w,h):
        return cv2.resize(frame,(new_w,new_h),interpolation=cv2.INTER_AREA)
    return frame

def extract_rgb_frames(video_path,frames_dir,max_side=960,progress=None):
    dest=Path(frames_dir)
    if dest.exists():shutil.rmtree(dest)
    dest.mkdir(parents=True,exist_ok=True)
    cap=cv2.VideoCapture(str(video_path))
    if not cap.isOpened():raise RuntimeError("Video could not be opened")
    fps=float(cap.get(cv2.CAP_PROP_FPS))
    if not fps or not np.isfinite(fps):fps=25.
    source_width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total=max(1,int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    index=0;w=h=0
    try:
        while True:
            ok,frame=cap.read()
            if not ok:break
            frame=resize_even(frame,max_side)
            h,w=frame.shape[:2]
            if not cv2.imwrite(str(dest/f"frame_{index:06d}.png"),frame):
                raise RuntimeError("Cannot write video frame")
            index+=1
            if progress and (index%8==0 or index==total):
                progress("extract",index,max(index,total))
    finally:cap.release()
    if index==0:raise RuntimeError("No video frames decoded")
    return dict(fps=fps,frames=index,width=w,height=h,
                source_width=source_width,source_height=source_height)

def first_frame(frames_dir):
    paths=sorted(Path(frames_dir).glob("frame_*.png"))
    if not paths:raise RuntimeError("No frames found")
    frame=cv2.imread(str(paths[0]))
    if frame is None:raise RuntimeError("Unreadable first frame")
    return frame
