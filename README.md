# Target Locker

**Pick one object. Follow it through the video.**

Single-object tracking (SOT) with SAM2. Select a target on the first frame, then track its segmentation mask and bounding box across subsequent frames—without selecting it again.

**Click to select · Video input · Mask + box output · Colab GPU**

## Watch the demos

| Demo | Video |
| :--- | :--- |
| 01 | [Watch / download](demos/demo-01.mp4) |
| 02 | [Watch / download](demos/demo-02.mp4) |
| 03 | [Watch / download](demos/demo-03.mp4) |

## Try it in Colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CptImtiaz/Target_Locker/blob/main/SAM2_Colab_Target_Locker.ipynb)

1. Select a **GPU runtime** and run the setup cell.
2. Upload a video, click your target, and confirm.
3. Run tracking, preview the result, and download the MP4.

The notebook processes **300 frames by default**. Change `MAX_FRAMES` for longer clips. Output videos have no audio.

## What is SOT?

**Single-Object Tracking** follows one user-selected object through a sequence—even when other objects are visible. SAM2 supplies the masks; the displayed box is calculated from each mask. No separate object detector is used.

Tracking can drift or lose the target during occlusion or fast motion. Speed depends on the hardware; real-time performance is not guaranteed.

## Credits

Built on [SAM2_streaming](https://github.com/khw11044/SAM2_streaming) and Meta's [SAM 2](https://github.com/facebookresearch/sam2), using the SAM2.1 Tiny checkpoint.

Maintained by [Ahmed Imtiaz](https://github.com/CptImtiaz).
