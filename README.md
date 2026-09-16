<div align="center">

# 🎯 Target Locker

### Lock one target. Track only that target.

**Persistent Single-Object Tracking (SOT) using SAM2.1 Tiny**

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-SAM2-EE4C2C?logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Tracking-5C3EE8?logo=opencv&logoColor=white)
![Task](https://img.shields.io/badge/Task-Single--Object%20Tracking-111827)

Select a single object in a video or webcam frame, then keep tracking **that same object only** with a segmentation mask and bounding box.

<a href="https://colab.research.google.com/github/CptImtiaz/Target_Locker/blob/main/SAM2_Colab_Target_Locker.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

</div>

---

## 🎬 Demo Previews

> GIF previews are the main focus so visitors can instantly see the tracker in action.

### Demo 01
<p align="center">
  <img src="demos/demo-01.gif" alt="Demo 01" width="900">
</p>

### Demo 02
<p align="center">
  <img src="demos/demo-02.gif" alt="Demo 02" width="900">
</p>

### Demo 03
<p align="center">
  <img src="demos/demo-03.gif" alt="Demo 03" width="900">
</p>

---

## ✨ Highlights

- 🎯 **Target Locking** — manually select one object
- 🧠 **SAM2.1 Tiny** — efficient mask-based tracking
- 🟩 **Mask + Bounding Box** — clear visualization of the tracked target
- 🎥 **Video Upload** — process your own recorded video
- 📷 **Colab Webcam Mode** — open the browser webcam, record a short clip, then track it on the Colab GPU
- 💻 **Local Webcam Mode** — direct camera access for real-time local use
- 💾 **Save Output** — export processed tracking videos
- ⚡ **No separate detector after initialization**

---

## ☁️ Run in Google Colab

Open the notebook directly:

[**SAM2_Colab_Target_Locker.ipynb**](SAM2_Colab_Target_Locker.ipynb)

or use the **Open in Colab** button at the top of this README.

### Colab input options

**1. Upload Video**  
Upload an existing video, click the target in the first frame, and process it with the GPU runtime.

**2. Webcam**  
Colab can open your **browser webcam** and record a short clip without requiring a local terminal or server. After recording, select the target and process the clip with SAM2.1 Tiny.

> Because Colab runs remotely, its webcam mode is **record → upload to runtime → track**, not direct low-latency `cv2.VideoCapture(0)` streaming. For continuous real-time webcam tracking, use the local version.

---

## ⚙️ How It Works

```text
Video Upload / Browser Webcam
            ↓
       Click target once
            ↓
       Initialize SAM2
            ↓
 Propagate target mask across frames
            ↓
    Track the same target only
            ↓
      Display mask + box
            ↓
       Save tracked result
```

---

## 🚀 Quick Workflow

1. Open the Colab notebook or local application.
2. Upload a video or record a webcam clip.
3. Click the object you want to lock.
4. SAM2 initializes that target.
5. The tracker follows the same object across subsequent frames.
6. The result is shown with a segmentation mask and bounding box.
7. Preview and save the tracked output.

---

## 💻 Runtime Notes

**macOS**: CPU or Apple GPU via **MPS**, with direct local webcam access.  
**Google Colab**: NVIDIA GPU with video upload, browser-webcam recording, tracking, preview, and download.

> Tracking speed depends on hardware, input resolution, and video length.

---

## ⚠️ Limitations

Tracking quality may drop during:

- heavy occlusion
- very fast motion
- large appearance changes
- target leaving the frame
- low-contrast or difficult scenes

Colab webcam input is recorded first and then processed; it is not continuous low-latency live streaming. Output videos currently contain **no audio**.

---

## 🧰 Built With

- [SAM2_streaming](https://github.com/khw11044/SAM2_streaming)
- [Meta SAM 2](https://github.com/facebookresearch/sam2)
- PyTorch
- OpenCV
- Python
- Google Colab

---

## 🙏 Acknowledgement

This project is based on **SAM2_streaming** and Meta's **SAM 2**.
