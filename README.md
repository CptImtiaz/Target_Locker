<div align="center">

# 🎯 Target Locker

### Lock one target. Track only that target.

**Persistent Single-Object Tracking (SOT) with SAM2.1 Tiny**

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-SAM2-EE4C2C?logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Tracking-5C3EE8?logo=opencv&logoColor=white)
![SOT](https://img.shields.io/badge/Task-Single--Object%20Tracking-111827)

Select one object in a video or webcam frame and **Target Locker keeps following that same object** with a segmentation mask and bounding box.

No separate object detector is required after target initialization.

</div>

---

## 🎬 Live Demos

> The previews below are configured to play muted and loop automatically when supported by GitHub/browser rendering.

### Demo 01

<video src="demos/demo-01.mp4" width="100%" autoplay muted loop playsinline controls></video>

### Demo 02

<video src="demos/demo-02.mp4" width="100%" autoplay muted loop playsinline controls></video>

### Demo 03

<video src="demos/demo-03.mp4" width="100%" autoplay muted loop playsinline controls></video>

<details>
<summary><b>Fallback previews</b></summary>
<br>

| Demo 01 | Demo 02 | Demo 03 |
| :---: | :---: | :---: |
| <img src="demos/demo-01.jpg" width="260"> | <img src="demos/demo-02.jpg" width="260"> | <img src="demos/demo-03.jpg" width="260"> |

</details>

---

## ✨ What It Does

| Feature | Description |
|---|---|
| 🎯 **Target Lock** | Select one object manually |
| 🧠 **SAM2.1 Tiny** | Segmentation-based target propagation |
| 📦 **Single Target Only** | Tracks the selected object instead of every object |
| 🟩 **Mask + Box** | Visualizes segmentation mask and bounding box |
| 🎥 **Video Input** | Process uploaded videos |
| 📷 **Webcam Input** | Supports live camera tracking |
| 💾 **Video Export** | Save the tracked result |

---

## ⚙️ How It Works

```text
Video / Webcam
      ↓
Select one target
      ↓
Initialize SAM2
      ↓
Propagate target mask
      ↓
Track the same object frame-by-frame
      ↓
Mask + Bounding Box
      ↓
Export tracked video
```

---

## 🚀 Workflow

1. Open a **video** or **webcam** stream.
2. Click the object you want to lock.
3. SAM2 initializes that target from your selection.
4. The tracker follows the same target across subsequent frames.
5. The result is displayed with a **segmentation mask + bounding box**.
6. Save the processed video.

---

## 💻 Runtime

**macOS**  
CPU or Apple Silicon GPU through **MPS**.

**Google Colab**  
GPU runtime with video upload, processing, preview, and download.

> Tracking speed depends on the available hardware, input resolution, and video length.

---

## ⚠️ Current Limitations

Target tracking may become less stable during:

- heavy occlusion
- very fast target motion
- major appearance changes
- target leaving the frame
- difficult low-contrast scenes

Output videos currently contain **no audio**.

---

## 🧰 Built With

- [SAM2_streaming](https://github.com/khw11044/SAM2_streaming)
- [Meta Segment Anything 2](https://github.com/facebookresearch/sam2)
- PyTorch
- OpenCV
- Python

---

## 🙏 Acknowledgement

This project is based on **SAM2_streaming** and Meta's **SAM 2** research/codebase.

SAM2_streaming is distributed under the **Apache 2.0 License**.

---

<div align="center">

### 🎯 Select → Lock → Track

**Target Locker — Persistent Single-Object Tracking**

</div>
