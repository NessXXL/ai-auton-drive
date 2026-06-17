# 🚗 AI Auton Drive

**Real-time object detection for autonomous driving assistance** — powered by YOLOv3 and TensorFlow.

Upload a dashcam image or video and get instant detection of cars, pedestrians, traffic lights, signs, and more, with neon-colored bounding boxes and confidence scores overlaid directly on the output.

---

## ✨ Features

- **Image detection** — upload any dashcam photo and see labeled bounding boxes in under a second
- **Video analysis** — processes up to 60 seconds of dashcam footage with frame-by-frame object detection
- **80 COCO object classes** — cars, trucks, buses, motorcycles, bicycles, pedestrians, traffic lights, stop signs, and more
- **Risk level indicator** — automatically rates each scene LOW / MEDIUM / HIGH based on detected hazards
- **Detection stats panel** — object counts by class, inference FPS, overall risk level
- **Smart frame skipping** — runs YOLO every Nth frame and reuses boxes on intermediate frames for fast video processing
- **Futuristic dark UI** — neon-accented interface built with Gradio
- **Pre-loaded sample clips** — bundled dashcam images and video to try instantly

---

## 🖼️ Demo

| Input | Output |
|-------|--------|
| Dashcam image | Labeled boxes: `car 92%`, `traffic light 87%`, `person 78%` |
| Dashcam video | Fully annotated MP4 with per-frame detection |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Model | YOLOv3 (Keras / TensorFlow) |
| Model hosting | HuggingFace Hub (`NessXXL/ai-auton-drive-model`) |
| Backend | Python 3.11 |
| UI | Gradio 6 |
| Image processing | OpenCV, Pillow |
| Runtime | CPU (no GPU required) |

---

## 📁 Project Structure

```
artifacts/ai-auton-drive/
├── app.py              # Main Gradio app — UI + inference pipeline
├── helpers.py          # YOLO decode / NMS / drawing helpers
├── model_setup.py      # Auto-downloads model from HuggingFace Hub
├── model_config.json   # HuggingFace repo + file list
├── labels.json         # 80 COCO class names
├── anchors.json        # YOLOv3 anchor boxes
├── requirements.txt    # Python dependencies
└── sample_images/      # Bundled dashcam images + video for quick demos
```

---

## 🚀 Running Locally

### Prerequisites

- Python 3.10 or 3.11
- pip

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/ai-auton-drive.git
cd ai-auton-drive
```

### 2. Install dependencies

```bash
pip install -r artifacts/ai-auton-drive/requirements.txt
```

> **Note:** `tensorflow` will be installed. If you want GPU support, replace it with `tensorflow[and-cuda]` in requirements.txt.

### 3. Run the app

```bash
python artifacts/ai-auton-drive/app.py
```

The app will:
1. Download `yolo_model.keras` (~249 MB) from HuggingFace Hub on first run — this only happens once
2. Start a Gradio server at `http://localhost:7860`

Open that URL in your browser and start detecting!

---

## ⚙️ Configuration

| Setting | Where | Default |
|---------|-------|---------|
| Confidence threshold | UI slider | 0.50 |
| Max video length | `MAX_SECONDS` in `app.py` | 60 s |
| Detection rate | `TARGET_DETECT_FPS` in `app.py` | 5 fps |
| Server port | `PORT` env var | 7860 |

### Tuning confidence threshold

- **Lower (0.3–0.4):** catches more objects, more false positives
- **Default (0.5):** balanced — good for typical dashcam footage
- **Higher (0.6–0.7):** only high-confidence detections, fewer false positives

---

## 📦 Dependencies

```
gradio
tensorflow           # or tensorflow-cpu for CPU-only environments
numpy
Pillow
opencv-python-headless
huggingface_hub
```

---

## 🤖 Model Details

- **Architecture:** YOLOv3
- **Dataset:** Trained on COCO (80 classes)
- **Input size:** 416 × 416 (images are auto-resized)
- **Model file:** `yolo_model.keras` (~249 MB, downloaded automatically)
- **HuggingFace repo:** [NessXXL/ai-auton-drive-model](https://huggingface.co/NessXXL/ai-auton-drive-model)

---

## 📄 License

MIT — see [LICENSE](LICENSE) for details.
