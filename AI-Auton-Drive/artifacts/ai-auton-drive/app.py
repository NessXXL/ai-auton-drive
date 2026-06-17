import os
import sys
import json
import time
import tempfile
import numpy as np
from PIL import Image
import cv2
import gradio as gr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_setup

_model = None
_labels = None
_anchors = None

def _load_resources():
    global _model, _labels, _anchors
    if _model is not None:
        return _model, _labels, _anchors

    import tensorflow as tf

    with open(model_setup.paths["labels.json"]) as f:
        _labels = json.load(f)
    with open(model_setup.paths["anchors.json"]) as f:
        raw = json.load(f)
        _anchors = [np.array(a) for a in raw]

    _model = tf.keras.models.load_model(model_setup.paths["yolo_model.keras"])
    return _model, _labels, _anchors


# ── Neon color palette per class index ───────────────────────────────────────
_PALETTE = [
    (0, 212, 255),   # cyan
    (123, 47, 255),  # purple
    (0, 255, 136),   # green
    (255, 204, 0),   # yellow
    (255, 68, 68),   # red
    (255, 128, 0),   # orange
    (0, 255, 255),   # aqua
    (255, 0, 200),   # magenta
    (100, 255, 100), # lime
    (255, 80, 180),  # pink
]

def _class_color(class_idx):
    r, g, b = _PALETTE[class_idx % len(_PALETTE)]
    return (b, g, r)  # OpenCV uses BGR


def draw_detections(frame_bgr, boxes, labels):
    """
    Draw neon bounding boxes + label badges onto a BGR numpy frame.
    Returns a new frame with annotations.
    """
    h, w = frame_bgr.shape[:2]
    out = frame_bgr.copy()

    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = max(0.45, min(w, h) / 900)
    box_thick = max(2, int(min(w, h) / 200))
    pad = 4

    for box in boxes:
        c = box.get_label()
        score = box.get_score()
        label_name = labels[c]
        color_bgr = _class_color(c)

        x1 = max(0, int(box.xmin))
        y1 = max(0, int(box.ymin))
        x2 = min(w - 1, int(box.xmax))
        y2 = min(h - 1, int(box.ymax))

        if x2 <= x1 or y2 <= y1:
            continue

        # Bounding box
        cv2.rectangle(out, (x1, y1), (x2, y2), color_bgr, box_thick)

        # Label text
        text = f"{label_name}  {score:.0%}"
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, 1)

        # Badge position: above box, clamp to frame
        bx1 = x1
        by1 = y1 - th - pad * 2 - baseline
        bx2 = x1 + tw + pad * 2
        by2 = y1

        if by1 < 0:          # flip below if no space above
            by1 = y2
            by2 = y2 + th + pad * 2 + baseline

        # Dark filled badge
        overlay = out.copy()
        cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (10, 12, 18), -1)
        cv2.addWeighted(overlay, 0.75, out, 0.25, 0, out)

        # Colored top border on badge
        cv2.rectangle(out, (bx1, by1), (bx2, by1 + 2), color_bgr, -1)

        # White text
        tx = bx1 + pad
        ty = by2 - pad - baseline
        cv2.putText(out, text, (tx, ty), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    return out


# ── Detection helpers ─────────────────────────────────────────────────────────

def _detect_boxes(image_pil, model, labels, anchors, obj_thresh):
    from helpers import preprocess_input, decode_netout, do_nms
    iw, ih = image_pil.size
    net_h = net_w = 416
    inp = preprocess_input(image_pil, net_h, net_w)
    yolo_out = model.predict(inp, verbose=0)
    boxes = decode_netout(yolo_out, obj_thresh, anchors, ih, iw, net_h, net_w)
    boxes = do_nms(boxes, 0.45, obj_thresh)
    return boxes


def _compute_risk(boxes, labels):
    high_risk = {"person", "bicycle", "motorcycle", "bus", "truck"}
    med_risk = {"traffic light", "stop sign", "fire hydrant"}
    score = 0
    for box in boxes:
        name = labels[box.get_label()]
        score += 3 if name in high_risk else (1 if name in med_risk else 0.5)
    if score == 0:   return "LOW"
    if score < 5:    return "MEDIUM"
    return "HIGH"


def _build_stats(boxes, labels, fps, extra_lines=None):
    from collections import Counter
    counts = Counter(labels[b.get_label()] for b in boxes)
    risk = _compute_risk(boxes, labels)
    lines = [
        f"**Detected Objects:** {len(boxes)}",
        f"**Inference FPS:** {fps:.1f}",
        f"**Risk Level:** {risk}",
        "",
        "**Object Counts:**",
    ]
    if counts:
        for name, cnt in sorted(counts.items(), key=lambda x: -x[1]):
            lines.append(f"- {name}: {cnt}")
    else:
        lines.append("- No objects detected above threshold")
    if extra_lines:
        lines += extra_lines
    return "\n".join(lines)


# ── Gradio handlers ───────────────────────────────────────────────────────────

def predict_image(image, threshold):
    if image is None:
        return None, "*Upload or select a sample image to run detection.*"

    model, labels, anchors = _load_resources()
    image_pil = Image.fromarray(image).convert("RGB")

    t0 = time.perf_counter()
    boxes = _detect_boxes(image_pil, model, labels, anchors, threshold)
    fps = 1.0 / max(time.perf_counter() - t0, 1e-6)

    frame_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    result_bgr = draw_detections(frame_bgr, boxes, labels)
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

    return result_rgb, _build_stats(boxes, labels, fps)


MAX_SECONDS = 60  # process up to 60 seconds of video

def predict_video(video_path, threshold, progress=gr.Progress()):
    if video_path is None:
        return None, "*Upload a dashcam video and click Analyse to run detection.*"

    from collections import Counter

    model, labels, anchors = _load_resources()

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Cap at MAX_SECONDS worth of frames
    max_frames = min(total_frames, int(fps_in * MAX_SECONDS))

    # Smart stride: run YOLO every Nth frame, reuse last boxes for the rest.
    # Target ~5 YOLO detections per second of video — keeps quality high while
    # making 60fps / 1080p videos finish in a reasonable time.
    TARGET_DETECT_FPS = 5
    stride = max(1, round(fps_in / TARGET_DETECT_FPS))

    out_path = tempfile.mktemp(suffix=".mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps_in, (width, height))

    all_counts = Counter()
    risk_scores = []
    inf_times = []
    last_boxes = []          # reused on stride-skipped frames

    progress(0, desc="Starting video analysis…")

    for i in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break

        if i % stride == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(frame_rgb)

            t0 = time.perf_counter()
            last_boxes = _detect_boxes(image_pil, model, labels, anchors, threshold)
            inf_times.append(time.perf_counter() - t0)

            for b in last_boxes:
                all_counts[labels[b.get_label()]] += 1
            risk_scores.append({"LOW": 0, "MEDIUM": 1, "HIGH": 2}[_compute_risk(last_boxes, labels)])

        annotated = draw_detections(frame, last_boxes, labels)
        writer.write(annotated)

        progress((i + 1) / max_frames,
                 desc=f"Frame {i+1} / {max_frames}  (YOLO every {stride} frames)")

    cap.release()
    writer.release()

    avg_fps = 1.0 / max(float(np.mean(inf_times)), 1e-6) if inf_times else 0
    avg_risk = np.mean(risk_scores) if risk_scores else 0
    overall_risk = "LOW" if avg_risk < 0.5 else ("MEDIUM" if avg_risk < 1.5 else "HIGH")
    duration_s = max_frames / fps_in
    detected_frames = len(inf_times)

    lines = [
        f"**Video Duration:** {duration_s:.1f}s  ({max_frames} frames @ {fps_in:.0f}fps)",
        f"**Frames Detected:** {detected_frames}  (every {stride} frames)",
        f"**Avg Inference FPS:** {avg_fps:.1f}",
        f"**Overall Risk Level:** {overall_risk}",
        "",
        "**Total Detections by Class:**",
    ]
    if all_counts:
        for name, cnt in sorted(all_counts.items(), key=lambda x: -x[1])[:12]:
            lines.append(f"- {name}: {cnt}")
    else:
        lines.append("- No objects detected above threshold")
    if total_frames > max_frames:
        lines.append(f"\n*Note: video was longer — processed first {MAX_SECONDS}s.*")

    return out_path, "\n".join(lines)


# ── Assets ────────────────────────────────────────────────────────────────────

SAMPLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_images")

SAMPLE_IMAGES = sorted([
    os.path.join(SAMPLE_DIR, f)
    for f in os.listdir(SAMPLE_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]) if os.path.isdir(SAMPLE_DIR) else []

SAMPLE_VIDEOS = sorted([
    os.path.join(SAMPLE_DIR, f)
    for f in os.listdir(SAMPLE_DIR)
    if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
]) if os.path.isdir(SAMPLE_DIR) else []


# ── CSS ───────────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Share+Tech+Mono&display=swap');

* { box-sizing: border-box; }

body, .gradio-container {
    background: #050a0f !important;
    color: #e0e8f0 !important;
    font-family: 'Inter', sans-serif !important;
}
.gradio-container { max-width: 1200px !important; margin: 0 auto !important; }

.hero-section {
    background: linear-gradient(135deg, #050a0f 0%, #0a1628 50%, #050a0f 100%);
    border-bottom: 1px solid #00d4ff22;
    padding: 2.5rem 2rem 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.hero-section::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%; width: 200%; height: 200%;
    background: radial-gradient(ellipse at center, #00d4ff08 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Share Tech Mono', monospace;
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(90deg, #00d4ff, #7b2fff, #00d4ff);
    background-size: 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.5rem;
    letter-spacing: 0.04em;
    animation: shimmer 4s linear infinite;
}
@keyframes shimmer { 0%{background-position:0%} 100%{background-position:200%} }
.hero-tagline { font-size: 1.05rem; color: #7ec8e3; letter-spacing: .08em; text-transform: uppercase; margin: 0; font-weight: 300; }
.hero-badge {
    display: inline-block; margin-top: 1rem; padding: .3rem 1rem;
    background: #00d4ff15; border: 1px solid #00d4ff44; border-radius: 100px;
    font-size: .75rem; color: #00d4ff; letter-spacing: .1em; text-transform: uppercase;
}

.panel-card { background: #0a1628 !important; border: 1px solid #1a2a40 !important; border-radius: 12px !important; overflow: hidden; }

.stats-box {
    background: #070d1a; border: 1px solid #00d4ff22; border-radius: 10px;
    padding: 1.25rem 1.5rem;
    font-family: 'Share Tech Mono', monospace; font-size: .9rem; line-height: 1.8;
    color: #a0c8e0; min-height: 180px;
}
.stats-box strong { color: #00d4ff; }
.stats-box p { margin: 0; }

.tab-nav button {
    background: transparent !important; border: none !important;
    border-bottom: 2px solid transparent !important;
    color: #7ec8e3 !important; font-family: 'Share Tech Mono', monospace !important;
    font-size: .85rem !important; letter-spacing: .1em !important;
    text-transform: uppercase !important; padding: .6rem 1.5rem !important; transition: all .2s !important;
}
.tab-nav button.selected { border-bottom-color: #00d4ff !important; color: #00d4ff !important; }

button.primary {
    background: linear-gradient(135deg,#00d4ff,#7b2fff) !important;
    color: #050a0f !important; font-weight: 600 !important;
    border: none !important; border-radius: 8px !important;
}
button.secondary {
    background: #00d4ff15 !important; border: 1px solid #00d4ff44 !important;
    color: #00d4ff !important; border-radius: 8px !important;
}

label, .label-wrap { color: #7ec8e3 !important; font-size: .8rem !important; letter-spacing: .06em !important; text-transform: uppercase !important; }
input[type=range] { accent-color: #00d4ff !important; }
.gr-examples, .examples { background: #070d1a !important; border: 1px solid #1a2a40 !important; border-radius: 10px !important; }
footer { display: none !important; }
"""


# ── UI ────────────────────────────────────────────────────────────────────────

def build_ui():
    _load_resources()

    with gr.Blocks(title="AI Auton Drive") as demo:

        gr.HTML("""
        <div class="hero-section">
            <h1 class="hero-title">AI AUTON DRIVE</h1>
            <p class="hero-tagline">Real-Time Object Detection for Autonomous Driving</p>
            <span class="hero-badge">YOLO · TensorFlow · Computer Vision</span>
        </div>
        """)

        with gr.Tabs():

            # ── IMAGE TAB ──────────────────────────────────────────────
            with gr.Tab("📷  Image"):
                with gr.Row():
                    with gr.Column(scale=5):
                        input_img = gr.Image(
                            label="DASHCAM FEED", type="numpy",
                            elem_classes=["panel-card"], height=420,
                        )
                        img_threshold = gr.Slider(
                            minimum=0.1, maximum=0.9, value=0.5, step=0.05,
                            label="CONFIDENCE THRESHOLD", interactive=True,
                        )
                        img_btn = gr.Button("⚡ RUN DETECTION", variant="primary", size="lg")
                        if SAMPLE_IMAGES:
                            gr.Examples(
                                examples=SAMPLE_IMAGES, inputs=[input_img],
                                label="SAMPLE DASHCAM IMAGES — click to load",
                                examples_per_page=4,
                            )

                    with gr.Column(scale=5):
                        output_img = gr.Image(
                            label="DETECTION OUTPUT", type="numpy",
                            elem_classes=["panel-card"], height=420, interactive=False,
                        )
                        gr.HTML('<div style="height:.5rem"></div>')
                        img_stats = gr.Markdown(
                            value="*Upload or select a sample image to run detection.*",
                            elem_classes=["stats-box"],
                        )

                img_btn.click(fn=predict_image, inputs=[input_img, img_threshold], outputs=[output_img, img_stats])
                input_img.change(fn=predict_image, inputs=[input_img, img_threshold], outputs=[output_img, img_stats])

            # ── VIDEO TAB ──────────────────────────────────────────────
            with gr.Tab("🎥  Video"):
                with gr.Row():
                    with gr.Column(scale=5):
                        input_vid = gr.Video(
                            label="DASHCAM VIDEO",
                            elem_classes=["panel-card"], height=420,
                        )
                        vid_threshold = gr.Slider(
                            minimum=0.1, maximum=0.9, value=0.5, step=0.05,
                            label="CONFIDENCE THRESHOLD", interactive=True,
                        )
                        vid_btn = gr.Button("⚡ ANALYSE VIDEO", variant="primary", size="lg")
                        if SAMPLE_VIDEOS:
                            gr.Examples(
                                examples=SAMPLE_VIDEOS, inputs=[input_vid],
                                label="SAMPLE DASHCAM CLIPS — click to load",
                                examples_per_page=2,
                            )

                    with gr.Column(scale=5):
                        output_vid = gr.Video(
                            label="ANNOTATED OUTPUT",
                            elem_classes=["panel-card"], height=420, interactive=False,
                        )
                        gr.HTML('<div style="height:.5rem"></div>')
                        vid_stats = gr.Markdown(
                            value="*Upload a dashcam video and click Analyse to run detection.*",
                            elem_classes=["stats-box"],
                        )

                vid_btn.click(fn=predict_video, inputs=[input_vid, vid_threshold], outputs=[output_vid, vid_stats])

    return demo


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0", server_port=port,
        share=False, css=CSS, theme=gr.themes.Base(),
    )
