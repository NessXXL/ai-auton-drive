import os
import sys
import json
import time
import numpy as np
from PIL import Image
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
    from helpers import detect_image

    with open(model_setup.paths["labels.json"]) as f:
        _labels = json.load(f)
    with open(model_setup.paths["anchors.json"]) as f:
        raw = json.load(f)
        _anchors = [np.array(a) for a in raw]

    _model = tf.keras.models.load_model(model_setup.paths["yolo_model.keras"])
    return _model, _labels, _anchors


RISK_THRESHOLDS = {
    "person": 3,
    "bicycle": 2,
    "motorcycle": 2,
    "bus": 2,
    "truck": 2,
    "traffic light": 1,
    "stop sign": 1,
}

def _compute_risk(boxes, labels):
    high_risk = {"person", "bicycle", "motorcycle", "bus", "truck"}
    med_risk = {"traffic light", "stop sign", "fire hydrant"}
    score = 0
    for box in boxes:
        label = labels[box.get_label()]
        if label in high_risk:
            score += 3
        elif label in med_risk:
            score += 1
        else:
            score += 0.5
    if score == 0:
        return "LOW", "#00ff88"
    elif score < 5:
        return "MEDIUM", "#ffcc00"
    else:
        return "HIGH", "#ff4444"


def run_detection(image_pil, obj_thresh=0.4):
    from helpers import detect_image, decode_netout, do_nms, preprocess_input, correct_yolo_boxes

    model, labels, anchors = _load_resources()

    t0 = time.perf_counter()
    image_w, image_h = image_pil.size
    net_h, net_w = 416, 416
    new_image = preprocess_input(image_pil, net_h, net_w)
    yolo_outputs = model.predict(new_image, verbose=0)
    boxes = decode_netout(yolo_outputs, obj_thresh, anchors, image_h, image_w, net_h, net_w)
    boxes = do_nms(boxes, 0.45, obj_thresh)
    t1 = time.perf_counter()

    elapsed = t1 - t0
    fps = 1.0 / elapsed if elapsed > 0 else 0.0

    result_img = detect_image(image_pil, model, anchors, labels, obj_thresh=obj_thresh)

    from collections import Counter
    counts = Counter(labels[b.get_label()] for b in boxes)
    risk_level, risk_color = _compute_risk(boxes, labels)

    stats_lines = []
    stats_lines.append(f"**Detected Objects:** {len(boxes)}")
    stats_lines.append(f"**Inference FPS:** {fps:.1f}")
    stats_lines.append(f"**Risk Level:** {risk_level}")
    stats_lines.append("")
    stats_lines.append("**Object Counts:**")
    if counts:
        for label_name, cnt in sorted(counts.items(), key=lambda x: -x[1]):
            stats_lines.append(f"- {label_name}: {cnt}")
    else:
        stats_lines.append("- No objects detected above threshold")

    stats_md = "\n".join(stats_lines)
    return result_img, stats_md, risk_level, fps


def predict(image, threshold):
    if image is None:
        return None, "No image provided.", "N/A", 0.0
    image_pil = Image.fromarray(image).convert("RGB")
    result_img, stats_md, risk_level, fps = run_detection(image_pil, obj_thresh=threshold)
    return np.array(result_img), stats_md


SAMPLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_images")
SAMPLE_IMAGES = sorted([
    os.path.join(SAMPLE_DIR, f)
    for f in os.listdir(SAMPLE_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]) if os.path.isdir(SAMPLE_DIR) else []


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Share+Tech+Mono&display=swap');

* { box-sizing: border-box; }

body, .gradio-container {
    background: #050a0f !important;
    color: #e0e8f0 !important;
    font-family: 'Inter', sans-serif !important;
}

.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
}

/* Hero */
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
@keyframes shimmer { 0% { background-position: 0% } 100% { background-position: 200% } }
.hero-tagline {
    font-size: 1.05rem;
    color: #7ec8e3;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin: 0;
    font-weight: 300;
}
.hero-badge {
    display: inline-block;
    margin-top: 1rem;
    padding: 0.3rem 1rem;
    background: #00d4ff15;
    border: 1px solid #00d4ff44;
    border-radius: 100px;
    font-size: 0.75rem;
    color: #00d4ff;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

/* Panels */
.panel-card {
    background: #0a1628 !important;
    border: 1px solid #1a2a40 !important;
    border-radius: 12px !important;
    overflow: hidden;
}

/* Stats box */
.stats-box {
    background: #070d1a;
    border: 1px solid #00d4ff22;
    border-radius: 10px;
    padding: 1.25rem 1.5rem;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.9rem;
    line-height: 1.8;
    color: #a0c8e0;
    min-height: 200px;
}
.stats-box strong { color: #00d4ff; }

/* Risk indicator */
.risk-indicator {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.75rem 1.25rem;
    border-radius: 8px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.85rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    border: 1px solid;
}

/* Gradio overrides */
.gr-button {
    background: linear-gradient(135deg, #00d4ff22, #7b2fff22) !important;
    border: 1px solid #00d4ff55 !important;
    color: #00d4ff !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    border-radius: 8px !important;
    transition: all 0.2s !important;
}
.gr-button:hover {
    background: linear-gradient(135deg, #00d4ff44, #7b2fff44) !important;
    border-color: #00d4ff !important;
    transform: translateY(-1px) !important;
}
.gr-button.primary {
    background: linear-gradient(135deg, #00d4ff, #7b2fff) !important;
    color: #050a0f !important;
    font-weight: 600 !important;
    border: none !important;
}
label, .gr-form label { color: #7ec8e3 !important; font-size: 0.85rem !important; letter-spacing: 0.05em !important; }
.gr-slider input[type=range] { accent-color: #00d4ff; }
.gr-markdown { color: #a0c8e0 !important; }
.gr-image { border: 1px solid #1a2a40 !important; border-radius: 10px !important; }
.gr-examples { background: #070d1a !important; border: 1px solid #1a2a40 !important; border-radius: 10px !important; }
footer { display: none !important; }
"""


def build_ui():
    _load_resources()

    with gr.Blocks(css=CSS, title="AI Auton Drive", theme=gr.themes.Base()) as demo:

        gr.HTML("""
        <div class="hero-section">
            <h1 class="hero-title">AI AUTON DRIVE</h1>
            <p class="hero-tagline">Real-Time Object Detection for Autonomous Driving</p>
            <span class="hero-badge">YOLO · TensorFlow · Computer Vision</span>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=5):
                input_img = gr.Image(
                    label="DASHCAM FEED",
                    type="numpy",
                    elem_classes=["panel-card"],
                    height=420,
                )
                threshold_slider = gr.Slider(
                    minimum=0.1, maximum=0.9, value=0.4, step=0.05,
                    label="CONFIDENCE THRESHOLD",
                    interactive=True,
                )
                detect_btn = gr.Button("⚡ RUN DETECTION", variant="primary", size="lg")

                if SAMPLE_IMAGES:
                    gr.Examples(
                        examples=SAMPLE_IMAGES,
                        inputs=[input_img],
                        label="SAMPLE DASHCAM CLIPS — click to load",
                        examples_per_page=4,
                    )

            with gr.Column(scale=5):
                output_img = gr.Image(
                    label="DETECTION OUTPUT",
                    type="numpy",
                    elem_classes=["panel-card"],
                    height=420,
                    interactive=False,
                )

                gr.HTML('<div style="height:0.75rem"></div>')

                stats_md = gr.Markdown(
                    value="*Run detection to see stats…*",
                    elem_classes=["stats-box"],
                )

        detect_btn.click(
            fn=predict,
            inputs=[input_img, threshold_slider],
            outputs=[output_img, stats_md],
        )

        input_img.change(
            fn=predict,
            inputs=[input_img, threshold_slider],
            outputs=[output_img, stats_md],
        )

    return demo


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=port, share=False, show_api=False)
