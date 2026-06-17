# Object Detection (YOLO): deployment notes for the build agent

This is a YOLO object-detection app. It takes an image and draws labeled boxes on
it using yolo_model.keras plus the functions in helpers.py (detect_image runs the
full pipeline).

DEPLOYMENT (must do):
- The simplest reliable setup is a SINGLE Gradio app, bound to host 0.0.0.0 and the
  port Replit provides. Run that one process directly as the web app.
- Do NOT put a separate API server or proxy in front of the Gradio app. Gradio
  already serves its own UI and API on one port. An extra server makes Replit route
  to the wrong port, so the Gradio endpoints (for example /gradio_api/...) return
  404. That shows up as a connection error and sample images that will not load.

KEY FILES:
- yolo_model.keras : trained detector (load with tf.keras)
- helpers.py       : detect_image(image, model, anchors, labels) + decode / NMS / draw
- labels.json      : class names
- anchors.json     : YOLO anchor boxes
- sample_images/   : real demo images for click-to-try
