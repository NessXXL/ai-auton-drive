"""Auto-generated. Provides `paths`. a dict mapping each project file to a
local path your app.py can load. Files that are too big for Replit live on the
HuggingFace Hub and are downloaded on first use; everything else stays local.

Usage in app.py:

    import model_setup

    # For Keras / TensorFlow:
    import tensorflow as tf
    model = tf.keras.models.load_model(model_setup.paths["my_model.h5"])

    # For scikit-learn / joblib:
    from joblib import load
    model = load(model_setup.paths["my_model.joblib"])

    # For HuggingFace Transformers (whole directory):
    from transformers import AutoModelForSequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained(model_setup.dir)

`paths` works for any filename present locally OR uploaded to HF Hub. Downloaded
Hub files are cached, so they only download once per environment.
"""
import json, os

_here = os.path.dirname(os.path.abspath(__file__))
dir = _here  # exposed for transformers/from_pretrained patterns

# 1. Every local file is reachable by its filename (including subdirectories)
paths = {}
for _root, _dirs, _fnames in os.walk(_here):
    for _fname in _fnames:
        _full = os.path.join(_root, _fname)
        _rel  = os.path.relpath(_full, _here)
        paths[_rel]   = _full
        paths[_fname] = _full

# 2. If model_config.json exists and points to HF Hub files, download them
#    and override the local entry. If config is missing or empty, we just use
#    local files. safe fallback for projects that don't need HF Hub.
_cfg_path = os.path.join(_here, "model_config.json")
if os.path.exists(_cfg_path):
    with open(_cfg_path) as _f:
        _cfg = json.load(_f)
    if _cfg.get("hub_repo_id"):
        from huggingface_hub import hf_hub_download
        for _fname in _cfg.get("hub_files", []):
            paths[_fname] = hf_hub_download(repo_id=_cfg["hub_repo_id"], filename=_fname)
