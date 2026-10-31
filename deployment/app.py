"""
Crack segmentation web app (Gradio) - fixed settings, one page per model.

demo_ouc      -> OUC model
demo_volker   -> Volker model
demo_crack500 -> Crack500 model

Folder layout (either works):
  Colab : Drive/Drone_Wall_Crack/code + Drive/Drone_Wall_Crack/deployment
  Local / Hugging Face Space:
      app.py, code/ (copy of your code folder), weights/ (3 .pth files + deploy_config.json)
      Run one model per Space with the environment variable CRACK_MODEL=OUC, Volker or Crack500.
"""
import os
import sys
import json
import importlib

import cv2
import numpy as np
import torch
import gradio as gr
from PIL import Image, ImageOps

# ============================================================
# FIXED SETTINGS (edit here if you ever want different values)
# ============================================================
TILES = 3               # 3x3 overlapping tiles + the whole image (finds thin cracks)
STRONG_THRESHOLD = 0.50 # pixel is a sure crack at or above this probability
FAINT_THRESHOLD = 0.25  # fainter pixels are kept only if connected to a sure crack
OVERLAY_ALPHA = 0.6
MAX_SIDE = 1600

HERE = os.path.dirname(os.path.abspath(__file__))
DRIVE = "/content/drive/MyDrive/Drone_Wall_Crack"

CODE_DIR = os.path.join(HERE, "code") if os.path.isdir(os.path.join(HERE, "code")) else os.path.join(DRIVE, "code")
WEIGHTS_DIR = os.path.join(HERE, "weights") if os.path.isdir(os.path.join(HERE, "weights")) else os.path.join(DRIVE, "deployment")

if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

model_module = importlib.import_module("12_model")
prep_module = importlib.import_module("03_preprocessing")

TRANSFORM = prep_module.val_transform
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(os.path.join(WEIGHTS_DIR, "deploy_config.json")) as f:
    CONFIG = json.load(f)

_MODELS = {}


def get_model(name):
    if name not in _MODELS:
        m = model_module.CrackAnalysisModel().to(DEVICE)
        m.load_state_dict(torch.load(os.path.join(WEIGHTS_DIR, CONFIG[name]["weights"]),
                                     map_location=DEVICE))
        m.eval()
        _MODELS[name] = m
    return _MODELS[name]


# ============================================================
# INFERENCE
# ============================================================
@torch.no_grad()
def infer(model, rgb):
    h, w = rgb.shape[:2]
    x = TRANSFORM(image=rgb)["image"].unsqueeze(0).to(DEVICE)
    p = torch.sigmoid(model(x)["segmentation_logits"]).squeeze().cpu().numpy()
    return cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR)


def _starts(length, n, overlap):
    tile = min(int(np.ceil(length / (1 + (n - 1) * (1 - overlap)))), length)
    if n == 1:
        return [0], tile
    stride = (length - tile) / (n - 1)
    return [int(round(i * stride)) for i in range(n)], tile


def infer_tiled(model, rgb, n, overlap=0.25):
    h, w = rgb.shape[:2]
    ys, th = _starts(h, n, overlap)
    xs, tw = _starts(w, n, overlap)
    acc = np.zeros((h, w), np.float32)
    cnt = np.zeros((h, w), np.float32)
    for y in ys:
        for x in xs:
            acc[y:y + th, x:x + tw] += infer(model, np.ascontiguousarray(rgb[y:y + th, x:x + tw]))
            cnt[y:y + th, x:x + tw] += 1
    return acc / np.maximum(cnt, 1)


def hysteresis(prob, high, low):
    low = min(low, high)
    weak = (prob >= low).astype(np.uint8)
    n, labels = cv2.connectedComponents(weak, connectivity=8)
    keep = np.zeros(n, dtype=bool)
    keep[np.unique(labels[prob >= high])] = True
    keep[0] = False
    return keep[labels]


def run(image, model_name):

    if image is None:
        raise gr.Error("Please upload or take a photo first.")

    model = get_model(model_name)

    image = ImageOps.exif_transpose(image).convert("RGB")      # fixes rotated phone photos
    if max(image.size) > MAX_SIDE:
        s = MAX_SIDE / max(image.size)
        image = image.resize((int(image.width * s), int(image.height * s)), Image.LANCZOS)
    rgb = np.array(image)

    prob = np.maximum(infer(model, rgb), infer_tiled(model, rgb, TILES))
    mask = hysteresis(prob, STRONG_THRESHOLD, FAINT_THRESHOLD)

    overlay = rgb.astype(np.float32)
    overlay[mask] = (1 - OVERLAY_ALPHA) * overlay[mask] + OVERLAY_ALPHA * np.array([255, 0, 0], np.float32)

    pct = 100.0 * mask.mean()
    text = ("No clear crack found." if pct < 0.05 else "Crack detected.") + \
           f"  Crack pixels: {pct:.2f}% of the image."

    return overlay.astype(np.uint8), (mask.astype(np.uint8) * 255), text


# ============================================================
# PAGES
# ============================================================
def make_demo(model_name):

    with gr.Blocks(title=f"Crack Segmentation - {model_name}") as demo:

        gr.Markdown(f"# Crack Segmentation  ({model_name} model)\n"
                    "Upload a photo or take one with your phone, then press **Find cracks**.")

        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="pil", label="Image", sources=["upload", "webcam", "clipboard"])
                btn = gr.Button("Find cracks", variant="primary")
            with gr.Column():
                out_overlay = gr.Image(label="Cracks in red")
                out_mask = gr.Image(label="Crack mask")
                out_text = gr.Textbox(label="Result", lines=1)

        btn.click(lambda img: run(img, model_name), inp, [out_overlay, out_mask, out_text])

    return demo


demo_ouc = make_demo("OUC")
demo_volker = make_demo("Volker")
demo_crack500 = make_demo("Crack500")

if __name__ == "__main__":
    # Hugging Face Space / local run: one model per process
    make_demo(os.environ.get("CRACK_MODEL", "OUC")).launch()
