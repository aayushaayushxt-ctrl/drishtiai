"""
============== MODEL INTEGRATION POINT ==============
analyze(image) -> dict (the API contract the UI consumes):
  {"label": "Healthy"|"Diseased", "confidence": 0-100, "heatmap": ndarray HxW in [0,1],
   "source": "real_model"|"mock", "classes": [...]}
Uses your real PyTorch model (model/predict.py + model/grad_cam.py) when torch is available.
To use a FastAPI service instead, replace _real() with requests.post(API_URL, files=...).
"""
import sys, pathlib, numpy as np
from PIL import Image
HERE = pathlib.Path(__file__).resolve().parent.parent / "model"
CKPT = HERE / "best_model.pth"
sys.path.insert(0, str(HERE))
MODEL_INFO = {"Model type": "CNN classifier (from checkpoint) + Grad-CAM",
              "Classes": "Healthy / Diseased (binary)", "Training data": "APTOS 2019 (per project notes)",
              "Validation": "Not independently validated — prototype", "Version": "v0.1-prototype"}

def _real(img):
    from grad_cam import explain  # imports predict.load_model
    pred, heat = explain(img, CKPT)
    return {"label": pred["predicted"], "confidence": pred["score_percent"], "heatmap": heat, "source": "real_model"}

def _mock(img):
    w, h = img.size; yy, xx = np.mgrid[0:h, 0:w]
    rng = np.random.default_rng(int(np.asarray(img.resize((8, 8))).sum()) % 9999)
    heat = np.zeros((h, w))
    for _ in range(3):
        cx, cy = rng.uniform(.25, .75) * w, rng.uniform(.25, .75) * h
        heat += np.exp(-(((xx - cx) / (w * .12)) ** 2 + ((yy - cy) / (h * .12)) ** 2))
    heat /= heat.max()
    return {"label": "Diseased" if rng.random() > .5 else "Healthy", "confidence": float(rng.uniform(72, 93)),
            "heatmap": heat, "source": "mock"}

def analyze(img: Image.Image) -> dict:
    try:
        if CKPT.exists():
            return _real(img)
    except Exception as e:  # torch missing / checkpoint issue -> clearly-labelled mock
        err = str(e)[:120]
        r = _mock(img); r["note"] = f"Real model unavailable ({err}); showing MOCK output."; return r
    return _mock(img)

def overlay(img, heat, alpha):
    pal = np.array([[15, 22, 52], [51, 47, 127], [149, 51, 122], [235, 111, 67], [255, 235, 132]], float)
    lv = np.linspace(0, 1, 5); col = np.stack([np.interp(heat, lv, c) for c in pal.T], -1)
    a = (alpha * heat)[..., None]; ph = np.asarray(img.convert("RGB"), float)
    return Image.fromarray(((1 - a) * ph + a * col).astype(np.uint8))

def demo_fundus(seed=1, size=512):
    """Synthetic retina-like image for Demo Mode (NOT a real patient image)."""
    r = np.random.default_rng(seed); yy, xx = np.mgrid[0:size, 0:size] - size / 2
    d = np.hypot(xx, yy) / (size / 2); img = np.zeros((size, size, 3))
    base = np.array([190, 80, 40]) * (1 - .45 * d[..., None] ** 2)
    img[:] = base; img[d > 1] = 0
    od = np.exp(-(((xx - size * .18) ** 2 + (yy + size * .02) ** 2) / (size * .06) ** 2))
    img += od[..., None] * np.array([70, 90, 30])
    for _ in range(14):
        ang = r.uniform(0, 6.28); L = np.linspace(0, size * .45, 120)
        wig = np.cumsum(r.normal(0, 2, 120))
        px = (size / 2 + size * .18 * 0 + L * np.cos(ang)).astype(int).clip(0, size - 1)
        py = (size / 2 + L * np.sin(ang) + wig).astype(int).clip(0, size - 1)
        img[py, px] = [110, 20, 15]
    from PIL import ImageFilter
    im = Image.fromarray(img.clip(0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.2))
    return im
