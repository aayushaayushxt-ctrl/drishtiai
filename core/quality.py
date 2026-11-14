"""Image-quality check (real, classical computation). Replace with a learned IQA model later."""
import numpy as np
from PIL import Image

def assess(img: Image.Image) -> dict:
    g = np.asarray(img.convert("L").resize((512, 512)), dtype=float)
    lap = g[1:-1, 1:-1] * 4 - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    focus = min(100, float(lap.var()) / 4)
    mean = float(g.mean()); bright = max(0, 100 - abs(mean - 95) * 1.1)
    contrast = min(100, float(g.std()) * 2.2)
    fov = min(100, float((g > 25).mean()) * 135)
    res = min(100, min(img.size) / 8)
    parts = {"Focus": focus, "Brightness": bright, "Contrast": contrast,
             "Retina visibility (field of view)": fov, "Resolution": res}
    score = int(round(0.3 * focus + 0.2 * bright + 0.15 * contrast + 0.25 * fov + 0.1 * res))
    label = "Good" if score >= 70 else "Acceptable" if score >= 50 else "Poor"
    return {"score": score, "label": label, "parts": {k: int(v) for k, v in parts.items()}}
