#!/usr/bin/env python3
"""
predict.py - Classify a single retinal image with a model trained by train.py.

COMMAND LINE
------------
  python predict.py path/to/image.jpg
  python predict.py path/to/image.jpg --checkpoint models/best_model.pth --top-k 3

FROM PYTHON
-----------
  from predict import predict
  label, confidence = predict("path/to/image.jpg")
  print(label, f"{confidence:.2f}%")      # e.g. "Glaucoma 93.41%"

The model is loaded once and cached, so calling predict() repeatedly is fast.
This file is standalone: it does not import train.py. Everything it needs
(architecture, class names, image size, normalisation) is stored in the checkpoint.
"""

import argparse
import functools
import sys
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms

DEFAULT_CHECKPOINT = "models/best_model.pth"


def _get_device() -> torch.device:
    """NVIDIA GPU > Apple Silicon GPU > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_model(arch: str, num_classes: int) -> nn.Module:
    """
    Recreate the network skeleton used in training (no pretrained download needed:
    the trained weights are loaded from the checkpoint right after).
    The head must match train.py exactly: Dropout -> Linear.
    """
    if arch.startswith("resnet"):
        ctor = {"resnet18": models.resnet18, "resnet34": models.resnet34,
                "resnet50": models.resnet50}[arch]
        model = ctor(weights=None)
        model.fc = nn.Sequential(nn.Dropout(0.0), nn.Linear(model.fc.in_features, num_classes))
    elif arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(0.0), nn.Linear(in_features, num_classes))
    else:
        raise ValueError(f"Unsupported architecture in checkpoint: '{arch}'")
    return model


@functools.lru_cache(maxsize=4)
def load_model(checkpoint_path: str = DEFAULT_CHECKPOINT):
    """
    Load the trained weights and metadata.
    Returns (model, class_names, preprocess_transform, device). Cached per path.
    """
    path = Path(checkpoint_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path.resolve()} (run train.py first)")

    device = _get_device()
    # weights_only=True: safe loading - refuses to execute arbitrary pickled code.
    checkpoint = torch.load(path, map_location=device, weights_only=True)

    model = _build_model(checkpoint["arch"], checkpoint["num_classes"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()  # eval(): disables dropout, uses fixed batch-norm statistics

    # Must match train.py's validation transform exactly.
    size = checkpoint["img_size"]
    preprocess = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(checkpoint["mean"], checkpoint["std"]),
    ])
    return model, tuple(checkpoint["class_names"]), preprocess, device


def predict_top_k(image_path: str, checkpoint_path: str = DEFAULT_CHECKPOINT, k: int = 3):
    """
    Return the k most likely classes as a list of (class_name, confidence_percent),
    sorted from most to least likely.
    """
    model, class_names, preprocess, device = load_model(str(checkpoint_path))

    path = Path(image_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path.resolve()}")
    try:
        # convert("RGB") handles greyscale, RGBA and palette images uniformly.
        image = Image.open(path).convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError(f"Not a readable image file: {path}") from exc

    batch = preprocess(image).unsqueeze(0).to(device)  # shape: [1, 3, H, W]

    with torch.inference_mode():
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)[0]  # raw scores -> probabilities summing to 1

    k = max(1, min(k, len(class_names)))
    top_probs, top_idx = probs.topk(k)
    return [(class_names[i], p * 100.0) for p, i in zip(top_probs.tolist(), top_idx.tolist())]


def predict(image_path: str, checkpoint_path: str = DEFAULT_CHECKPOINT):
    """
    Classify one image.
    Returns (predicted_class_name, confidence_percent), e.g. ("Glaucoma", 93.41).
    """
    return predict_top_k(image_path, checkpoint_path, k=1)[0]


def main():
    parser = argparse.ArgumentParser(description="Classify a retinal image.")
    parser.add_argument("image", help="Path to the image file.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Path to best_model.pth.")
    parser.add_argument("--top-k", type=int, default=1, help="Also show the next most likely classes.")
    args = parser.parse_args()

    try:
        results = predict_top_k(args.image, args.checkpoint, k=args.top_k)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    label, confidence = results[0]
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.2f}%")
    if len(results) > 1:
        print("\nTop predictions:")
        for rank, (name, conf) in enumerate(results, 1):
            print(f"  {rank}. {name:<25} {conf:6.2f}%")


if __name__ == "__main__":
    main()
