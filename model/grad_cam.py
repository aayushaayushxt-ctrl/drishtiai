"""Explain Member 4's retinal classifier with Grad-CAM."""

import argparse
import json
import random
from pathlib import Path
from threading import get_ident

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision.datasets.folder import IMG_EXTENSIONS

from predict import load_model


CHECKPOINT = Path(__file__).resolve().parent / "models" / "best_model.pth"

# Low to high positive contribution, with low values transparent in the overlay.
PALETTE = np.array([[15, 22, 52], [51, 47, 127], [149, 51, 122],
                    [235, 111, 67], [255, 235, 132]])


@torch.inference_mode(False)
@torch.enable_grad()
def explain(image: Image.Image, checkpoint_path: str | Path = CHECKPOINT) -> tuple[dict, np.ndarray]:
    """Return prediction metadata and a float heatmap in the input photo's coordinates."""
    model, class_names, preprocess, device = load_model(str(Path(checkpoint_path).expanduser().resolve()))
    if hasattr(model, "layer4"):
        layer_name, layer = "layer4[-1]", model.layer4[-1]
    else:
        layer_name, layer = "features[-1]", model.features[-1]

    image = image.convert("RGB")
    batch = preprocess(image).unsqueeze(0).to(device).requires_grad_(True)
    activations = []
    request_thread = get_ident()

    def capture_features(_layer, _inputs, output):
        if get_ident() == request_thread:
            activations.append(output)

    hook = layer.register_forward_hook(capture_features)
    try:
        logits = model(batch)
    finally:
        hook.remove()

    scores = logits.detach().softmax(dim=1)[0]
    index = int(scores.argmax())
    features = activations[0]
    gradients, = torch.autograd.grad(logits[0, index], features)
    channel_weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = (features.detach() * channel_weights).sum(dim=1, keepdim=True).relu()
    if not bool(torch.isfinite(cam).all()):
        raise ValueError("Grad-CAM produced non-finite values.")
    peak = float(cam.max())
    if peak > 0:
        cam = cam / peak
    resized = F.interpolate(cam, size=(image.height, image.width),
                            mode="bilinear", align_corners=False)
    prediction = {
        "predicted": class_names[index],
        "score_percent": float(scores[index]) * 100,
        "target_layer": layer_name,
        "feature_grid": list(cam.shape[-2:]),
        "has_positive_activation": peak > 0,
    }
    return prediction, resized[0, 0].cpu().numpy()


def render_maps(image: Image.Image, heatmap: np.ndarray) -> tuple[Image.Image, Image.Image]:
    """Return a colour heatmap and an overlay, both aligned to the original photo."""
    levels = np.linspace(0, 1, len(PALETTE))
    colours = np.stack([np.interp(heatmap, levels, channel) for channel in PALETTE.T], axis=-1)
    opacity = 0.65 * heatmap[..., None]
    photo = np.asarray(image.convert("RGB"))
    overlay = (1 - opacity) * photo + opacity * colours
    return Image.fromarray(colours.astype(np.uint8)), Image.fromarray(overlay.astype(np.uint8))


def save_figure(path, image, heatmap, result):
    coloured, overlay = render_maps(image, heatmap)
    panel_width, panel_height = 300, 245
    figure = Image.new("RGB", (948, 362), "#f5f6f9")
    draw = ImageDraw.Draw(figure)
    ink = "#182136"
    title = (f"{result['outcome'].title()} prediction   |   "
             f"Reference: {result['reference'] or 'unlabeled'}   |   "
             f"Predicted: {result['predicted']} ({result['score_percent']:.1f}% model score)")
    draw.text((16, 13), title, fill=ink, font_size=19)
    draw.text((16, 40), result["image"], fill=ink, font_size=14)
    for i, (panel, label) in enumerate(zip((image, coloured, overlay),
                                           ("Original photo", "Grad-CAM heatmap", "Overlay"))):
        x = 16 + i * (panel_width + 8)
        panel = panel.copy()
        panel.thumbnail((panel_width, panel_height), Image.Resampling.LANCZOS)
        figure.paste(panel, (x + (panel_width - panel.width) // 2,
                             81 + (panel_height - panel.height) // 2))
        draw.text((x, 60), label, fill=ink, font_size=15)

    legend = Image.new("RGB", (105, 10))
    legend_heat = np.linspace(0, 1, 105)[None, :].repeat(10, axis=0)
    legend, _ = render_maps(legend, legend_heat)
    figure.paste(legend, (16, 338))
    note = "Low to high relative support for the predicted class"
    if not result["has_positive_activation"]:
        note = "No positive contribution at the chosen layer for this prediction"
    draw.text((131, 333), note, fill=ink, font_size=14)
    figure.save(path)


def score_photo(path, checkpoint, reference, display_name):
    with Image.open(path) as source:
        image = source.convert("RGB")
    prediction, heatmap = explain(image, checkpoint)
    outcome = "unlabeled"
    if reference is not None:
        outcome = "correct" if prediction["predicted"] == reference else "incorrect"
    result = {"image": display_name, "reference": reference, "outcome": outcome, **prediction}
    return image, heatmap, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--image", type=Path, help="One original retinal photo.")
    inputs.add_argument("--dataset", type=Path, help="Labeled image folders (one per model class).")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT,
                        help="Checkpoint to explain (defaults to the bundled best_model.pth).")
    parser.add_argument("--label", help="Reference class for --image, if known.")
    parser.add_argument("--per-outcome", type=int, default=1,
                        help="Number of examples to save for each outcome in dataset mode.")
    parser.add_argument("--seed", type=int, default=42, help="Dataset shuffle seed.")
    parser.add_argument("--output", type=Path, default=Path("heatmaps"))
    args = parser.parse_args()
    if args.per_outcome < 1:
        parser.error("--per-outcome must be positive.")
    if args.dataset and args.label:
        parser.error("--label applies to --image only.")

    checkpoint = args.checkpoint.expanduser().resolve()
    _, classes, _, device = load_model(str(checkpoint))
    if device.type == "cpu":
        torch.set_num_threads(min(torch.get_num_threads(), 4))
    if args.image:
        if args.label is not None and args.label not in classes:
            parser.error(f"Use one of the checkpoint class names: {', '.join(classes)}")
        entries = [(args.image, args.label, args.image.name)]
    else:
        if not args.dataset.is_dir():
            parser.error(f"Dataset directory does not exist: {args.dataset}")
        entries = [
            (path, label, path.relative_to(args.dataset).as_posix())
            for label in classes for path in sorted((args.dataset / label).rglob("*"))
            if path.is_file() and path.suffix.lower() in IMG_EXTENSIONS
        ]
        if not entries:
            parser.error(f"No images found under the checkpoint class folders: {', '.join(classes)}")
        random.Random(args.seed).shuffle(entries)

    args.output.mkdir(parents=True, exist_ok=True)
    counts = {"correct": 0, "incorrect": 0, "unlabeled": 0}
    samples, scanned = [], 0
    for path, reference, display_name in entries:
        image, heatmap, result = score_photo(path, checkpoint, reference, display_name)
        scanned += 1
        outcome = result["outcome"]
        if counts[outcome] >= args.per_outcome:
            continue
        counts[outcome] += 1
        result["figure"] = f"{outcome}-{counts[outcome]:02d}.png"
        save_figure(args.output / result["figure"], image, heatmap, result)
        samples.append(result)
        if args.dataset and min(counts["correct"], counts["incorrect"]) >= args.per_outcome:
            break

    summary = {"checkpoint": str(checkpoint), "classes": classes,
               "images_scanned": scanned, "seed": args.seed, "samples": samples}
    (args.output / "results.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(samples)} figure(s) from {scanned} image(s) to {args.output}.")
    if args.dataset and min(counts["correct"], counts["incorrect"]) < args.per_outcome:
        print(f"Available outcomes saved: {counts['correct']} correct, {counts['incorrect']} incorrect.")


if __name__ == "__main__":
    main()
