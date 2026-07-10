"""Comparison figure: RGB | IR | ground truth | RGB-model pred | IR-model pred
for up to 3 test images containing the class where IR and RGB diverge most
(per results/thermal_bfdd.json). Requires the trained checkpoints from Task 4.

Usage: .venv/bin/python scripts/make_thermal_figure.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from defectlens.thermal.bfdd import CLASS_IDS, CLASS_NAMES, frozen_split_pairs, load_mask
from defectlens.thermal.train_seg import compose_input

PALETTE = np.array(
    [[0, 0, 0], [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48]],
    dtype=np.uint8,
)


def colorize(mask: np.ndarray) -> np.ndarray:
    return PALETTE[mask]


def predict(model_dir: Path, x: torch.Tensor, size) -> np.ndarray:
    from transformers import SegformerForSemanticSegmentation

    model = SegformerForSemanticSegmentation.from_pretrained(model_dir).eval()
    with torch.no_grad():
        logits = model(pixel_values=x.unsqueeze(0)).logits
        logits = F.interpolate(logits, size=size, mode="bilinear", align_corners=False)
    return logits.argmax(1)[0].numpy()


def main() -> None:
    results = json.loads(Path("results/thermal_bfdd.json").read_text())
    # pick the defect class with the largest |IR - RGB| IoU gap as the story class
    gaps = {
        name: abs(results["ir"]["per_class_iou"][name] - results["rgb"]["per_class_iou"][name])
        for name in results["ir"]["per_class_iou"]
        if name != "background"
    }
    story = max(gaps, key=gaps.get)
    story_id = next(c for c in CLASS_IDS if CLASS_NAMES[c] == story)
    print(f"story class: {story} (IoU gap {gaps[story]:.3f})")

    test_pairs = frozen_split_pairs()["test"]
    picks = [p for p in test_pairs if story_id in np.unique(load_mask(p.label))][:3]
    if not picks:
        raise SystemExit(f"no test images contain class {story!r} ({story_id})")

    fig, axes = plt.subplots(len(picks), 5, figsize=(16, 3.2 * len(picks)))
    axes = np.atleast_2d(axes)
    for r, p in enumerate(picks):
        rgb = np.array(Image.open(p.rgb).convert("RGB"))
        ir = np.array(Image.open(p.ir).convert("RGB"))
        gt = load_mask(p.label)
        pred_rgb = predict(Path("models/thermal_bfdd/rgb/checkpoint"), compose_input(rgb, ir, "rgb"), gt.shape)
        pred_ir = predict(Path("models/thermal_bfdd/ir/checkpoint"), compose_input(rgb, ir, "ir"), gt.shape)
        for c, (img, title) in enumerate(
            [
                (rgb, "RGB"),
                (ir, "IR (thermal)"),
                (colorize(gt), "ground truth"),
                (colorize(pred_rgb), "RGB-only pred"),
                (colorize(pred_ir), "IR-only pred"),
            ]
        ):
            axes[r, c].imshow(img)
            axes[r, c].set_title(title if r == 0 else "", fontsize=11)
            axes[r, c].axis("off")

    # Colour legend so mask blobs are readable (story class stands out).
    handles = [
        mpatches.Patch(color=PALETTE[c] / 255.0, label=CLASS_NAMES[c] + (" (story)" if c == story_id else ""))
        for c in CLASS_IDS
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=len(CLASS_IDS),
        fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.0),
    )
    fig.suptitle(
        f"BFDD {story}: RGB-only vs IR-only predictions (largest modality gap, "
        f"IoU {results['rgb']['per_class_iou'][story]:.2f} vs {results['ir']['per_class_iou'][story]:.2f})",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig("docs/images/thermal-comparison.png", dpi=120, bbox_inches="tight")
    print("wrote docs/images/thermal-comparison.png")


if __name__ == "__main__":
    main()
