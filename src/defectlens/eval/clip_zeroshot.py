"""CLIP zero-shot baseline on the frozen test split (spec §5).

Produces results/clip_zeroshot_baseline.json + confusion_matrix.png.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm

from defectlens.ingest import read_manifest
from defectlens.metrics import confusion_matrix, macro_topk_accuracy, per_class_topk_accuracy
from defectlens.taxonomy import UNIFIED_CLASSES


def expand_prompts(
    class_phrases: dict[str, str], templates: list[str]
) -> dict[str, list[str]]:
    return {
        cls: [t.format(phrase) for t in templates]
        for cls, phrase in class_phrases.items()
    }


def rank_from_similarity(sims: np.ndarray, classes: list[str]) -> list[list[str]]:
    """sims: [n_images, n_classes] -> per-image class ranking, best first."""
    order = np.argsort(-sims, axis=1)
    return [[classes[j] for j in row] for row in order]


def _nan_to_none(value: float) -> float | None:
    """NaN (absent class) -> null in JSON; json.dumps NaN is invalid per RFC 8259."""
    return None if math.isnan(value) else value


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _features(out) -> torch.Tensor:
    """Unwrap get_text_features/get_image_features across transformers versions.

    v4 returns the projected features tensor directly; v5 returns a
    BaseModelOutputWithPooling whose pooler_output has been REPLACED with the
    projected features (verified against transformers 5.13.0 source).
    """
    return out if isinstance(out, torch.Tensor) else out.pooler_output


def build_text_features(model, processor, prompts: dict[str, list[str]], device: str):
    feats = []
    for cls in UNIFIED_CLASSES:
        inputs = processor(text=prompts[cls], padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            emb = _features(model.get_text_features(**inputs))
        emb = emb / emb.norm(dim=-1, keepdim=True)
        feats.append(emb.mean(dim=0))
    feats = torch.stack(feats)
    return feats / feats.norm(dim=-1, keepdim=True)


def main() -> None:
    from transformers import CLIPModel, CLIPProcessor

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/test.csv"))
    parser.add_argument("--config", type=Path, default=Path("configs/clip_prompts.yaml"))
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    prompts = expand_prompts(cfg["class_phrases"], cfg["templates"])
    assert set(prompts) == set(UNIFIED_CLASSES), "prompt config must cover all classes"

    device = pick_device()
    print(f"Device: {device}; model: {cfg['model']}")
    model = CLIPModel.from_pretrained(cfg["model"]).to(device).eval()
    processor = CLIPProcessor.from_pretrained(cfg["model"])

    text_feats = build_text_features(model, processor, prompts, device)

    rows = read_manifest(args.manifest)
    y_true = [r.unified_label for r in rows]
    all_sims: list[np.ndarray] = []
    for i in tqdm(range(0, len(rows), args.batch_size), desc="images"):
        batch = rows[i : i + args.batch_size]
        images = [Image.open(r.image_path).convert("RGB") for r in batch]
        inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            emb = _features(model.get_image_features(**inputs))
        emb = emb / emb.norm(dim=-1, keepdim=True)
        all_sims.append((emb @ text_feats.T).cpu().numpy())
    sims = np.concatenate(all_sims)
    ranked = rank_from_similarity(sims, UNIFIED_CLASSES)
    top1 = [r[0] for r in ranked]

    per1 = per_class_topk_accuracy(y_true, ranked, UNIFIED_CLASSES, k=1)
    per3 = per_class_topk_accuracy(y_true, ranked, UNIFIED_CLASSES, k=3)
    results = {
        "model": cfg["model"],
        "manifest": str(args.manifest),
        "n_images": len(rows),
        "macro_top1": _nan_to_none(macro_topk_accuracy(y_true, ranked, UNIFIED_CLASSES, k=1)),
        "macro_top3": _nan_to_none(macro_topk_accuracy(y_true, ranked, UNIFIED_CLASSES, k=3)),
        "per_class_top1": {c: _nan_to_none(v) for c, v in per1.items()},
        "per_class_top3": {c: _nan_to_none(v) for c, v in per3.items()},
        "confusion_matrix": confusion_matrix(y_true, top1, UNIFIED_CLASSES),
        "classes": UNIFIED_CLASSES,
    }
    args.out_dir.mkdir(exist_ok=True)
    out_json = args.out_dir / "clip_zeroshot_baseline.json"
    out_json.write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
    print(f"macro top-1: {results['macro_top1']:.3f}  macro top-3: {results['macro_top3']:.3f}")
    print(f"Wrote {out_json}")

    # Confusion matrix figure
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = np.array(results["confusion_matrix"], dtype=float)
    m_norm = m / np.maximum(m.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(m_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(UNIFIED_CLASSES)), UNIFIED_CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(UNIFIED_CLASSES)), UNIFIED_CLASSES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("CLIP zero-shot — row-normalized confusion")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(args.out_dir / "clip_zeroshot_confusion.png", dpi=150)
    print(f"Wrote {args.out_dir / 'clip_zeroshot_confusion.png'}")


if __name__ == "__main__":
    main()
