"""Fine-tune SegFormer-b0 on BFDD for the rgb | ir | rgbir comparison.

Usage (from repo root, MPS):
  .venv/bin/python -m defectlens.thermal.train_seg --variant ir \
      --epochs 25 --batch-size 4 --output-dir models/thermal_bfdd/ir

Writes <output-dir>/metrics.json: per-class IoU on the frozen test split,
plus config. Weights stay out of git (models/ is gitignored).
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from defectlens.thermal.bfdd import (
    CLASS_IDS,
    CLASS_NAMES,
    BfddPair,
    frozen_split_pairs,
    load_mask,
)

VARIANT_CHANNELS = {"rgb": 3, "ir": 3, "rgbir": 6, "rgbir_hybrid": 6}
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _norm(img_u8: np.ndarray) -> np.ndarray:
    x = img_u8.astype(np.float32) / 255.0
    return (x - IMAGENET_MEAN) / IMAGENET_STD


def compose_input(rgb: np.ndarray, ir: np.ndarray, variant: str) -> torch.Tensor:
    """(H,W,3) uint8 arrays -> normalized CHW float tensor per variant."""
    if variant == "rgb":
        x = _norm(rgb)
    elif variant == "ir":
        x = _norm(ir)
    elif variant.startswith("rgbir"):  # rgbir and rgbir_hybrid: same 6-ch input
        x = np.concatenate([_norm(rgb), _norm(ir)], axis=-1)
    else:
        raise ValueError(f"unknown variant {variant!r}")
    return torch.from_numpy(x).permute(2, 0, 1).contiguous()


def _force_batchnorm_contiguous_input(model) -> None:
    """Work around an MPS BatchNorm2d backward crash.

    SegFormer's decode head feeds its BatchNorm2d a non-contiguous input; on
    Apple MPS the batchnorm backward then raises "view size is not compatible
    with input tensor's size and stride" at the real 512x640 feature size
    (contiguous input is fine, and CPU is unaffected). Force the input
    contiguous via a forward pre-hook — numerically a no-op, only a memory
    layout change — so training runs natively on MPS. Locked by
    test_build_model_backward_runs_on_mps.
    """
    import torch.nn as nn

    def _to_contiguous(_module, args):
        return (args[0].contiguous(),) + args[1:]

    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.register_forward_pre_hook(_to_contiguous)


def _init_hybrid_stem(model, num_labels: int) -> None:
    """Seed the fused 6-channel stem so the RGB half is pretrained and the IR
    half is zero, i.e. fusion starts as *exactly* the pretrained RGB model plus a
    learnable IR delta. This removes the fusion-init confound documented in the
    README Phase 5.6 analysis: the plain `rgbir` stem is fully re-initialized by
    `ignore_mismatched_sizes`, handicapping fusion vs the pretrained rgb/ir stems.
    The zero IR half still receives gradient, so it is a learnable delta, not dead.
    """
    from transformers import SegformerForSemanticSegmentation

    ref = SegformerForSemanticSegmentation.from_pretrained(  # cached; cheap
        "nvidia/mit-b0", num_labels=num_labels, ignore_mismatched_sizes=True
    )
    stem = model.segformer.stages[0].patch_embeddings.proj  # verified path (transformers 5.x)
    ref_stem = ref.segformer.stages[0].patch_embeddings.proj
    if stem.weight.shape[1] != 6 or ref_stem.weight.shape[1] != 3:
        raise ValueError(
            f"unexpected stem shapes: fused {tuple(stem.weight.shape)}, "
            f"ref {tuple(ref_stem.weight.shape)} (expected 6-ch and 3-ch)"
        )
    with torch.no_grad():
        stem.weight[:, :3].copy_(ref_stem.weight)  # RGB half: pretrained
        stem.weight[:, 3:].zero_()  # IR half: zero (learnable delta)
        stem.bias.copy_(ref_stem.bias)


def build_model(variant: str, num_labels: int = len(CLASS_IDS)):
    from transformers import SegformerForSemanticSegmentation

    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b0",
        num_labels=num_labels,
        num_channels=VARIANT_CHANNELS[variant],
        ignore_mismatched_sizes=True,  # 6-ch stem + fresh head re-init
    )
    if variant == "rgbir_hybrid":
        _init_hybrid_stem(model, num_labels)
    _force_batchnorm_contiguous_input(model)
    return model


class BfddSegDataset(Dataset):
    def __init__(self, pairs: list[BfddPair], variant: str):
        self.pairs = pairs
        self.variant = variant

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        p = self.pairs[i]
        rgb = np.array(Image.open(p.rgb).convert("RGB"))
        ir = np.array(Image.open(p.ir).convert("RGB"))
        x = compose_input(rgb, ir, self.variant)
        y = torch.from_numpy(load_mask(p.label))
        return x, y


def iou_from_confusion(conf: np.ndarray) -> np.ndarray:
    inter = np.diag(conf).astype(np.float64)
    union = conf.sum(0) + conf.sum(1) - np.diag(conf)
    return np.where(union > 0, inter / np.maximum(union, 1), np.nan)


def _finite_or_none(x: float) -> float | None:
    """NaN/inf -> None so json.dumps emits valid `null`, not bare `NaN`."""
    return float(x) if np.isfinite(x) else None


def per_class_iou_json(ious: np.ndarray) -> dict[str, float | None]:
    """CLASS_NAMES-keyed IoU; a class absent from the test split has IoU NaN
    (undefined union) which serializes as null, not an invalid JSON `NaN`."""
    return {CLASS_NAMES[c]: _finite_or_none(ious[c]) for c in CLASS_IDS}


def build_metrics(
    variant: str,
    ious: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    steps: int,
    train_pairs: int,
    test_pairs: int,
    final_train_loss: float | None = None,
) -> dict:
    """Assemble the metrics.json payload, including the run config needed to
    reproduce it (epochs/batch_size/lr/seed), the last training-batch loss, and
    JSON-safe per-class IoU."""
    defect_ious = np.array([ious[c] for c in CLASS_IDS if c != 0], dtype=float)
    mean_defect = np.nanmean(defect_ious) if np.any(np.isfinite(defect_ious)) else np.nan
    return {
        "variant": variant,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "seed": seed,
        "steps": steps,
        "train_pairs": train_pairs,
        "test_pairs": test_pairs,
        "final_train_loss": None if final_train_loss is None else _finite_or_none(final_train_loss),
        "per_class_iou": per_class_iou_json(ious),
        "mean_defect_iou": _finite_or_none(mean_defect),
    }


@torch.no_grad()
def evaluate(model, loader, device, num_labels: int) -> np.ndarray:
    model.eval()
    conf = np.zeros((num_labels, num_labels), dtype=np.int64)
    for x, y in loader:
        logits = model(pixel_values=x.to(device)).logits
        logits = F.interpolate(logits, size=y.shape[-2:], mode="bilinear", align_corners=False)
        pred = logits.argmax(1).cpu().numpy().ravel()
        gt = y.numpy().ravel()
        conf += np.bincount(gt * num_labels + pred, minlength=num_labels**2).reshape(
            num_labels, num_labels
        )
    return conf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=sorted(VARIANT_CHANNELS), required=True)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=6e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--subset", type=int, default=0, help="train on N pairs (smoke)")
    ap.add_argument("--max-steps", type=int, default=0, help="stop early (smoke)")
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    buckets = frozen_split_pairs()  # authoritative committed manifest, NOT args.seed
    # buckets["val"] (126) is intentionally reserved and unused here: the
    # comparison uses fixed hyperparameters with no model selection, so the
    # frozen test split is evaluated and reported directly.
    train_pairs = buckets["train"][: args.subset or None]
    num_labels = len(CLASS_IDS)

    model = build_model(args.variant, num_labels).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    train_loader = DataLoader(
        BfddSegDataset(train_pairs, args.variant),
        batch_size=args.batch_size, shuffle=True, num_workers=0,
    )
    test_loader = DataLoader(
        BfddSegDataset(buckets["test"], args.variant), batch_size=args.batch_size
    )

    step = 0
    final_train_loss = None
    for epoch in range(args.epochs):
        model.train()
        for x, y in train_loader:
            logits = model(pixel_values=x.to(device)).logits
            logits = F.interpolate(logits, size=y.shape[-2:], mode="bilinear", align_corners=False)
            loss = F.cross_entropy(logits, y.to(device))
            opt.zero_grad(); loss.backward(); opt.step()
            final_train_loss = loss.item()
            step += 1
            if step % 50 == 0:
                print(f"epoch {epoch} step {step} loss {final_train_loss:.4f}", flush=True)
            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break

    conf = evaluate(model, test_loader, device, num_labels)
    ious = iou_from_confusion(conf)
    metrics = build_metrics(
        args.variant,
        ious,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        steps=step,
        train_pairs=len(train_pairs),
        test_pairs=len(buckets["test"]),
        final_train_loss=final_train_loss,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    model.save_pretrained(args.output_dir / "checkpoint")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
