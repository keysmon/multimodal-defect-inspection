"""BFDD (Building Facade Defect Dataset) access: pairs, frozen split, masks.

BFDD: 838 pixel-aligned RGB+IR facade image pairs with 6-class (background +
5 defect) segmentation masks, 640x512, CC BY 4.0.
Source: https://data.mendeley.com/datasets/9ych7czvyg/1 (fetch via
scripts/fetch_bfdd.sh). CLASS_NAMES provenance is documented in
docs/datasets.md (verified against Label_color + the dataset description).
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

BFDD_ROOT = Path.home() / "datasets" / "bfdd" / "Dataset_1x"

CLASS_IDS = (0, 1, 2, 3, 4, 5)
# PROVISIONAL until Task 2 verifies the id->name order against Label_color
# and the Mendeley description; Task 2 replaces this dict and this comment.
CLASS_NAMES = {
    0: "background",
    1: "class1",
    2: "class2",
    3: "class3",
    4: "class4",
    5: "class5",
}

VAL_FRAC = 0.15
TEST_FRAC = 0.15


@dataclass(frozen=True)
class BfddPair:
    stem: str
    rgb: Path
    ir: Path
    label: Path


def list_pairs(root: Path = BFDD_ROOT) -> list[BfddPair]:
    """All complete RGB/IR/Label triples, sorted by stem (deterministic)."""
    pairs = []
    for lab in sorted((root / "Label").glob("*.png")):
        stem = lab.stem
        rgb = root / "RGB" / f"{stem}.JPG"
        ir = root / "IR" / f"{stem}.png"
        if rgb.exists() and ir.exists():
            pairs.append(BfddPair(stem=stem, rgb=rgb, ir=ir, label=lab))
    return pairs


def split_stems(stems: list[str], seed: int = 42) -> dict[str, list[str]]:
    """Frozen 70/15/15 split. Sorts first so input order can't leak in."""
    ordered = sorted(stems)
    rng = random.Random(seed)
    rng.shuffle(ordered)
    n = len(ordered)
    n_test = round(n * TEST_FRAC)
    n_val = round(n * VAL_FRAC)
    return {
        "test": ordered[:n_test],
        "val": ordered[n_test : n_test + n_val],
        "train": ordered[n_test + n_val :],
    }


def split_pairs(pairs: list[BfddPair], seed: int = 42) -> dict[str, list[BfddPair]]:
    buckets = split_stems([p.stem for p in pairs], seed=seed)
    member = {s: k for k, ss in buckets.items() for s in ss}
    out: dict[str, list[BfddPair]] = {"train": [], "val": [], "test": []}
    for p in pairs:
        out[member[p.stem]].append(p)
    return out


def load_mask(path: Path) -> np.ndarray:
    """L-mode PNG -> int64 (H, W) class-id array."""
    return np.array(Image.open(path), dtype=np.int64)
