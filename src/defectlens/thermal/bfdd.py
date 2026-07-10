"""BFDD (Building Facade Defect Dataset) access: pairs, frozen split, masks.

BFDD: 838 pixel-aligned RGB+IR facade image pairs with 6-class (background +
5 defect) segmentation masks, 640x512, CC BY 4.0.
Source: https://data.mendeley.com/datasets/9ych7czvyg/1 (fetch via
scripts/fetch_bfdd.sh). CLASS_NAMES provenance is documented in
docs/datasets.md (verified against Label_color + the dataset description).
"""
from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

BFDD_ROOT = Path.home() / "datasets" / "bfdd" / "Dataset_1x"

# Committed frozen-split artifact (stem,split). This CSV — not split_pairs — is
# the authoritative train/val/test partition; see frozen_split_pairs.
SPLIT_MANIFEST = Path("data/manifests/bfdd_split.csv")

CLASS_IDS = (0, 1, 2, 3, 4, 5)
# Verified 2026-07-09 against the Label_color legend + Mendeley description +
# per-id RGB/IR inspection; evidence recorded in docs/datasets.md. crack/
# hollow_area/stain are corroborated by the description and IR behavior;
# peeling vs erosion rests on visual inference (blistering vs material loss).
CLASS_NAMES = {
    0: "background",
    1: "crack",
    2: "hollow_area",
    3: "peeling",
    4: "erosion",
    5: "stain",
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
    """Frozen 70/15/15 split. Sorts first so input order can't leak in.

    Records HOW the committed manifest was generated; it is not the runtime
    source of truth. Training/eval must call frozen_split_pairs so a
    missing/added file can never silently re-partition the split.
    """
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
    """Compute the split from scratch (used ONCE to generate SPLIT_MANIFEST).

    The committed manifest is authoritative at runtime; prefer
    frozen_split_pairs. Regenerating the manifest invalidates any previously
    reported numbers (same freeze discipline as defectlens.split).
    """
    buckets = split_stems([p.stem for p in pairs], seed=seed)
    member = {s: k for k, ss in buckets.items() for s in ss}
    out: dict[str, list[BfddPair]] = {"train": [], "val": [], "test": []}
    for p in pairs:
        out[member[p.stem]].append(p)
    return out


def frozen_split_pairs(
    root: Path = BFDD_ROOT, manifest: Path = SPLIT_MANIFEST
) -> dict[str, list[BfddPair]]:
    """Load the authoritative committed split, verified against on-disk pairs.

    Reads the committed manifest (SPLIT_MANIFEST, columns stem,split) and
    returns train/val/test buckets of complete BfddPairs. Fails loudly if the
    manifest and the on-disk complete triples disagree in either direction — a
    manifest stem with no file on disk, or an on-disk stem absent from the
    manifest — so one missing/extra file can never silently re-partition the
    split. This is the freeze contract; regenerating the manifest invalidates
    previously reported numbers (mirrors defectlens.split's --force guard).

    Buckets preserve list_pairs' stem-sorted order (deterministic).
    """
    with open(manifest, newline="") as fh:
        rows = {row["stem"]: row["split"] for row in csv.DictReader(fh)}

    pairs = list_pairs(root)
    disk_stems = {p.stem for p in pairs}
    manifest_stems = set(rows)

    missing_on_disk = manifest_stems - disk_stems
    missing_in_manifest = disk_stems - manifest_stems
    if missing_on_disk or missing_in_manifest:
        problems = []
        if missing_on_disk:
            problems.append(
                f"{len(missing_on_disk)} manifest stem(s) with no file on disk: "
                f"{sorted(missing_on_disk)}"
            )
        if missing_in_manifest:
            problems.append(
                f"{len(missing_in_manifest)} on-disk stem(s) absent from manifest: "
                f"{sorted(missing_in_manifest)}"
            )
        raise ValueError(
            f"BFDD split manifest {manifest} disagrees with pairs under {root}: "
            + "; ".join(problems)
        )

    out: dict[str, list[BfddPair]] = {"train": [], "val": [], "test": []}
    for p in pairs:  # pairs are stem-sorted, so each bucket stays stem-sorted
        split = rows[p.stem]
        if split not in out:
            raise ValueError(
                f"{manifest}: unknown split {split!r} for stem {p.stem!r} "
                f"(expected one of {sorted(out)})"
            )
        out[split].append(p)
    return out


def load_mask(path: Path) -> np.ndarray:
    """L-mode PNG -> int64 (H, W) class-id array; rejects non-2-D or out-of-range ids.

    Guards against the superseded Label_backup_7classes dir (ids up to 6) or an
    RGB mask leaking into the 6-class head — either raises ValueError naming path.
    """
    arr = np.array(Image.open(path), dtype=np.int64)
    if arr.ndim != 2:
        raise ValueError(
            f"{path}: expected a 2-D L-mode mask, got array with shape {arr.shape}"
        )
    extra = set(np.unique(arr).tolist()) - set(CLASS_IDS)
    if extra:
        raise ValueError(
            f"{path}: mask has out-of-range class ids {sorted(extra)} "
            f"(expected subset of {list(CLASS_IDS)})"
        )
    return arr
