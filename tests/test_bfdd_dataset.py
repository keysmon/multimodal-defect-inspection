"""BFDD dataset module: pair listing, frozen split, mask loading.

Two layers of split protection are tested here:
- Synthetic, always-runs: locks the Python-RNG-stream + algorithm determinism
  of split_stems (how the committed manifest was generated).
- Real-data, runs only when ~/datasets/bfdd exists: locks the authoritative
  committed manifest (data/manifests/bfdd_split.csv) via frozen_split_pairs,
  including its loud-fail-on-mismatch guard. CI without the data skips these.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from defectlens.thermal.bfdd import (
    BFDD_ROOT,
    CLASS_IDS,
    CLASS_NAMES,
    SPLIT_MANIFEST,
    BfddPair,
    frozen_split_pairs,
    list_pairs,
    load_mask,
    split_stems,
)

HAVE_DATA = BFDD_ROOT.exists()


def test_split_stems_is_deterministic_and_disjoint():
    stems = [f"img_{i:04d}" for i in range(100)]
    a = split_stems(stems, seed=42)
    b = split_stems(list(reversed(stems)), seed=42)  # order-insensitive
    assert a == b
    assert len(a["test"]) == 15 and len(a["val"]) == 15 and len(a["train"]) == 70
    assert not (set(a["train"]) & set(a["val"])) and not (set(a["val"]) & set(a["test"]))
    assert not (set(a["train"]) & set(a["test"]))


def test_split_stems_determinism_lock_synthetic():
    """Locks split_stems' RNG-stream + algorithm determinism on a SYNTHETIC
    838-stem list (not the real dataset). This guards that seeding/shuffle
    logic never drifts; the REAL split is locked separately against the
    committed manifest in test_frozen_split_pairs_matches_committed_manifest."""
    stems = [f"s{i}" for i in range(838)]
    s = split_stems(stems, seed=42)
    # Bucket sizes: round(838*0.15)=126 test, 126 val, 586 train.
    assert len(s["train"]) == 586 and len(s["val"]) == 126 and len(s["test"]) == 126
    # First member of each bucket, computed once from the seed-42 shuffle and
    # hard-coded here. Ordering is lexicographic on the sorted stems
    # (s0, s1, s10, s100, ...), so these look non-numeric and that is correct.
    assert s["train"][0] == "s681"
    assert s["val"][0] == "s123"
    assert s["test"][0] == "s81"


def test_class_names_cover_all_ids_and_are_not_placeholders():
    assert set(CLASS_NAMES) == set(CLASS_IDS)
    assert CLASS_NAMES[0] == "background"
    for cid in CLASS_IDS[1:]:
        assert not CLASS_NAMES[cid].startswith("class"), "provisional name left in"


@pytest.mark.skipif(not HAVE_DATA, reason="BFDD data not present")
def test_list_pairs_finds_838_complete_pairs():
    pairs = list_pairs()
    assert len(pairs) == 838
    p = pairs[0]
    assert isinstance(p, BfddPair)
    assert p.rgb.exists() and p.ir.exists() and p.label.exists()
    assert p.rgb.suffix == ".JPG" and p.ir.suffix == ".png"


@pytest.mark.skipif(not HAVE_DATA, reason="BFDD data not present")
def test_frozen_split_pairs_matches_committed_manifest():
    """Locks the REAL split against the committed manifest: exact 586/126/126,
    all 838 stems accounted for and disjoint, and the first (stem-sorted)
    member of each bucket frozen to its literal value."""
    buckets = frozen_split_pairs()
    assert len(buckets["train"]) == 586
    assert len(buckets["val"]) == 126
    assert len(buckets["test"]) == 126

    stems = {p.stem for b in buckets.values() for p in b}
    assert len(stems) == 838  # complete + disjoint (no stem in two buckets)

    # Buckets preserve list_pairs' stem-sorted order, so [0] is the lexicographic
    # minimum of each split. Frozen literals from data/manifests/bfdd_split.csv.
    assert buckets["train"][0].stem == "DJI_20250624181809_0003"
    assert buckets["val"][0].stem == "DJI_20250624181838_0008"
    assert buckets["test"][0].stem == "DJI_20250624181856_0011"


@pytest.mark.skipif(not HAVE_DATA, reason="BFDD data not present")
def test_frozen_split_pairs_fails_loudly_on_manifest_disk_mismatch(tmp_path):
    """The HIGH-finding protection: any disagreement between the manifest and
    the on-disk complete triples must raise ValueError naming the offenders —
    never silently drop or re-partition."""
    import csv

    real = list(csv.DictReader(SPLIT_MANIFEST.open(newline="")))

    def write(rows, name):
        path = tmp_path / name
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["stem", "split"])
            w.writeheader()
            w.writerows(rows)
        return path

    # (a) manifest references a stem with no file on disk -> loud fail naming it
    ghost = write(real + [{"stem": "GHOST_STEM", "split": "train"}], "ghost.csv")
    with pytest.raises(ValueError, match="GHOST_STEM"):
        frozen_split_pairs(manifest=ghost)

    # (b) an on-disk stem is absent from the manifest -> loud fail naming it
    dropped = real[0]["stem"]
    short = write(real[1:], "short.csv")
    with pytest.raises(ValueError, match=dropped):
        frozen_split_pairs(manifest=short)


@pytest.mark.skipif(not HAVE_DATA, reason="BFDD data not present")
def test_load_mask_values_within_class_ids():
    import numpy as np

    pairs = list_pairs()
    # Sample across the dataset, not just pairs[0], so a stray out-of-range id
    # anywhere is caught rather than assumed absent from one image.
    for p in pairs[:: len(pairs) // 10 or 1]:
        m = load_mask(p.label)
        assert m.dtype == np.int64 and m.shape == (512, 640)
        assert set(np.unique(m).tolist()) <= set(CLASS_IDS)
