"""BFDD dataset module: pair listing, frozen split, mask loading.

Real-data tests run only when ~/datasets/bfdd exists (CI skips them);
split determinism is locked with a synthetic stem list so it always runs.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from defectlens.thermal.bfdd import (
    BFDD_ROOT,
    CLASS_IDS,
    BfddPair,
    list_pairs,
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


def test_split_stems_regression_lock_first_members():
    """Frozen-split discipline (Phase 1 convention): the seed-42 split of the
    real 838 stems must never silently change. Locks the first member of each
    bucket computed at plan time."""
    stems = [f"s{i}" for i in range(838)]
    s = split_stems(stems, seed=42)
    # Bucket sizes: round(838*0.15)=126 test, 126 val, 586 train.
    assert len(s["train"]) == 586 and len(s["val"]) == 126 and len(s["test"]) == 126
    # First member of each bucket, computed once from the seed-42 shuffle and
    # hard-coded here so the frozen split can never silently drift. Ordering is
    # lexicographic on the sorted stems (s0, s1, s10, s100, ...), so these look
    # non-numeric and that is correct.
    assert s["train"][0] == "s681"
    assert s["val"][0] == "s123"
    assert s["test"][0] == "s81"


def test_class_names_cover_all_ids_and_are_not_placeholders():
    from defectlens.thermal.bfdd import CLASS_IDS, CLASS_NAMES

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
def test_load_mask_values_within_class_ids():
    import numpy as np

    from defectlens.thermal.bfdd import load_mask

    pairs = list_pairs()
    m = load_mask(pairs[0].label)
    assert m.dtype == np.int64 and m.shape == (512, 640)
    assert set(np.unique(m)) <= set(CLASS_IDS)
