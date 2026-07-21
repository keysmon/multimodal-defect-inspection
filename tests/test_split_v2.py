"""Taxonomy-v2 frozen-split invariants (plan Task A5, user-gated 2026-07-21).

The v1 frozen test set (data/manifests/test_v1_frozen.csv, 2,648 rows) is
IMMUTABLE: the per-(dataset,label) split RNG makes v1 group membership
invariant when new datasets are added, so every v1 test row must reappear in
the v2 test split. If test_v1_test_rows_survive_in_v2 fails, the split
contract itself was violated — that is never fixable by editing this test.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from defectlens.ingest import read_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = REPO_ROOT / "data" / "manifests"


def test_v1_test_rows_survive_in_v2():
    v1 = {r.image_path for r in read_manifest(MANIFESTS / "test_v1_frozen.csv")}
    v2 = {r.image_path for r in read_manifest(MANIFESTS / "test.csv")}
    assert len(v1) == 2648
    missing = v1 - v2
    assert not missing, (
        f"v2 split lost {len(missing)} frozen v1 test rows - split contract violated"
    )


def test_v2_split_covers_every_splittable_class():
    # split.py gives >=1 test row to every (dataset,label) group of size >= 4,
    # so any class with at least one group of >= 4 rows must appear in test.
    manifest = read_manifest(MANIFESTS / "manifest.csv")
    groups = Counter((r.source_dataset, r.unified_label) for r in manifest)
    expected = {label for (_, label), n in groups.items() if n >= 4}
    test_labels = {r.unified_label for r in read_manifest(MANIFESTS / "test.csv")}
    assert expected <= test_labels, (
        f"missing from test split: {expected - test_labels}"
    )
