"""Tests for stratified train/test split."""
from pathlib import Path

from defectlens.ingest import ManifestRow, read_manifest
from defectlens.split import stratified_split
from defectlens.taxonomy import UNIFIED_CLASSES

REPO_ROOT = Path(__file__).resolve().parents[1]


def make_rows(dataset: str, label: str, n: int) -> list[ManifestRow]:
    return [
        ManifestRow(f"data/raw/{dataset}/{label}/{i}.jpg", dataset, label, label)
        for i in range(n)
    ]


def test_split_is_disjoint_and_complete():
    rows = make_rows("d1", "crack", 100) + make_rows("d2", "spalling", 50)
    train, test = stratified_split(rows, test_fraction=0.2, seed=42)
    assert len(train) + len(test) == 150
    assert set(r.image_path for r in train).isdisjoint(r.image_path for r in test)


def test_split_is_stratified():
    rows = make_rows("d1", "crack", 100) + make_rows("d2", "spalling", 50)
    train, test = stratified_split(rows, test_fraction=0.2, seed=42)
    test_crack = sum(1 for r in test if r.unified_label == "crack")
    test_spall = sum(1 for r in test if r.unified_label == "spalling")
    assert test_crack == 20
    assert test_spall == 10


def test_split_is_deterministic():
    rows = make_rows("d1", "crack", 100)
    a = stratified_split(rows, test_fraction=0.2, seed=42)
    b = stratified_split(rows, test_fraction=0.2, seed=42)
    assert a == b


def test_small_groups_get_test_representation():
    rows = make_rows("d1", "efflorescence", 5)
    train, test = stratified_split(rows, test_fraction=0.15, seed=42)
    assert len(test) >= 1


def test_split_stable_when_other_datasets_added():
    """Frozen-split contract: adding a dataset must not reshuffle existing groups."""
    rows = make_rows("d1", "crack", 100)
    extra = make_rows("a_new", "spalling", 50)
    _, test_a = stratified_split(rows, test_fraction=0.2, seed=42)
    _, test_b = stratified_split(rows + extra, test_fraction=0.2, seed=42)
    assert test_a == [r for r in test_b if r.source_dataset == "d1"]


def test_frozen_split_artifacts_unchanged():
    """Regression lock on the FROZEN v2 split (user-gated regeneration
    2026-07-21): 27,330/4,824 rows, all 12 classes in both splits. If this
    fails, the frozen artifacts were touched — that requires explicit
    sign-off, not a code fix. (v1 lock: 15,004/2,648, archived as
    test_v1_frozen.csv and enforced by tests/test_split_v2.py.)"""
    train = read_manifest(REPO_ROOT / "data" / "manifests" / "train.csv")
    test = read_manifest(REPO_ROOT / "data" / "manifests" / "test.csv")
    assert len(train) == 27330
    assert len(test) == 4824
    assert {r.unified_label for r in test} == set(UNIFIED_CLASSES)
    assert {r.unified_label for r in train} == set(UNIFIED_CLASSES)


def test_tiny_groups_stay_in_train():
    rows = make_rows("d1", "corrosion_stain", 3)
    train, test = stratified_split(rows, test_fraction=0.15, seed=42)
    assert len(train) == 3
    assert len(test) == 0
