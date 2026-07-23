"""Deterministic photo sampling for the localization spike."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from localization_spike import SPIKE_CLASSES, sample_manifest_rows


class FakeRow:
    def __init__(self, image_path, unified_label):
        self.image_path = image_path
        self.unified_label = unified_label


def test_sample_is_deterministic_and_capped_per_class():
    rows = [
        FakeRow(f"img_{label}_{i}.png", label)
        for label in SPIKE_CLASSES
        for i in range(10)
    ] + [FakeRow("other.png", "no_defect")]
    picked = sample_manifest_rows(rows, n_per_class=4, seed=42)
    assert len(picked) == 4 * len(SPIKE_CLASSES)
    assert {r.unified_label for r in picked} == set(SPIKE_CLASSES)
    assert [r.image_path for r in picked] == [
        r.image_path for r in sample_manifest_rows(rows, n_per_class=4, seed=42)
    ]
