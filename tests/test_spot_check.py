import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from spot_check import sample_per_class  # noqa: E402

from defectlens.ingest import ManifestRow


def test_sample_per_class_deterministic_and_bounded():
    rows = [
        ManifestRow(f"data/raw/d/crack/{i}.jpg", "d", "crack", "crack") for i in range(100)
    ] + [
        ManifestRow(f"data/raw/d/algae/{i}.jpg", "d", "algae", "mold_algae") for i in range(3)
    ]
    picked = sample_per_class(rows, n_per_class=30, seed=7)
    by_class = {}
    for r in picked:
        by_class.setdefault(r.unified_label, []).append(r)
    assert len(by_class["crack"]) == 30
    assert len(by_class["mold_algae"]) == 3  # fewer than n -> take all
    assert picked == sample_per_class(rows, n_per_class=30, seed=7)
