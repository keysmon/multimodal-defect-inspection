from pathlib import Path

from defectlens.ingest import ManifestRow, apply_caps, scan_dataset, write_manifest, read_manifest
from defectlens.taxonomy import load_mapping

MAPPING_YAML = """
mappings:
  - source_dataset: bd3
    source_label: algae
    unified_label: mold_algae
    rationale: t
  - source_dataset: bd3
    source_label: normal
    unified_label: no_defect
    rationale: t
"""


def make_raw(tmp_path: Path) -> Path:
    repo = tmp_path
    for label, names in {"algae": ["a1.jpg", "a2.jpg"], "normal": ["n1.jpg"]}.items():
        d = repo / "data" / "raw" / "bd3" / label
        d.mkdir(parents=True)
        for n in names:
            (d / n).write_bytes(b"fake")
    (repo / "configs").mkdir()
    (repo / "configs" / "mapping.yaml").write_text(MAPPING_YAML)
    return repo


def test_scan_dataset(tmp_path):
    repo = make_raw(tmp_path)
    mapping = load_mapping(repo / "configs" / "mapping.yaml")
    rows = scan_dataset(repo, "bd3", mapping)
    assert len(rows) == 3
    assert rows[0].image_path.startswith("data/raw/bd3/")
    assert {r.unified_label for r in rows} == {"mold_algae", "no_defect"}


def test_scan_is_deterministic(tmp_path):
    repo = make_raw(tmp_path)
    mapping = load_mapping(repo / "configs" / "mapping.yaml")
    assert scan_dataset(repo, "bd3", mapping) == scan_dataset(repo, "bd3", mapping)


def test_apply_caps():
    rows = [
        ManifestRow(f"data/raw/x/l/{i}.jpg", "x", "l", "crack") for i in range(10)
    ]
    capped = apply_caps(rows, caps={"x": {"l": 4}}, seed=17)
    assert len(capped) == 4
    # deterministic
    assert capped == apply_caps(rows, caps={"x": {"l": 4}}, seed=17)
    # uncapped groups untouched
    assert apply_caps(rows, caps={}, seed=17) == sorted(rows, key=lambda r: r.image_path)


def test_manifest_roundtrip(tmp_path):
    rows = [ManifestRow("data/raw/x/l/0.jpg", "x", "l", "crack")]
    out = tmp_path / "manifest.csv"
    write_manifest(rows, out)
    assert read_manifest(out) == rows
