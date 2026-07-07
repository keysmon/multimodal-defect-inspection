from pathlib import Path

import pytest

from defectlens.taxonomy import UNIFIED_CLASSES, load_mapping, map_label

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_MAPPING = REPO_ROOT / "configs" / "label_mapping.yaml"


def write_mapping(tmp_path, text):
    p = tmp_path / "mapping.yaml"
    p.write_text(text)
    return p


def test_unified_classes_are_nine():
    assert len(UNIFIED_CLASSES) == 9
    assert "no_defect" in UNIFIED_CLASSES


def test_load_and_map(tmp_path):
    p = write_mapping(
        tmp_path,
        """
mappings:
  - source_dataset: bd3
    source_label: algae
    unified_label: mold_algae
    rationale: biological growth
""",
    )
    mapping = load_mapping(p)
    assert map_label(mapping, "bd3", "algae") == "mold_algae"


def test_exclude_returns_none(tmp_path):
    p = write_mapping(
        tmp_path,
        """
mappings:
  - source_dataset: bd3
    source_label: junk
    unified_label: EXCLUDE
    rationale: not a defect class
""",
    )
    mapping = load_mapping(p)
    assert map_label(mapping, "bd3", "junk") is None


def test_unknown_unified_label_rejected(tmp_path):
    p = write_mapping(
        tmp_path,
        """
mappings:
  - source_dataset: bd3
    source_label: algae
    unified_label: not_a_class
    rationale: typo
""",
    )
    with pytest.raises(ValueError, match="not_a_class"):
        load_mapping(p)


def test_duplicate_mapping_rejected(tmp_path):
    p = write_mapping(
        tmp_path,
        """
mappings:
  - source_dataset: bd3
    source_label: algae
    unified_label: mold_algae
    rationale: a
  - source_dataset: bd3
    source_label: algae
    unified_label: water_damage
    rationale: b
""",
    )
    with pytest.raises(ValueError, match="Duplicate"):
        load_mapping(p)


def test_unmapped_label_raises(tmp_path):
    p = write_mapping(
        tmp_path,
        """
mappings:
  - source_dataset: bd3
    source_label: algae
    unified_label: mold_algae
    rationale: a
""",
    )
    mapping = load_mapping(p)
    with pytest.raises(KeyError):
        map_label(mapping, "bd3", "never_seen")


def test_real_mapping_file_is_valid_and_complete():
    mapping = load_mapping(REAL_MAPPING)
    expected_sources = {
        ("codebrim", "background"),
        ("codebrim", "crack"),
        ("codebrim", "spallation"),
        ("codebrim", "efflorescence"),
        ("codebrim", "exposed_bars"),
        ("codebrim", "corrosion_stain"),
        ("bd3", "algae"),
        ("bd3", "major_crack"),
        ("bd3", "minor_crack"),
        ("bd3", "peeling"),
        ("bd3", "spalling"),
        ("bd3", "stain"),
        ("bd3", "normal"),
        ("roboflow_walls", "crack"),
        ("roboflow_walls", "mold"),
        ("roboflow_walls", "peeling_paint"),
        ("roboflow_walls", "stairstep_crack"),
        ("roboflow_walls", "water_seepage"),
        ("sdnet2018", "cracked"),
        ("sdnet2018", "non_cracked"),
    }
    assert expected_sources.issubset(mapping.keys())
