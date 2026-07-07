"""Unified defect taxonomy and the versioned source→unified label mapping."""
from __future__ import annotations

from pathlib import Path

import yaml

UNIFIED_CLASSES = [
    "crack",
    "spalling",
    "efflorescence",
    "exposed_rebar",
    "corrosion_stain",
    "mold_algae",
    "water_damage",
    "peeling_paint",
    "no_defect",
]

EXCLUDE = "EXCLUDE"

Mapping = dict[tuple[str, str], str]


def load_mapping(path: Path) -> Mapping:
    """Load and validate configs/label_mapping.yaml."""
    raw = yaml.safe_load(Path(path).read_text())
    mapping: Mapping = {}
    for entry in raw["mappings"]:
        key = (entry["source_dataset"], entry["source_label"])
        if key in mapping:
            raise ValueError(f"Duplicate mapping for {key}")
        unified = entry["unified_label"]
        if unified != EXCLUDE and unified not in UNIFIED_CLASSES:
            raise ValueError(f"Unknown unified label {unified!r} for {key}")
        mapping[key] = unified
    return mapping


def map_label(mapping: Mapping, source_dataset: str, source_label: str) -> str | None:
    """Return the unified label, or None if the sample is excluded.

    Raises KeyError for unmapped labels so new upstream labels surface loudly
    instead of being silently dropped.
    """
    key = (source_dataset, source_label)
    if key not in mapping:
        raise KeyError(
            f"No mapping for {key} — add it to configs/label_mapping.yaml"
        )
    unified = mapping[key]
    return None if unified == EXCLUDE else unified
