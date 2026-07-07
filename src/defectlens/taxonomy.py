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

LabelMapping = dict[tuple[str, str], str]

REQUIRED_KEYS = ("source_dataset", "source_label", "unified_label", "rationale")


def load_mapping(path: Path | str) -> LabelMapping:
    """Load and validate configs/label_mapping.yaml."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict) or not isinstance(raw.get("mappings"), list):
        raise ValueError(f"{path}: expected a top-level 'mappings' list")
    mapping: LabelMapping = {}
    for i, entry in enumerate(raw["mappings"]):
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: entry {i} is not a mapping: {entry!r}")
        missing = [k for k in REQUIRED_KEYS if not entry.get(k)]
        if missing:
            raise ValueError(f"{path}: entry {i} missing required key(s) {missing}: {entry!r}")
        key = (entry["source_dataset"], entry["source_label"])
        if key in mapping:
            raise ValueError(f"Duplicate mapping for {key}")
        unified = entry["unified_label"]
        if unified != EXCLUDE and unified not in UNIFIED_CLASSES:
            raise ValueError(f"Unknown unified label {unified!r} for {key}")
        mapping[key] = unified
    return mapping


def map_label(mapping: LabelMapping, source_dataset: str, source_label: str) -> str | None:
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
