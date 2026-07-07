"""Reorganize CODEBRIM classification_dataset into per-class dirs via metadata XMLs.

CODEBRIM ships {train,val,test}/{background,defects}/ folders with labels in
metadata/*.xml (multi-label binary flags). Our v1 taxonomy is single-label, so
this keeps ONLY crops with exactly one flag set (multi-label crops are counted
and skipped), symlinking them into a per-class staging tree that
scripts/normalize_raw.py can then ingest as `--dataset codebrim`.

Usage:
  python scripts/prepare_codebrim.py --source ~/datasets/codebrim/classification_dataset --out ~/datasets/codebrim_by_class
"""
from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

# XML tag -> canonical source label (must match DATASET_LABELS["codebrim"]
# in scripts/normalize_raw.py and configs/label_mapping.yaml).
TAG_TO_LABEL = {
    "Background": "background",
    "Crack": "crack",
    "Spallation": "spallation",
    "Efflorescence": "efflorescence",
    "ExposedBars": "exposed_bars",
    "CorrosionStain": "corrosion_stain",
}


def parse_metadata(xml_path: Path) -> dict[str, list[str]]:
    """name -> list of active canonical labels."""
    root = ET.parse(xml_path).getroot()
    out: dict[str, list[str]] = {}
    for defect in root.findall("Defect"):
        name = defect.get("name")
        if not name:
            raise ValueError(f"{xml_path}: <Defect> without name attribute")
        labels = [
            label
            for tag, label in TAG_TO_LABEL.items()
            if (el := defect.find(tag)) is not None and (el.text or "").strip() == "1"
        ]
        out[name] = labels
    return out


def prepare(source: Path, out: Path) -> Counter:
    """Symlink single-label crops into out/<label>/; return outcome counts."""
    metadata_dir = source / "metadata"
    name_to_labels: dict[str, list[str]] = {}
    for xml_name in ("defects.xml", "background.xml"):
        name_to_labels.update(parse_metadata(metadata_dir / xml_name))

    stats: Counter = Counter()
    seen_names: set[str] = set()
    for img in sorted(source.rglob("*.png")):
        if img.parent.name not in {"background", "defects"}:
            continue
        seen_names.add(img.name)
        labels = name_to_labels.get(img.name)
        if labels is None:
            stats["missing_metadata"] += 1
            continue
        if len(labels) != 1:
            stats["multi_label_skipped"] += 1
            continue
        label_dir = out / labels[0]
        label_dir.mkdir(parents=True, exist_ok=True)
        dest = label_dir / img.name
        if dest.is_symlink() and not dest.exists():
            dest.unlink()
        if not dest.exists():
            dest.symlink_to(img.resolve())
        stats[f"kept_{labels[0]}"] += 1

    stats["metadata_without_file"] = sum(
        1 for name in name_to_labels if name not in seen_names
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    stats = prepare(args.source, args.out)
    for key in sorted(stats):
        print(f"{key}: {stats[key]}")
    kept = sum(v for k, v in stats.items() if k.startswith("kept_"))
    print(f"total kept: {kept}")
    if kept == 0:
        raise SystemExit("No crops kept — check --source points at classification_dataset")


if __name__ == "__main__":
    main()
