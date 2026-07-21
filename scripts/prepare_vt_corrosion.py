"""Stage VT Corrosion Condition State images into per-state dirs for normalize_raw.

The VT dataset (DOI 10.7294/16624663, CC0) is semantic segmentation: labelme
JSONs whose polygon labels carry AASHTO/BIRM condition states
(2_Fair / 3_Poor / 4_Severe steel corrosion). "Good" regions are unannotated
background, so NO whole image is state good — every shipped image contains
corrosion. This script derives one classification label per image = the WORST
(highest-numbered) state present among its polygons, and symlinks whole images
into a per-state staging tree that scripts/normalize_raw.py ingests as
`--dataset vt_corrosion`.

Source stems collide across Train/Test (both have 0.jpeg...), so staged names
are prefixed with the source split: train_379.jpeg, test_10.jpeg.

Usage:
  python scripts/prepare_vt_corrosion.py \
      --source ~/datasets/vt_corrosion/extracted/"Corrosion Condition State Classification" \
      --out ~/datasets/vt_corrosion_by_state
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

# Leading digit of the labelme polygon label -> canonical source state
# (must match DATASET_LABELS["vt_corrosion"] in scripts/normalize_raw.py
# and configs/label_mapping.yaml).
STATE_BY_DIGIT = {1: "good", 2: "fair", 3: "poor", 4: "severe"}


def worst_state(json_path: Path) -> str | None:
    """Return the worst (max-digit) condition state among the image's polygons."""
    shapes = json.loads(json_path.read_text(encoding="utf-8"))["shapes"]
    digits = []
    for shape in shapes:
        label = shape["label"]
        digit = int(label.split("_", 1)[0])
        if digit not in STATE_BY_DIGIT:
            raise ValueError(f"{json_path}: unknown state digit in label {label!r}")
        digits.append(digit)
    return STATE_BY_DIGIT[max(digits)] if digits else None


def prepare(source: Path, out: Path) -> Counter:
    """Symlink whole images into out/<state>/; return per-state counts."""
    stats: Counter = Counter()
    for split in ("Train", "Test"):
        json_dir = source / "original" / split / "json"
        image_dir = source / "original" / split / "images"
        if not json_dir.is_dir():
            raise SystemExit(f"missing {json_dir} — check --source")
        for json_path in sorted(json_dir.glob("*.json")):
            image_path = image_dir / f"{json_path.stem}.jpeg"
            if not image_path.is_file():
                stats["missing_image"] += 1
                continue
            state = worst_state(json_path)
            if state is None:
                stats["no_polygons_skipped"] += 1
                continue
            state_dir = out / state
            state_dir.mkdir(parents=True, exist_ok=True)
            dest = state_dir / f"{split.lower()}_{image_path.name}"
            if dest.is_symlink() and not dest.exists():
                dest.unlink()
            if not dest.exists():
                dest.symlink_to(image_path.resolve())
            stats[f"kept_{state}"] += 1
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
        raise SystemExit(
            "No images kept — check --source points at 'Corrosion Condition State Classification'"
        )


if __name__ == "__main__":
    main()
