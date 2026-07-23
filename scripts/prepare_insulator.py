"""Derive single-label classification crops from the insulator detection set.

Source (figshare 21200986, CC BY 4.0): 1,600 grid transmission-insulator images
with YOLO-format boxes; label.txt order is [pollution-flashover, broken,
insulator]. The "insulator" class marks the intact insulator OBJECT, so it is
staged as `normal`; defect boxes typically nest inside those object boxes, which
is exactly the containment case crop_utils' overlap coefficient skips — an
"insulator" box containing a flashover/broken region must NOT become a `normal`
crop.

Crops (bbox + 15% margin, min side 96px, content-hash dedupe) are written as
real JPEG files into --out/<staged_label>/, then linked into
data/raw/insulator/ by scripts/normalize_raw.py.

Usage:
  python scripts/prepare_insulator.py \
      --source ~/datasets/insulator/extracted/VOC --out ~/datasets/insulator_crops
"""
from __future__ import annotations

import argparse
from collections import Counter
from io import BytesIO
from pathlib import Path

from PIL import Image

from crop_utils import Box, conflicting, content_hash, expanded_pixel_box, yolo_to_box

# label.txt order (source class ids) -> staged label dirs (must match
# DATASET_LABELS["insulator"] in scripts/normalize_raw.py and
# configs/label_mapping.yaml).
SOURCE_NAMES = ["pollution_flashover", "broken", "normal"]
LABEL_FILE_NAMES = ["pollution-flashover", "broken", "insulator"]


def load_boxes(label_path: Path, width: int, height: int) -> list[Box]:
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            boxes.append(yolo_to_box(line, SOURCE_NAMES, width, height))
    return boxes


def prepare(source: Path, out: Path) -> Counter:
    label_txt = (source / "label.txt").read_text(encoding="utf-8").split()
    if label_txt != LABEL_FILE_NAMES:
        raise SystemExit(
            f"label.txt order changed ({label_txt!r}) — re-verify SOURCE_NAMES"
        )
    stats: Counter = Counter()
    seen_hashes: set[str] = set()
    for split in ("train", "val", "test"):
        image_dir = source / "images" / split
        label_dir = source / "labels" / split
        if not image_dir.is_dir():
            raise SystemExit(f"missing {image_dir} — check --source")
        for image_path in sorted(image_dir.glob("*.jpg")):
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                stats["missing_label"] += 1
                continue
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                boxes = load_boxes(label_path, img.width, img.height)
                for i, box in enumerate(boxes):
                    # "normal" is the OBJECT class: it never pollutes a defect
                    # crop nested inside it, but any defect box pollutes a
                    # would-be normal crop (see crop_utils.conflicting).
                    if conflicting(box, boxes, benign=frozenset({"normal"})):
                        stats["conflicting_overlap_skipped"] += 1
                        continue
                    pixel_box = expanded_pixel_box(box, img.width, img.height)
                    if pixel_box is None:
                        stats["too_small_skipped"] += 1
                        continue
                    crop = img.crop(pixel_box)
                    buf = BytesIO()
                    crop.save(buf, "JPEG", quality=95)
                    digest = content_hash(buf.getvalue())
                    if digest in seen_hashes:
                        stats["duplicate_dropped"] += 1
                        continue
                    seen_hashes.add(digest)
                    dest_dir = out / box.label
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    dest = dest_dir / f"{split}_{image_path.stem}_{i}.jpg"
                    if not dest.exists():
                        dest.write_bytes(buf.getvalue())
                    stats[f"kept_{box.label}"] += 1
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
        raise SystemExit("No crops kept — check --source points at the VOC dir")


if __name__ == "__main__":
    main()
