"""Derive single-label classification crops from MBDD2025 detection boxes.

Source (Zenodo 15622584, CC BY 4.0): 14,471 UAV images of multi-material
building walls with PASCAL VOC XML boxes over 5 defect classes
(crack, leakage, corrosion, abscission, bulge). The XMLs are authoritative
per the dataset README; the parallel YOLO Labels/ dir is ignored.

Crops (bbox + 15% margin, min side 96px, different-class overlap skip,
content-hash dedupe — see scripts/crop_utils.py) are written as real JPEG
files into --out/<source_label>/, then linked into data/raw/mbdd2025/ by
scripts/normalize_raw.py.

Usage:
  python scripts/prepare_mbdd.py \
      --source ~/datasets/mbdd2025/MBDD2025 --out ~/datasets/mbdd2025_crops
"""
from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from collections import Counter
from io import BytesIO
from pathlib import Path

from PIL import Image

from crop_utils import Box, conflicting, content_hash, expanded_pixel_box

# Dataset's own label names (must match DATASET_LABELS["mbdd2025"] in
# scripts/normalize_raw.py and configs/label_mapping.yaml).
SOURCE_LABELS = {"crack", "leakage", "corrosion", "abscission", "bulge"}


def _coord(bnd: ET.Element, tag: str, xml_path: Path) -> float:
    text = bnd.findtext(tag)
    if text is None:
        raise ValueError(f"{xml_path}: <bndbox> missing <{tag}>")
    return float(text)


def load_boxes(xml_path: Path) -> list[Box]:
    root = ET.parse(xml_path).getroot()
    boxes = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        if name not in SOURCE_LABELS:
            raise ValueError(f"{xml_path}: unknown class {name!r}")
        bnd = obj.find("bndbox")
        if bnd is None:
            raise ValueError(f"{xml_path}: <object> without <bndbox>")
        boxes.append(
            Box(
                label=name,
                x1=_coord(bnd, "xmin", xml_path),
                y1=_coord(bnd, "ymin", xml_path),
                x2=_coord(bnd, "xmax", xml_path),
                y2=_coord(bnd, "ymax", xml_path),
            )
        )
    return boxes


def prepare(source: Path, out: Path) -> Counter:
    annotations = source / "Annotations"
    images = source / "JPEGImages"
    if not annotations.is_dir():
        raise SystemExit(f"missing {annotations} — check --source")
    stats: Counter = Counter()
    seen_hashes: set[str] = set()
    for xml_path in sorted(annotations.glob("*.xml")):
        image_path = images / f"{xml_path.stem}.jpg"
        if not image_path.is_file():
            stats["missing_image"] += 1
            continue
        boxes = load_boxes(xml_path)
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            for i, box in enumerate(boxes):
                if conflicting(box, boxes):
                    stats["conflicting_overlap_skipped"] += 1
                    continue
                pixel_box = expanded_pixel_box(box, img.width, img.height)
                if pixel_box is None:
                    stats["too_small_skipped"] += 1
                    continue
                buf = BytesIO()
                img.crop(pixel_box).save(buf, "JPEG", quality=95)
                digest = content_hash(buf.getvalue())
                if digest in seen_hashes:
                    stats["duplicate_dropped"] += 1
                    continue
                seen_hashes.add(digest)
                dest_dir = out / box.label
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest = dest_dir / f"{xml_path.stem}_{i}.jpg"
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
        raise SystemExit("No crops kept — check --source points at MBDD2025/")


if __name__ == "__main__":
    main()
