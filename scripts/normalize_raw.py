"""Normalize downloaded datasets into data/raw/<dataset>/<source_label>/ symlinks.

Usage:
  python scripts/normalize_raw.py --dataset bd3 --source /path/to/bd3_clone
  python scripts/normalize_raw.py --dataset sdnet2018 --source /path/to/SDNET2018
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# Canonical source labels per dataset (must match configs/label_mapping.yaml).
DATASET_LABELS: dict[str, set[str]] = {
    "codebrim": {
        "background", "crack", "spallation", "efflorescence",
        "exposed_bars", "corrosion_stain",
    },
    "bd3": {
        "algae", "major_crack", "minor_crack", "peeling",
        "spalling", "stain", "normal", "plain",
    },
    "roboflow_walls": {
        "crack", "mold", "peeling_paint", "stairstep_crack", "water_seepage",
    },
    "sdnet2018": {"cracked", "non_cracked"},
}


def canon(name: str) -> str:
    """Lowercase and strip all non-alphanumeric chars for fuzzy dir matching."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def match_label(dirname: str, labels: set[str]) -> str | None:
    """Match a directory name against canonical labels, tolerant of case/spacing."""
    c = canon(dirname)
    for label in labels:
        if c == canon(label):
            return label
    return None


def _link_images(label_dir_src: Path, label_dir_dest: Path, rel_root: Path) -> int:
    label_dir_dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for img in sorted(label_dir_src.rglob("*")):
        if img.suffix.lower() not in IMAGE_EXTS or not img.is_file():
            continue
        # Flatten relative path into the filename to avoid collisions.
        flat = "__".join(img.relative_to(rel_root).parts)
        dest_path = label_dir_dest / flat
        if dest_path.is_symlink() and not dest_path.exists():
            dest_path.unlink()  # dangling link from a moved/deleted source — relink
        if not dest_path.exists():
            dest_path.symlink_to(img.resolve())
            n += 1
    return n


def normalize_generic(source: Path, dest: Path, labels: set[str]) -> int:
    """Find dirs anywhere under `source` whose name matches a known label;
    symlink their images into dest/<canonical_label>/.

    Refuses to run if a matched label dir is nested inside another matched
    label dir — _link_images recurses, so nesting would silently link the
    same image under two labels.
    """
    matched = {d for d in source.rglob("*") if d.is_dir() and match_label(d.name, labels)}
    nested = sorted(str(d) for d in matched if any(a in matched for a in d.parents))
    if nested:
        raise SystemExit(
            "Refusing to normalize: label directories nested inside other "
            f"label directories (ambiguous labeling): {nested}"
        )
    n = 0
    for d in sorted(matched):
        label = match_label(d.name, labels)
        n += _link_images(d, dest / label, rel_root=source)
    return n


def normalize_sdnet(source: Path, dest: Path) -> int:
    """SDNET2018: {D,P,W}/{C*,U*}/*.jpg — C=cracked, U=non_cracked."""
    n = 0
    for sub in sorted(p for p in source.rglob("*") if p.is_dir()):
        if sub.parent.name not in {"D", "P", "W"}:
            continue
        if sub.name.upper().startswith("C") and len(sub.name) == 2:
            label = "cracked"
        elif sub.name.upper().startswith("U") and len(sub.name) == 2:
            label = "non_cracked"
        else:
            continue
        n += _link_images(sub, dest / label, rel_root=source)
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_LABELS))
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument(
        "--dest-root", type=Path, default=Path("data/raw"),
        help="canonical raw root (default: data/raw)",
    )
    args = parser.parse_args()

    dest = args.dest_root / args.dataset
    if args.dataset == "sdnet2018":
        n = normalize_sdnet(args.source, dest)
    else:
        n = normalize_generic(args.source, dest, DATASET_LABELS[args.dataset])
    print(f"Linked {n} images into {dest}")
    if n == 0:
        existing = sum(1 for p in dest.rglob("*") if p.is_file())
        if existing == 0:
            raise SystemExit(
                "No images linked — check that --source points at the extracted dataset."
            )
        print(f"(idempotent re-run: {existing} images already linked)")


if __name__ == "__main__":
    main()
