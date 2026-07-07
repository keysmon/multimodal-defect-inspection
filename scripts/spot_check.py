"""Sample N images per unified class into review_grid/ for manual QA.

Protocol (spec §4): review ~30 images per class; any class whose mapping looks
wrong (>10% of samples don't match their unified label) gets its mapping entry
revisited in configs/label_mapping.yaml BEFORE the split is trusted.
"""
from __future__ import annotations

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from defectlens.ingest import ManifestRow, read_manifest  # noqa: E402


def sample_per_class(
    rows: list[ManifestRow], n_per_class: int, seed: int
) -> list[ManifestRow]:
    grouped: dict[str, list[ManifestRow]] = defaultdict(list)
    for r in rows:
        grouped[r.unified_label].append(r)
    picked: list[ManifestRow] = []
    for label in sorted(grouped):
        group = sorted(grouped[label], key=lambda r: r.image_path)
        rng = random.Random(f"{seed}:{label}")
        k = min(n_per_class, len(group))
        picked.extend(rng.sample(group, k))
    return sorted(picked, key=lambda r: r.image_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/manifest.csv"))
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=Path("review_grid"))
    args = parser.parse_args()

    rows = read_manifest(args.manifest)
    picked = sample_per_class(rows, args.n, args.seed)
    if args.out.exists():
        shutil.rmtree(args.out)
    for r in picked:
        dest_dir = args.out / f"{r.unified_label}"
        dest_dir.mkdir(parents=True, exist_ok=True)
        src = Path(r.image_path)
        dest = dest_dir / f"{r.source_dataset}__{src.name}"
        dest.symlink_to(src.resolve())
    print(f"Wrote review grid to {args.out}/ — open in Finder and review per class.")


if __name__ == "__main__":
    main()
