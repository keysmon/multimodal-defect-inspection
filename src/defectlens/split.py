"""Seeded stratified split; outputs are FROZEN once committed (spec §4)."""
from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

from defectlens.ingest import ManifestRow, read_manifest, write_manifest


def stratified_split(
    rows: list[ManifestRow], test_fraction: float, seed: int
) -> tuple[list[ManifestRow], list[ManifestRow]]:
    """Stratify by (source_dataset, unified_label)."""
    grouped: dict[tuple[str, str], list[ManifestRow]] = defaultdict(list)
    for r in rows:
        grouped[(r.source_dataset, r.unified_label)].append(r)
    train: list[ManifestRow] = []
    test: list[ManifestRow] = []
    for key in sorted(grouped):
        group = sorted(grouped[key], key=lambda r: r.image_path)
        # Per-group RNG: split membership must not depend on which other
        # datasets exist (frozen-split contract; string seeding is
        # SHA-512-based and PYTHONHASHSEED-immune).
        rng = random.Random(f"{seed}:{key[0]}:{key[1]}")
        rng.shuffle(group)
        n_test = round(len(group) * test_fraction)
        if len(group) >= 4:
            n_test = max(1, n_test)
        test.extend(group[:n_test])
        train.extend(group[n_test:])
    key = lambda r: r.image_path  # noqa: E731
    return sorted(train, key=key), sorted(test, key=key)


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the train/test split.")
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/manifest.csv"))
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = read_manifest(args.manifest)
    train, test = stratified_split(rows, args.test_fraction, args.seed)
    out_dir = args.manifest.parent
    write_manifest(train, out_dir / "train.csv")
    write_manifest(test, out_dir / "test.csv")

    def table(name: str, split_rows: list[ManifestRow]) -> None:
        counts: dict[str, int] = defaultdict(int)
        for r in split_rows:
            counts[r.unified_label] += 1
        print(f"{name}: {len(split_rows)} rows")
        for label in sorted(counts):
            print(f"  {label}: {counts[label]}")

    table("train", train)
    table("test", test)


if __name__ == "__main__":
    main()
