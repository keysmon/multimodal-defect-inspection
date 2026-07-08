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
        # Groups of size <= 3 may get zero test rows by design: too small to
        # split meaningfully, they stay train-only. A future rare-class dataset
        # addition should raise the group above this threshold or accept it.
        test.extend(group[:n_test])
        train.extend(group[n_test:])
    return (
        sorted(train, key=lambda r: r.image_path),
        sorted(test, key=lambda r: r.image_path),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the train/test split.")
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/manifest.csv"))
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--force", action="store_true",
        help="allow overwriting an existing frozen split (requires explicit sign-off)",
    )
    args = parser.parse_args()

    if not args.manifest.is_file():
        raise SystemExit(
            f"{args.manifest} not found — run `python -m defectlens.ingest` first"
        )
    out_dir = args.manifest.parent
    existing = [p for p in (out_dir / "train.csv", out_dir / "test.csv") if p.exists()]
    if existing and not args.force:
        raise SystemExit(
            "Refusing to overwrite the FROZEN split "
            f"({', '.join(str(p) for p in existing)}) — regenerating invalidates "
            "all previously reported numbers; rerun with --force only with "
            "explicit sign-off"
        )

    rows = read_manifest(args.manifest)
    train, test = stratified_split(rows, args.test_fraction, args.seed)
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
