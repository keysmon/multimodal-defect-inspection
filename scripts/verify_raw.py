"""Verify canonical raw layout: per-dataset/per-label counts + symlink health.

Exits 1 if a REQUIRED dataset is missing/empty, has broken symlinks, or if
any real file is reachable from two different label dirs (double-labeling).
Roboflow is optional (its classes are covered by BD3).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REQUIRED = [
    "codebrim", "bd3", "sdnet2018",
    # Taxonomy v2 (2026-07-21):
    "mbdd2025", "vt_corrosion", "insulator",
]
OPTIONAL = ["roboflow_walls"]


def verify(raw: Path) -> int:
    failed = False
    for dataset in REQUIRED + OPTIONAL:
        ddir = raw / dataset
        required = dataset in REQUIRED
        if not ddir.is_dir():
            level = "MISSING (required)" if required else "absent (optional, OK)"
            print(f"{dataset}: {level}")
            failed |= required
            continue
        seen: dict[Path, str] = {}  # resolved real path -> label
        for label_dir in sorted(p for p in ddir.iterdir() if p.is_dir()):
            entries = list(label_dir.iterdir())
            count = sum(1 for f in entries if f.is_file())
            broken = sum(1 for f in entries if f.is_symlink() and not f.exists())
            for f in entries:
                if not f.is_file():
                    continue
                real = f.resolve()
                other = seen.get(real)
                if other is not None and other != label_dir.name:
                    print(f"  DOUBLE-LABELED: {real} in both '{other}' and '{label_dir.name}'")
                    failed = True
                else:
                    seen[real] = label_dir.name
            flags = []
            if count == 0:
                flags.append("EMPTY")
            if broken:
                flags.append(f"{broken} BROKEN LINKS")
            suffix = f"  <-- {', '.join(flags)}" if flags else ""
            print(f"{dataset}/{label_dir.name}: {count}{suffix}")
            failed |= required and (count == 0 or broken > 0)
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    return verify(args.raw_root)


if __name__ == "__main__":
    sys.exit(main())
