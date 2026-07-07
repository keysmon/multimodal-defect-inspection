"""Verify canonical raw layout: print per-dataset/per-label counts.

Exits 1 if a REQUIRED dataset is missing or has an empty label dir.
Roboflow is optional (its classes are covered by BD3).
"""
from __future__ import annotations

import sys
from pathlib import Path

REQUIRED = ["codebrim", "bd3", "sdnet2018"]
OPTIONAL = ["roboflow_walls"]


def main() -> int:
    raw = Path("data/raw")
    failed = False
    for dataset in REQUIRED + OPTIONAL:
        ddir = raw / dataset
        required = dataset in REQUIRED
        if not ddir.is_dir():
            level = "MISSING (required)" if required else "absent (optional, OK)"
            print(f"{dataset}: {level}")
            failed |= required
            continue
        for label_dir in sorted(p for p in ddir.iterdir() if p.is_dir()):
            count = sum(1 for f in label_dir.iterdir() if f.is_file())
            flag = "" if count > 0 else "  <-- EMPTY"
            print(f"{dataset}/{label_dir.name}: {count}{flag}")
            failed |= (count == 0 and required)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
