"""Build the frozen agent golden set from the (already frozen) test split.

15 synthetic properties x 5 images: 2-3 distinct defect classes + no_defect
distractors each. Deterministic (seed 42). Refuses to overwrite without
--force: same discipline as every other committed manifest.
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_SPLIT = REPO_ROOT / "data" / "manifests" / "test.csv"
N_PROPERTIES = 15
IMAGES_PER_PROPERTY = 5
SEED = 42


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "data" / "manifests" / "agent_golden.csv"
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.out.exists() and not args.force:
        print(f"{args.out} exists; refusing to overwrite without --force", file=sys.stderr)
        return 1

    by_label: dict[str, list[str]] = defaultdict(list)
    with TEST_SPLIT.open() as f:
        for row in csv.DictReader(f):
            by_label[row["unified_label"]].append(row["image_path"])
    for paths in by_label.values():
        paths.sort()  # order-independence before seeding

    rng = random.Random(SEED)
    defect_classes = sorted(l for l in by_label if l != "no_defect")
    rows: list[dict] = []
    for i in range(N_PROPERTIES):
        pid = f"prop_{i:02d}"
        n_defects = rng.choice([2, 3])
        classes = rng.sample(defect_classes, n_defects)
        picks = [(c, rng.choice(by_label[c])) for c in classes]
        # Fill with no_defect distractors, sampled without replacement so a
        # property never repeats the same image.
        n_fill = IMAGES_PER_PROPERTY - len(picks)
        picks.extend(("no_defect", p) for p in rng.sample(by_label["no_defect"], n_fill))
        rng.shuffle(picks)
        rows.extend(
            {"property_id": pid, "image_path": path, "unified_label": label}
            for label, path in picks
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        # lineterminator matches ingest.py and .gitattributes (eol=lf): the
        # working copy must be byte-identical to the committed manifest.
        writer = csv.DictWriter(
            f,
            fieldnames=["property_id", "image_path", "unified_label"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
