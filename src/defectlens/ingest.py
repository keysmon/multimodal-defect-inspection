"""Scan canonical raw layout into the unified manifest CSV."""
from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import yaml

from defectlens.taxonomy import LabelMapping, load_mapping, map_label

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
FIELDS = ["image_path", "source_dataset", "source_label", "unified_label"]


@dataclass(frozen=True)
class ManifestRow:
    image_path: str  # posix path relative to repo root
    source_dataset: str
    source_label: str
    unified_label: str


def scan_dataset(repo_root: Path, dataset_name: str, mapping: LabelMapping) -> list[ManifestRow]:
    dataset_dir = repo_root / "data" / "raw" / dataset_name
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"{dataset_dir} not found — run scripts/normalize_raw.py")
    rows: list[ManifestRow] = []
    for label_dir in sorted(d for d in dataset_dir.iterdir() if d.is_dir()):
        unified = map_label(mapping, dataset_name, label_dir.name)
        if unified is None:
            continue
        for img in sorted(label_dir.rglob("*")):
            if img.suffix.lower() in IMAGE_EXTS and img.is_file():
                rows.append(
                    ManifestRow(
                        image_path=img.relative_to(repo_root).as_posix(),
                        source_dataset=dataset_name,
                        source_label=label_dir.name,
                        unified_label=unified,
                    )
                )
    return rows


def apply_caps(
    rows: list[ManifestRow], caps: dict[str, dict[str, int]], seed: int
) -> list[ManifestRow]:
    """Deterministically subsample groups exceeding their cap."""
    grouped: dict[tuple[str, str], list[ManifestRow]] = defaultdict(list)
    for r in rows:
        grouped[(r.source_dataset, r.source_label)].append(r)
    rng = random.Random(seed)
    out: list[ManifestRow] = []
    for key in sorted(grouped):
        group = sorted(grouped[key], key=lambda r: r.image_path)
        cap = caps.get(key[0], {}).get(key[1])
        if cap is not None and len(group) > cap:
            group = rng.sample(group, cap)
        out.extend(group)
    return sorted(out, key=lambda r: r.image_path)


def write_manifest(rows: list[ManifestRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow(r.__dict__)


def read_manifest(path: Path) -> list[ManifestRow]:
    with path.open(newline="") as f:
        return [ManifestRow(**row) for row in csv.DictReader(f)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the unified manifest CSV.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=Path("data/manifests/manifest.csv"))
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    mapping = load_mapping(repo / "configs" / "label_mapping.yaml")
    sampling = yaml.safe_load((repo / "configs" / "sampling.yaml").read_text())

    rows: list[ManifestRow] = []
    for dataset_dir in sorted((repo / "data" / "raw").iterdir()):
        if dataset_dir.is_dir():
            rows.extend(scan_dataset(repo, dataset_dir.name, mapping))
    rows = apply_caps(rows, sampling.get("caps", {}), sampling["seed"])
    write_manifest(rows, repo / args.out)

    counts: dict[str, int] = defaultdict(int)
    for r in rows:
        counts[r.unified_label] += 1
    print(f"Wrote {len(rows)} rows to {args.out}")
    for label in sorted(counts):
        print(f"  {label}: {counts[label]}")


if __name__ == "__main__":
    main()
