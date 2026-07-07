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
    """Scan one dataset's canonical raw tree into manifest rows.

    Re-checks raw-tree integrity independently of scripts/verify_raw.py as
    defense in depth: broken symlinks and images reachable from two label
    dirs fail loud here rather than silently corrupting the manifest.
    """
    dataset_dir = repo_root / "data" / "raw" / dataset_name
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"{dataset_dir} not found — run scripts/normalize_raw.py")
    rows: list[ManifestRow] = []
    seen_real: dict[Path, str] = {}
    for label_dir in sorted(d for d in dataset_dir.iterdir() if d.is_dir()):
        unified = map_label(mapping, dataset_name, label_dir.name)
        if unified is None:
            continue
        for img in sorted(label_dir.rglob("*")):
            if img.suffix.lower() not in IMAGE_EXTS:
                continue
            if img.is_symlink() and not img.exists():
                raise FileNotFoundError(
                    f"Broken symlink {img} — re-run scripts/normalize_raw.py "
                    "(scripts/verify_raw.py gives a full report)"
                )
            if not img.is_file():
                continue
            real = img.resolve()
            other = seen_real.get(real)
            if other is not None and other != label_dir.name:
                raise ValueError(
                    f"{real} is reachable from two labels ('{other}', "
                    f"'{label_dir.name}') — raw tree is double-labeled "
                    "(scripts/verify_raw.py gives a full report)"
                )
            seen_real[real] = label_dir.name
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
    """Deterministically subsample groups exceeding their cap.

    Each group gets its own RNG seeded from (seed, dataset, label), so a
    group's sample is stable regardless of which other datasets or groups
    exist in the manifest — required by the frozen-split contract.
    """
    grouped: dict[tuple[str, str], list[ManifestRow]] = defaultdict(list)
    for r in rows:
        grouped[(r.source_dataset, r.source_label)].append(r)
    out: list[ManifestRow] = []
    for key in sorted(grouped):
        group = sorted(grouped[key], key=lambda r: r.image_path)
        cap = caps.get(key[0], {}).get(key[1])
        if cap is not None and len(group) > cap:
            rng = random.Random(f"{seed}:{key[0]}:{key[1]}")
            group = rng.sample(group, cap)
        out.extend(group)
    return sorted(out, key=lambda r: r.image_path)


def write_manifest(rows: list[ManifestRow], path: Path) -> None:
    """Write manifest rows to a CSV with a fixed header (creates parent dirs)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow(r.__dict__)


def read_manifest(path: Path) -> list[ManifestRow]:
    """Read a manifest CSV back into ManifestRow objects."""
    with path.open(newline="", encoding="utf-8") as f:
        return [ManifestRow(**row) for row in csv.DictReader(f)]


def main() -> None:
    """CLI entrypoint: scan data/raw, apply sampling caps, write the manifest."""
    parser = argparse.ArgumentParser(description="Build the unified manifest CSV.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=Path("data/manifests/manifest.csv"))
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    for p in (repo / "configs" / "label_mapping.yaml", repo / "configs" / "sampling.yaml"):
        if not p.is_file():
            raise SystemExit(f"{p} not found — run from the repo root")
    raw_root = repo / "data" / "raw"
    if not raw_root.is_dir():
        raise SystemExit(f"{raw_root} not found — run scripts/normalize_raw.py first")

    mapping = load_mapping(repo / "configs" / "label_mapping.yaml")
    sampling = yaml.safe_load((repo / "configs" / "sampling.yaml").read_text(encoding="utf-8"))

    rows: list[ManifestRow] = []
    for dataset_dir in sorted(raw_root.iterdir()):
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
