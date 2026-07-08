"""Package Phase 3 fine-tune data + project code for S3 upload.

Builds dist/phase3/{train_images.tar,test_images.tar} containing exactly the
images referenced by data/manifests/{train,test}.csv, stored at their
manifest-relative paths. data/raw/**/*.jpg entries are symlinks into a
machine-local dataset cache outside the repo (see scripts/normalize_raw.py),
so the tars are built with tarfile's `dereference=True` — the GPU box gets
real image bytes at the same relative paths, not dangling links to a path
that only exists on this laptop.

Also copies the manifests + configs/label_mapping.yaml into dist/phase3/,
and (by default) builds a wheel of the defectlens project so the GPU box can
`pip install` the project code without git/network access to GitHub.

Usage:
  python scripts/package_data.py                  # full package, no upload
  python scripts/package_data.py --subset 20       # tiny validation tars
  python scripts/package_data.py --upload          # also aws s3 cp to S3
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from defectlens.ingest import ManifestRow, read_manifest  # noqa: E402

DEFAULT_BUCKET = "defectlens-phase3-002559670021"
DEFAULT_PROFILE = "defectlens"

# ---------------------------------------------------------------------------
# Pure functions (TDD-covered; no AWS/subprocess/network involved)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TarMember:
    arcname: str  # path inside the tar == manifest-relative image_path
    source: Path  # absolute filesystem path to read bytes from


def load_rows(manifest_path: Path, subset: int | None) -> list[ManifestRow]:
    """Read a manifest CSV, optionally truncated to its first `subset` rows.

    `--subset N` exists for cheap end-to-end validation of the packaging
    pipeline (tiny tars) — it takes the manifest's first N rows in file
    order, not a class-balanced sample (that's qlora.subset_rows's job for
    training; packaging just needs *some* real images fast).
    """
    rows = read_manifest(manifest_path)
    if subset is not None:
        rows = rows[:subset]
    return rows


def manifest_members(rows: list[ManifestRow], repo_root: Path) -> list[TarMember]:
    """Manifest rows -> deduped list of TarMember, preserving first-seen order.

    Dedup guards against a manifest accidentally listing the same image_path
    twice — the second occurrence contributes nothing to the tar. Order is
    preserved (not sorted) so tar contents are deterministic and diffable
    across re-packaging runs of the same manifest.
    """
    seen: set[str] = set()
    members: list[TarMember] = []
    for row in rows:
        if row.image_path in seen:
            continue
        seen.add(row.image_path)
        members.append(TarMember(arcname=row.image_path, source=repo_root / row.image_path))
    return members


# ---------------------------------------------------------------------------
# I/O functions (real tar/file writes; wheel build and upload are
# subprocess-based and intentionally not unit-tested — exercised for real by
# `--subset N` runs instead)
# ---------------------------------------------------------------------------


def build_tar(members: list[TarMember], tar_path: Path) -> None:
    """Write `members` into tar_path, dereferencing symlinks.

    Fails loudly (before opening the tar) if any member's source is missing
    or a broken symlink, rather than letting tarfile raise mid-write and
    leave a truncated tar on disk.
    """
    missing = [m for m in members if not m.source.exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} manifest image(s) missing or broken symlinks, "
            f"e.g. {missing[0].source} (re-run scripts/normalize_raw.py?)"
        )
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "w", dereference=True) as tar:
        for m in members:
            tar.add(m.source, arcname=m.arcname, recursive=False)


def copy_support_files(repo_root: Path, out_dir: Path, manifest_paths: list[Path]) -> None:
    """Copy the given manifest CSVs + configs/label_mapping.yaml into out_dir."""
    manifests_out = out_dir / "manifests"
    manifests_out.mkdir(parents=True, exist_ok=True)
    for p in manifest_paths:
        shutil.copy2(p, manifests_out / p.name)

    configs_out = out_dir / "configs"
    configs_out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        repo_root / "configs" / "label_mapping.yaml", configs_out / "label_mapping.yaml"
    )


def build_wheel(repo_root: Path, out_dir: Path) -> Path:
    """`pip wheel` the project (no deps) into out_dir, built from an isolated
    temp copy of just pyproject.toml + src/.

    Lets the GPU box `pip install` the exact project code from S3 without
    needing git/GitHub access from inside the bootstrap script.

    Building from a throwaway copy (rather than `pip wheel repo_root`
    directly) avoids polluting the real working tree: empirically,
    setuptools' build backend writes a `build/` directory into whatever
    source tree it's given, regardless of cwd or the `-w` destination — that
    directory landing in repo_root every packaging run is unwanted clutter
    (and a git-status footgun) in the actual project checkout.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        src_copy = Path(tmp) / "defectlens-src"
        shutil.copytree(
            repo_root / "src",
            src_copy / "src",
            ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"),
        )
        shutil.copy2(repo_root / "pyproject.toml", src_copy / "pyproject.toml")
        subprocess.run(
            [sys.executable, "-m", "pip", "wheel", str(src_copy), "--no-deps", "-w", str(out_dir)],
            check=True,
        )
    wheels = sorted(out_dir.glob("defectlens-*.whl"))
    if not wheels:
        raise RuntimeError(f"pip wheel produced no defectlens-*.whl in {out_dir}")
    return wheels[-1]


def upload(out_dir: Path, bucket: str, profile: str) -> None:
    subprocess.run(
        [
            "aws", "s3", "cp", str(out_dir), f"s3://{bucket}/phase3/",
            "--recursive", "--profile", profile,
        ],
        check=True,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    repo_root_default = Path(__file__).resolve().parents[1]
    parser.add_argument("--repo-root", type=Path, default=repo_root_default)
    parser.add_argument(
        "--out-dir", type=Path, default=None, help="default: <repo-root>/dist/phase3"
    )
    parser.add_argument(
        "--subset", type=int, default=None,
        help="use only the first N rows of each manifest (small validation tars)",
    )
    parser.add_argument(
        "--skip-wheel", action="store_true", help="skip the pip-wheel build step"
    )
    parser.add_argument(
        "--upload", action="store_true", help="aws s3 cp the package to S3 after building"
    )
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)

    repo_root = args.repo_root.resolve()
    out_dir = (args.out_dir or repo_root / "dist" / "phase3").resolve()
    manifests_dir = repo_root / "data" / "manifests"
    train_path = manifests_dir / "train.csv"
    test_path = manifests_dir / "test.csv"

    train_rows = load_rows(train_path, args.subset)
    test_rows = load_rows(test_path, args.subset)

    train_members = manifest_members(train_rows, repo_root)
    test_members = manifest_members(test_rows, repo_root)

    build_tar(train_members, out_dir / "train_images.tar")
    print(f"train_images.tar: {len(train_members)} images")
    build_tar(test_members, out_dir / "test_images.tar")
    print(f"test_images.tar: {len(test_members)} images")

    copy_support_files(repo_root, out_dir, [train_path, test_path])
    print("copied manifests + configs/label_mapping.yaml")

    if not args.skip_wheel:
        wheel_path = build_wheel(repo_root, out_dir)
        print(f"wheel: {wheel_path.name}")

    print(f"Packaged to {out_dir}")

    if args.upload:
        upload(out_dir, args.bucket, args.profile)
        print(f"Uploaded to s3://{args.bucket}/phase3/")


if __name__ == "__main__":
    main()
