import sys
import tarfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from defectlens.ingest import ManifestRow, write_manifest  # noqa: E402

from package_data import (  # noqa: E402
    TarMember,
    build_tar,
    copy_support_files,
    load_rows,
    manifest_members,
)

# ---------------------------------------------------------------------------
# manifest_members
# ---------------------------------------------------------------------------


def test_manifest_members_dedup_and_order(tmp_path):
    rows = [
        ManifestRow("data/raw/a/1.jpg", "x", "l", "crack"),
        ManifestRow("data/raw/a/2.jpg", "x", "l", "crack"),
        ManifestRow("data/raw/a/1.jpg", "x", "l", "crack"),  # dup, later in list
    ]
    members = manifest_members(rows, tmp_path)
    assert [m.arcname for m in members] == ["data/raw/a/1.jpg", "data/raw/a/2.jpg"]
    assert members[0].source == tmp_path / "data/raw/a/1.jpg"
    assert members[1].source == tmp_path / "data/raw/a/2.jpg"


def test_manifest_members_empty():
    assert manifest_members([], Path("/repo")) == []


def test_manifest_members_preserves_manifest_order_not_sorted():
    rows = [
        ManifestRow("data/raw/z.jpg", "x", "l", "crack"),
        ManifestRow("data/raw/a.jpg", "x", "l", "crack"),
    ]
    members = manifest_members(rows, Path("/repo"))
    assert [m.arcname for m in members] == ["data/raw/z.jpg", "data/raw/a.jpg"]


# ---------------------------------------------------------------------------
# load_rows (--subset semantics: first N rows per manifest, file order)
# ---------------------------------------------------------------------------


def test_load_rows_subset_takes_first_n_in_file_order(tmp_path):
    manifest = tmp_path / "m.csv"
    write_manifest(
        [ManifestRow(f"data/raw/x/{i}.jpg", "x", "l", "crack") for i in range(10)], manifest
    )
    rows = load_rows(manifest, subset=3)
    assert [r.image_path for r in rows] == [f"data/raw/x/{i}.jpg" for i in range(3)]


def test_load_rows_no_subset_returns_all_rows(tmp_path):
    manifest = tmp_path / "m.csv"
    write_manifest(
        [ManifestRow(f"data/raw/x/{i}.jpg", "x", "l", "crack") for i in range(5)], manifest
    )
    rows = load_rows(manifest, subset=None)
    assert len(rows) == 5


def test_load_rows_subset_larger_than_manifest_returns_all(tmp_path):
    manifest = tmp_path / "m.csv"
    write_manifest([ManifestRow("data/raw/x/0.jpg", "x", "l", "crack")], manifest)
    rows = load_rows(manifest, subset=1000)
    assert len(rows) == 1


# ---------------------------------------------------------------------------
# build_tar
# ---------------------------------------------------------------------------


def test_build_tar_dereferences_symlinks_and_preserves_arcnames(tmp_path):
    # Mirrors the real repo layout: a manifest-relative path that's actually
    # a symlink pointing outside the "repo" (data/raw/*/*.jpg -> ~/datasets/...).
    external = tmp_path / "external_cache"
    external.mkdir()
    real_file = external / "img.jpg"
    real_file.write_bytes(b"fake-image-bytes")

    link_path = tmp_path / "repo" / "data" / "raw" / "x" / "crack" / "img.jpg"
    link_path.parent.mkdir(parents=True)
    link_path.symlink_to(real_file)

    members = [TarMember(arcname="data/raw/x/crack/img.jpg", source=link_path)]
    tar_path = tmp_path / "out" / "train_images.tar"
    build_tar(members, tar_path)

    assert tar_path.is_file()
    with tarfile.open(tar_path) as tar:
        assert tar.getnames() == ["data/raw/x/crack/img.jpg"]
        info = tar.getmember("data/raw/x/crack/img.jpg")
        assert info.isreg()  # dereferenced -> regular file entry, not a symlink
        assert tar.extractfile(info).read() == b"fake-image-bytes"


def test_build_tar_multiple_members_and_missing_dirs_created(tmp_path):
    for name in ("a.jpg", "b.jpg"):
        (tmp_path / name).write_bytes(name.encode())
    members = [
        TarMember(arcname=f"data/raw/{n}", source=tmp_path / n) for n in ("a.jpg", "b.jpg")
    ]
    tar_path = tmp_path / "nested" / "does" / "not" / "exist" / "out.tar"
    build_tar(members, tar_path)
    with tarfile.open(tar_path) as tar:
        assert sorted(tar.getnames()) == ["data/raw/a.jpg", "data/raw/b.jpg"]


def test_build_tar_raises_on_missing_source(tmp_path):
    members = [TarMember(arcname="data/raw/x/1.jpg", source=tmp_path / "nope.jpg")]
    with pytest.raises(FileNotFoundError):
        build_tar(members, tmp_path / "out.tar")


def test_build_tar_raises_on_broken_symlink(tmp_path):
    link_path = tmp_path / "dangling.jpg"
    link_path.symlink_to(tmp_path / "does-not-exist.jpg")
    members = [TarMember(arcname="dangling.jpg", source=link_path)]
    with pytest.raises(FileNotFoundError):
        build_tar(members, tmp_path / "out.tar")


def test_build_tar_empty_members_creates_empty_tar(tmp_path):
    tar_path = tmp_path / "out" / "empty.tar"
    build_tar([], tar_path)
    with tarfile.open(tar_path) as tar:
        assert tar.getnames() == []


# ---------------------------------------------------------------------------
# copy_support_files
# ---------------------------------------------------------------------------


def test_copy_support_files(tmp_path):
    repo = tmp_path / "repo"
    (repo / "configs").mkdir(parents=True)
    (repo / "configs" / "label_mapping.yaml").write_text("mappings: []\n")
    manifests_dir = repo / "data" / "manifests"
    manifests_dir.mkdir(parents=True)
    train = manifests_dir / "train.csv"
    train.write_text("image_path\n")
    test = manifests_dir / "test.csv"
    test.write_text("image_path\n")

    out = tmp_path / "dist"
    copy_support_files(repo, out, [train, test])

    assert (out / "manifests" / "train.csv").read_text() == "image_path\n"
    assert (out / "manifests" / "test.csv").read_text() == "image_path\n"
    assert (out / "configs" / "label_mapping.yaml").read_text() == "mappings: []\n"
