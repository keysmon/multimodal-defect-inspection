import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from verify_raw import verify  # noqa: E402


def real_file(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"fake")


def make_ok_tree(raw: Path):
    real_file(raw / "codebrim" / "crack" / "a.jpg")
    real_file(raw / "bd3" / "algae" / "b.jpg")
    real_file(raw / "sdnet2018" / "cracked" / "c.jpg")
    # Taxonomy v2 (2026-07-21) required datasets:
    real_file(raw / "mbdd2025" / "crack" / "d.jpg")
    real_file(raw / "vt_corrosion" / "poor" / "e.jpg")
    real_file(raw / "insulator" / "broken" / "f.jpg")


def test_verify_ok(tmp_path):
    raw = tmp_path / "raw"
    make_ok_tree(raw)
    assert verify(raw) == 0


def test_verify_missing_required(tmp_path):
    raw = tmp_path / "raw"
    real_file(raw / "bd3" / "algae" / "b.jpg")  # codebrim omitted
    real_file(raw / "sdnet2018" / "cracked" / "c.jpg")
    assert verify(raw) == 1


def test_verify_broken_symlink(tmp_path):
    raw = tmp_path / "raw"
    make_ok_tree(raw)
    (raw / "bd3" / "algae" / "gone.jpg").symlink_to(tmp_path / "no_such_file.jpg")
    assert verify(raw) == 1


def test_verify_double_labeled(tmp_path):
    raw = tmp_path / "raw"
    make_ok_tree(raw)
    shared = tmp_path / "shared.jpg"
    shared.write_bytes(b"fake")
    (raw / "bd3" / "algae" / "dup1.jpg").symlink_to(shared)
    (raw / "bd3" / "stain").mkdir(parents=True)
    (raw / "bd3" / "stain" / "dup2.jpg").symlink_to(shared)
    assert verify(raw) == 1
