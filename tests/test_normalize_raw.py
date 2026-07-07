import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from normalize_raw import (  # noqa: E402
    DATASET_LABELS,
    canon,
    match_label,
    normalize_generic,
    normalize_sdnet,
)


def touch(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"fake")


def test_canon():
    assert canon("Major Crack") == "majorcrack"
    assert canon("peeling-paint") == "peelingpaint"
    assert canon("Exposed_Bars") == "exposedbars"


def test_match_label():
    labels = DATASET_LABELS["bd3"]
    assert match_label("Major Crack", labels) == "major_crack"
    assert match_label("ALGAE", labels) == "algae"
    assert match_label("random_junk", labels) is None


def test_normalize_generic(tmp_path):
    src = tmp_path / "bd3_clone"
    touch(src / "dataset" / "Major Crack" / "img1.jpg")
    touch(src / "dataset" / "Algae" / "img2.jpg")
    touch(src / "dataset" / "Algae" / "notes.txt")  # non-image ignored
    dest = tmp_path / "raw" / "bd3"
    n = normalize_generic(src, dest, DATASET_LABELS["bd3"])
    assert n == 2
    assert (dest / "major_crack").is_dir()
    linked = list((dest / "algae").iterdir())
    assert len(linked) == 1
    assert linked[0].is_symlink()


def test_normalize_sdnet(tmp_path):
    src = tmp_path / "SDNET2018"
    touch(src / "D" / "CD" / "c1.jpg")
    touch(src / "D" / "UD" / "u1.jpg")
    touch(src / "W" / "CW" / "c2.jpg")
    dest = tmp_path / "raw" / "sdnet2018"
    n = normalize_sdnet(src, dest)
    assert n == 3
    assert len(list((dest / "cracked").iterdir())) == 2
    assert len(list((dest / "non_cracked").iterdir())) == 1


def test_normalize_generic_no_collisions(tmp_path):
    src = tmp_path / "clone"
    touch(src / "train" / "Crack" / "img.jpg")
    touch(src / "test" / "Crack" / "img.jpg")  # same filename, different split dir
    dest = tmp_path / "raw" / "roboflow_walls"
    n = normalize_generic(src, dest, DATASET_LABELS["roboflow_walls"])
    assert n == 2
    assert len(list((dest / "crack").iterdir())) == 2
