import shutil
import sys
from pathlib import Path

import pytest

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


def test_nested_label_dirs_rejected(tmp_path):
    # A label dir nested inside another label dir is ambiguous labeling:
    # _link_images recurses, so the same image would land under BOTH labels.
    src = tmp_path / "codebrim_clone"
    touch(src / "background" / "crack" / "img.jpg")
    dest = tmp_path / "raw" / "codebrim"
    with pytest.raises(SystemExit):
        normalize_generic(src, dest, DATASET_LABELS["codebrim"])


def test_rerun_is_idempotent(tmp_path):
    src = tmp_path / "clone"
    touch(src / "dataset" / "Algae" / "img1.jpg")
    touch(src / "dataset" / "Major Crack" / "img2.jpg")
    dest = tmp_path / "raw" / "bd3"
    assert normalize_generic(src, dest, DATASET_LABELS["bd3"]) == 2
    # Second run links nothing new and leaves counts unchanged.
    assert normalize_generic(src, dest, DATASET_LABELS["bd3"]) == 0
    assert len(list((dest / "algae").iterdir())) == 1
    assert len(list((dest / "major_crack").iterdir())) == 1


def test_dangling_link_relinked(tmp_path):
    src1 = tmp_path / "v1"
    touch(src1 / "dataset" / "Algae" / "img.jpg")
    dest = tmp_path / "raw" / "bd3"
    assert normalize_generic(src1, dest, DATASET_LABELS["bd3"]) == 1
    shutil.rmtree(src1)  # source moved/deleted -> existing links now dangle
    src2 = tmp_path / "v2"  # NEW path, same internal layout
    touch(src2 / "dataset" / "Algae" / "img.jpg")
    n = normalize_generic(src2, dest, DATASET_LABELS["bd3"])  # must not raise
    assert n == 1
    linked = list((dest / "algae").iterdir())
    assert len(linked) == 1
    assert linked[0].exists()  # no longer dangling
    assert linked[0].resolve() == (src2 / "dataset" / "Algae" / "img.jpg").resolve()


def test_sdnet_ignores_non_dpw_parents(tmp_path):
    src = tmp_path / "SDNET2018"
    touch(src / "X" / "CD" / "bad.jpg")  # parent not D/P/W -> must be skipped
    touch(src / "D" / "CD" / "ok.jpg")
    dest = tmp_path / "raw" / "sdnet2018"
    n = normalize_sdnet(src, dest)
    assert n == 1
    linked = list((dest / "cracked").iterdir())
    assert len(linked) == 1
    assert "ok.jpg" in linked[0].name
