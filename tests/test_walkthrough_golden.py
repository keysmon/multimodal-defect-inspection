"""The frozen walkthrough golden set: shape + referential integrity."""
import json
from pathlib import Path

GOLDEN = Path("data/manifests/walkthrough_golden.json")


def test_golden_shape_and_uniqueness():
    data = json.loads(GOLDEN.read_text())
    walks = data["walkthroughs"]
    assert len(walks) >= 6
    ids = [w["walkthrough_id"] for w in walks]
    assert len(ids) == len(set(ids))
    for w in walks:
        assert w["visit_note"].strip()
        assert 1 <= len(w["photos"]) <= 10
        pids = [p["photo_id"] for p in w["photos"]]
        assert len(pids) == len(set(pids))
        for p in w["photos"]:
            assert p["image_path"].startswith("data/raw/")


def test_golden_paths_exist_when_raw_data_present():
    data = json.loads(GOLDEN.read_text())
    all_paths = [
        Path(p["image_path"]) for w in data["walkthroughs"] for p in w["photos"]
    ]
    present = [p for p in all_paths if p.exists()]
    # On a machine with data/raw checked out, EVERY path must resolve;
    # elsewhere (CI without datasets) the test degrades to shape-only.
    if present:
        missing = [str(p) for p in all_paths if not p.exists()]
        assert missing == []
