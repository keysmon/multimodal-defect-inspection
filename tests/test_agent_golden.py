import csv
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_agent_golden.py"


def _run(out, extra=()):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--out", str(out), *extra],
        capture_output=True, text=True,
    )


def test_builds_15x5_stratified_manifest(tmp_path):
    out = tmp_path / "golden.csv"
    result = _run(out)
    assert result.returncode == 0, result.stderr
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 75
    by_prop = {}
    for r in rows:
        by_prop.setdefault(r["property_id"], []).append(r["unified_label"])
    assert len(by_prop) == 15
    for labels in by_prop.values():
        assert len(labels) == 5
        defects = {l for l in labels if l != "no_defect"}
        assert 2 <= len(defects) <= 3
        assert labels.count("no_defect") >= 2


def test_deterministic_across_runs(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    _run(a)
    _run(b)
    assert a.read_text() == b.read_text()


def test_refuses_overwrite_without_force(tmp_path):
    out = tmp_path / "golden.csv"
    _run(out)
    result = _run(out)
    assert result.returncode != 0 and "force" in result.stderr.lower()
