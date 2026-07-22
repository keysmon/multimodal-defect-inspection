"""Pure-logic locks for the B4 gate-floor derivation + VT severity metric
(script modules loaded by path, like the other scripts/ tests)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


gate = _load("derive_gate_floor")
severity = _load("eval_corrosion_severity")


def _rec(true, top, conf, dataset="d", state="x"):
    other = "crack" if top != "crack" else "spalling"
    return {
        "image_path": "p.jpg",
        "source_dataset": dataset,
        "source_label": state,
        "true": true,
        "ranked": [top, other],
        "probs": {top: conf, other: round(1 - conf, 6)},
    }


def test_floor_curve_counts_kept_and_incorrect():
    records = [
        _rec("crack", "crack", 0.9),     # correct, high conf
        _rec("crack", "crack", 0.6),     # correct, mid conf
        _rec("crack", "spalling", 0.7),  # incorrect, mid conf
        _rec("crack", "spalling", 0.2),  # incorrect, low conf
    ]
    curve = gate.floor_curve(records, [0.0, 0.65, 0.95])
    by_floor = {row["floor"]: row for row in curve}
    assert by_floor[0.0] == {
        "floor": 0.0, "merged": 4, "kept_correct_frac": 1.0, "merged_incorrect_frac": 0.5,
    }
    # 0.65 keeps the 0.9 correct and 0.7 incorrect
    assert by_floor[0.65]["merged"] == 2
    assert by_floor[0.65]["kept_correct_frac"] == 0.5
    assert by_floor[0.65]["merged_incorrect_frac"] == 0.5
    # 0.95 keeps nothing
    assert by_floor[0.95]["merged"] == 0
    assert by_floor[0.95]["merged_incorrect_frac"] == 0.0


def test_choose_floor_maximizes_kept_correct_under_constraint():
    curve = [
        {"floor": 0.0, "merged": 10, "kept_correct_frac": 1.0, "merged_incorrect_frac": 0.30},
        {"floor": 0.4, "merged": 8, "kept_correct_frac": 0.9, "merged_incorrect_frac": 0.05},
        {"floor": 0.6, "merged": 6, "kept_correct_frac": 0.9, "merged_incorrect_frac": 0.03},
        {"floor": 0.8, "merged": 3, "kept_correct_frac": 0.5, "merged_incorrect_frac": 0.0},
    ]
    chosen = gate.choose_floor(curve, 0.05)
    # 0.4 and 0.6 tie on kept-correct; the LOWER floor wins (more merges).
    assert chosen["floor"] == 0.4


def test_choose_floor_returns_none_when_unsatisfiable():
    curve = [
        {"floor": 0.0, "merged": 4, "kept_correct_frac": 1.0, "merged_incorrect_frac": 0.5},
    ]
    assert gate.choose_floor(curve, 0.05) is None


def test_severity_report_per_state_rates_and_bands():
    records = [
        _rec("corrosion_stain", "corrosion_stain", 0.9, "vt_corrosion", "severe"),
        _rec("corrosion_stain", "crack", 0.6, "vt_corrosion", "severe"),
        _rec("corrosion_stain", "corrosion_stain", 0.8, "vt_corrosion", "fair"),
        _rec("crack", "crack", 0.9, "mbdd2025", "crack"),  # non-VT: ignored
    ]
    report = severity.severity_report(records)
    assert set(report["states"]) == {"fair", "severe"}
    assert report["states"]["severe"] == {
        "n": 2, "top1_corrosion_rate": 0.5, "top3_corrosion_rate": 0.5,
        "band_when_recognized": "structural",
    }
    assert report["states"]["fair"]["top1_corrosion_rate"] == 1.0
    assert report["states"]["fair"]["band_when_recognized"] == "monitor"


def test_severity_report_requires_vt_rows():
    with pytest.raises(SystemExit):
        severity.severity_report([_rec("crack", "crack", 0.9, "mbdd2025")])


def test_per_image_record_probs_sum_to_one_and_rank():
    from defectlens.eval.vlm_topk import per_image_record

    class Row:
        image_path = "x.jpg"
        source_dataset = "vt_corrosion"
        source_label = "poor"
        unified_label = "corrosion_stain"

    rec = per_image_record(Row(), {"crack": -0.5, "corrosion_stain": -0.1, "no_defect": -2.0})
    assert rec["ranked"][0] == "corrosion_stain"
    assert abs(sum(rec["probs"].values()) - 1.0) < 1e-4
    assert rec["probs"]["corrosion_stain"] > rec["probs"]["crack"] > rec["probs"]["no_defect"]
