"""Walkthrough eval metrics + the shared regression gate."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from eval_walkthrough import aggregate_metrics, report_metrics, write_spotcheck_template  # noqa: E402

from defectlens.eval.gate import finalize_run, regression_check  # noqa: E402


def _report(flagged=None, answers=None, per_photo=None, action_items=None, concerns=None):
    return {
        "concerns": concerns if concerns is not None else ["c1", "c2"],
        "per_photo": per_photo
        if per_photo is not None
        else [
            {"photo_id": "photo_1", "observation": "crack", "cited": ["crack-01"], "no_evidence": False},
            {"photo_id": "photo_2", "observation": "n/a", "cited": [], "no_evidence": True},
        ],
        "summary": {
            "overall_assessment": "ok",
            "assessment_citations": ["crack-01"],
            "action_items": action_items
            if action_items is not None
            else [{"priority": "high", "text": "t", "citations": ["crack-01"], "photo_refs": []}],
            "answers": answers
            if answers is not None
            else [
                {"concern": "c1", "answer": "a", "citations": ["crack-01"], "not_observed": False},
                {"concern": "c2", "answer": "not observed", "citations": [], "not_observed": True},
            ],
        },
        "flagged_claims": flagged if flagged is not None else [],
    }


def test_clean_report_metrics():
    m = report_metrics(_report())
    assert m["groundedness"] == 1.0
    assert m["raw_groundedness"] == 1.0
    assert m["coverage"] == 1.0
    assert m["answered_with_evidence_rate"] == 0.5
    assert m["flagged_rate"] == 0.0


def test_dropped_claims_lower_raw_groundedness_not_post_gate():
    m = report_metrics(_report(flagged=[{"text": "x", "reason": "no_valid_citation"}]))
    assert m["groundedness"] == 1.0
    # 4 kept claims (photo_1 + action + c1 answer + cited assessment) vs 1 dropped
    assert m["raw_groundedness"] == 0.8
    assert m["flagged_rate"] == 0.2


def test_missing_answer_flag_lowers_coverage():
    m = report_metrics(_report(flagged=[{"concern": "c2", "reason": "missing_answer"}]))
    assert m["coverage"] == 0.5


def test_no_concerns_coverage_is_one():
    m = report_metrics(_report(concerns=[], answers=[]))
    assert m["coverage"] == 1.0


def test_aggregate_means_and_valid_rate():
    per = {
        "w1": {"groundedness": 1.0, "raw_groundedness": 0.8, "coverage": 1.0,
               "answered_with_evidence_rate": 0.5, "flagged_rate": 0.2},
        "w2": {"error": "Boom"},
    }
    agg = aggregate_metrics(per)
    assert agg["raw_groundedness"] == 0.8
    assert agg["schema_valid_rate"] == 0.5


def test_regression_gate_flags_drop_beyond_tolerance():
    prev = {"raw_groundedness": 0.9, "coverage": 1.0}
    curr = {"raw_groundedness": 0.85, "coverage": 1.0}
    assert regression_check(prev, curr, gated=("raw_groundedness", "coverage")) == ["raw_groundedness"]
    assert regression_check(prev, {"raw_groundedness": 0.89, "coverage": 1.0},
                            gated=("raw_groundedness", "coverage")) == []


def test_finalize_run_rejects_regression(tmp_path):
    results = tmp_path / "eval.json"
    rejected = tmp_path / "eval.rejected.json"
    ok = finalize_run(
        {"metrics": {"coverage": 1.0}}, None,
        results_path=results, rejected_path=rejected,
        gated=("coverage",), tolerance=0.02,
    )
    assert ok == 0 and results.exists()
    bad = finalize_run(
        {"metrics": {"coverage": 0.5}}, {"coverage": 1.0},
        results_path=results, rejected_path=rejected,
        gated=("coverage",), tolerance=0.02,
    )
    assert bad == 1 and rejected.exists()
    assert json.loads(results.read_text())["metrics"]["coverage"] == 1.0


def test_spotcheck_template_lists_observations(tmp_path):
    out = tmp_path / "spot.md"
    write_spotcheck_template({"walk_01": _report()}, out)
    text = out.read_text()
    assert "walk_01" in text and "photo_1" in text and "- [ ]" in text
    assert "NOT auto-measured" in text
