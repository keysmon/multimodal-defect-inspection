import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import eval_agent  # noqa: E402
from eval_agent import (  # noqa: E402
    aggregate_metrics,
    finalize_run,
    property_metrics,
    regression_check,
)


def _report(measured, citations_by_finding=None):
    citations_by_finding = citations_by_finding or {}
    return {
        "findings": [
            {
                "tier": "measured",
                "defect_class": c,
                "citations": citations_by_finding.get(c, []),
            }
            for c in measured
        ]
    }


CARD_TAGS = {"epa-001": ["crack"], "hud-002": ["spalling"]}


def test_perfect_property():
    m = property_metrics(
        _report(["crack"], {"crack": [{"card_id": "epa-001"}]}),
        expected={"crack"},
        card_tags=CARD_TAGS,
    )
    assert m["findings_recall"] == 1.0
    assert m["findings_precision"] == 1.0
    assert m["citation_validity"] == 1.0


def test_missed_and_spurious_findings():
    m = property_metrics(
        _report(["spalling"]), expected={"crack", "mold_algae"}, card_tags=CARD_TAGS
    )
    assert m["findings_recall"] == 0.0
    assert m["findings_precision"] == 0.0


def test_invalid_citation_detected():
    m = property_metrics(
        _report(["crack"], {"crack": [{"card_id": "hud-002"}]}),
        expected={"crack"},
        card_tags=CARD_TAGS,
    )
    assert m["citation_validity"] == 0.0


def test_regression_check_flags_drop():
    prev = {"findings_recall": 0.80, "citation_validity": 0.95}
    curr = {"findings_recall": 0.70, "citation_validity": 0.95}
    failed = regression_check(prev, curr, tolerance=0.02)
    assert failed == ["findings_recall"]


def test_regression_check_passes_within_tolerance():
    prev = {"findings_recall": 0.80, "citation_validity": 0.95}
    curr = {"findings_recall": 0.79, "citation_validity": 0.96}
    assert regression_check(prev, curr, tolerance=0.02) == []


def _metrics(recall, precision, validity):
    return {
        "findings_recall": recall,
        "findings_precision": precision,
        "citation_validity": validity,
    }


def test_aggregate_metrics_excludes_failed_properties():
    per_property = {
        "p1": _metrics(1.0, 0.5, 1.0),
        "p2": _metrics(0.0, 1.0, 0.5),
        "p3": {"error": "RuntimeError: boom"},
        "p4": {"error": "ValidationError: bad report"},
    }
    m = aggregate_metrics(per_property)
    # Quality means over the 2 successes only; failures excluded.
    assert m["findings_recall"] == 0.5
    assert m["findings_precision"] == 0.75
    assert m["citation_validity"] == 0.75
    # Reliability over all 4 attempted properties.
    assert m["schema_valid_rate"] == 0.5


def test_aggregate_metrics_all_success_rate_is_one():
    per_property = {"p1": _metrics(1.0, 1.0, 1.0), "p2": _metrics(0.5, 1.0, 1.0)}
    m = aggregate_metrics(per_property)
    assert m["schema_valid_rate"] == 1.0
    assert m["findings_recall"] == 0.75


def test_aggregate_metrics_all_failed_raises():
    with pytest.raises(ValueError):
        aggregate_metrics({"p1": {"error": "RuntimeError: boom"}})


def _payload(recall, validity):
    return {
        "run_config": {"provider": "mock", "n_properties": 1},
        "metrics": {
            "findings_recall": recall,
            "findings_precision": 1.0,
            "citation_validity": validity,
            "schema_valid_rate": 1.0,
        },
        "per_property": {},
    }


@pytest.fixture
def eval_paths(tmp_path, monkeypatch):
    results = tmp_path / "agent_eval.json"
    rejected = tmp_path / "agent_eval.rejected.json"
    monkeypatch.setattr(eval_agent, "RESULTS", results)
    monkeypatch.setattr(eval_agent, "REJECTED", rejected)
    return results, rejected


def test_finalize_run_regression_preserves_baseline(eval_paths):
    results, rejected = eval_paths
    baseline = _payload(0.90, 0.95)
    results.write_text(json.dumps(baseline))
    regressed = _payload(0.50, 0.95)
    rc = finalize_run(regressed, baseline["metrics"], tolerance=0.02)
    assert rc == 1
    assert json.loads(results.read_text()) == baseline
    assert json.loads(rejected.read_text()) == regressed


def test_finalize_run_pass_overwrites_baseline(eval_paths):
    results, rejected = eval_paths
    baseline = _payload(0.90, 0.95)
    results.write_text(json.dumps(baseline))
    improved = _payload(0.95, 0.96)
    rc = finalize_run(improved, baseline["metrics"], tolerance=0.02)
    assert rc == 0
    assert json.loads(results.read_text()) == improved
    assert not rejected.exists()


def test_finalize_run_first_run_writes_baseline(eval_paths):
    results, rejected = eval_paths
    first = _payload(0.80, 0.90)
    rc = finalize_run(first, previous=None, tolerance=0.02)
    assert rc == 0
    assert json.loads(results.read_text()) == first
    assert not rejected.exists()
