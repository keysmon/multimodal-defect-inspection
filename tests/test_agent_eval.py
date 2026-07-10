import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from eval_agent import property_metrics, regression_check  # noqa: E402


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
