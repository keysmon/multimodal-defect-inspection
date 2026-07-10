import pytest
from pydantic import ValidationError

from defectlens.agent.schema import Citation, Finding, InspectionReport, parse_report_json


def _finding(**overrides):
    base = dict(
        finding="crack",
        tier="measured",
        defect_class="crack",
        severity="structural",
        evidence_photo="photos/img_003.jpg",
        citations=[Citation(card_id="epa-001", title="Crack guidance")],
        notes="",
    )
    base.update(overrides)
    return Finding(**base)


def test_finding_roundtrip():
    f = _finding()
    assert f.tier == "measured" and f.citations[0].card_id == "epa-001"


def test_tier_is_constrained():
    with pytest.raises(ValidationError):
        _finding(tier="guess")


def test_measured_requires_known_class():
    with pytest.raises(ValidationError):
        _finding(defect_class="sagging_gutter")  # not in UNIFIED_CLASSES


def test_observation_allows_free_text_class():
    f = _finding(tier="observation", defect_class=None, finding="corroded TPR valve")
    assert f.defect_class is None


def test_report_requires_at_least_summary():
    with pytest.raises(ValidationError):
        InspectionReport(property_id="p1", findings=[], summary="")


def test_parse_report_json_extracts_fenced_json():
    raw = 'preamble\n```json\n{"property_id": "p1", "findings": [], "summary": "ok"}\n```'
    report = parse_report_json(raw)
    assert report.property_id == "p1"


def test_parse_report_json_raises_on_garbage():
    with pytest.raises(ValueError):
        parse_report_json("not json at all")
