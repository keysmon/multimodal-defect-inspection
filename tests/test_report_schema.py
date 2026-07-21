"""Walkthrough report schema: honesty rules enforced by types."""
import pytest
from pydantic import ValidationError

from defectlens.report.schema import (
    DISCLAIMER,
    ActionItem,
    ConcernAnswer,
    PhotoFinding,
    WalkthroughReport,
    WalkthroughSummary,
    parse_synthesis_json,
)


def _report(**overrides):
    base = dict(
        concerns=["is the crack active?"],
        per_photo=[
            PhotoFinding(photo_id="photo_1", observation="hairline crack", cited=["crack-01"])
        ],
        summary=WalkthroughSummary(
            overall_assessment="One crack observed.",
            action_items=[
                ActionItem(priority="high", text="Monitor the crack", citations=["crack-01"], photo_refs=["photo_1"])
            ],
            answers=[
                ConcernAnswer(concern="is the crack active?", answer="Monitor it", citations=["crack-01"])
            ],
        ),
    )
    base.update(overrides)
    return WalkthroughReport(**base)


def test_valid_report_defaults():
    r = _report()
    assert r.disclaimer == DISCLAIMER
    assert r.flagged_claims == []


def test_grounded_photo_finding_requires_citation():
    with pytest.raises(ValidationError):
        PhotoFinding(photo_id="p", observation="crack", cited=[])


def test_no_evidence_finding_forbids_citations():
    with pytest.raises(ValidationError):
        PhotoFinding(photo_id="p", observation="n/a", cited=["crack-01"], no_evidence=True)
    ok = PhotoFinding(photo_id="p", observation="not observed", no_evidence=True)
    assert ok.cited == []


def test_action_item_requires_citation_and_priority():
    with pytest.raises(ValidationError):
        ActionItem(priority="high", text="do it", citations=[])
    with pytest.raises(ValidationError):
        ActionItem(priority="urgent", text="do it", citations=["c"])


def test_not_observed_answer_forbids_citations():
    with pytest.raises(ValidationError):
        ConcernAnswer(concern="c", answer="a", citations=["x"], not_observed=True)
    ok = ConcernAnswer(concern="c", answer="not observed - verify on-site", not_observed=True)
    assert ok.citations == []


def test_grounded_answer_requires_citations():
    with pytest.raises(ValidationError):
        ConcernAnswer(concern="c", answer="a", citations=[])


def test_parse_synthesis_json_bare_and_prosewrapped():
    assert parse_synthesis_json('{"a": 1}') == {"a": 1}
    assert parse_synthesis_json('Sure!\n```json\n{"a": 1}\n```\ndone') == {"a": 1}


def test_parse_synthesis_json_picks_last_balanced_object():
    raw = 'example: {"schema": true} and the answer {"a": 2}'
    assert parse_synthesis_json(raw) == {"a": 2}


def test_parse_synthesis_json_raises_on_garbage():
    with pytest.raises(ValueError):
        parse_synthesis_json("no json at all")
