"""The walkthrough synthesizer: retrieval fan-out, one multi-image call, gate."""
import io
import json
from dataclasses import dataclass, field

import pytest
from PIL import Image

from defectlens.agent.providers import MockProvider
from defectlens.report.synthesize import (
    MAX_CONCERNS,
    MAX_PHOTOS,
    MAX_VISIT_NOTE_CHARS,
    NOT_OBSERVED_ANSWER,
    NOT_OBSERVED_PHOTO,
    run_walkthrough,
)


@dataclass(frozen=True)
class FakeCard:
    id: str
    title: str = "t"
    class_tags: tuple = ("crack",)
    passage: str = "guidance passage"


@dataclass(frozen=True)
class FakeHit:
    card: FakeCard


@dataclass
class FakeResult:
    hits: list


class FakeRecognizer:
    """Photo retrieval returns crack-01; text retrieval returns damp-02."""

    def analyze_image_bytes(self, data, k=5, note=None):
        return FakeResult(hits=[FakeHit(FakeCard("crack-01"))])

    def search_text(self, query, k=5):
        return [FakeHit(FakeCard("damp-02", class_tags=("water_damage",)))]


@dataclass
class SequencedRecognizer:
    """Distinct cards per photo call, to exercise per-photo citation scoping."""

    per_photo_cards: list
    calls: list = field(default_factory=list)

    def analyze_image_bytes(self, data, k=5, note=None):
        cards = self.per_photo_cards[len(self.calls)]
        self.calls.append(note)
        return FakeResult(hits=[FakeHit(c) for c in cards])

    def search_text(self, query, k=5):
        return []


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, format="PNG")
    return buf.getvalue()


def _photos(n=2):
    return [
        {"photo_id": f"photo_{i + 1}", "image_bytes": _png_bytes(), "note": None}
        for i in range(n)
    ]


def _synthesis(
    per_photo=None,
    action_items=None,
    answers=None,
    overall="Overall fine.",
    overall_citations=("crack-01",),
):
    return json.dumps(
        {
            "per_photo": per_photo if per_photo is not None else [],
            "summary": {
                "overall_assessment": {
                    "text": overall,
                    "citations": list(overall_citations),
                },
                "action_items": action_items if action_items is not None else [],
                "answers": answers if answers is not None else [],
            },
        }
    )


def test_happy_path_grounded_report():
    provider = MockProvider(
        responses=[
            '["is the crack active?"]',
            _synthesis(
                per_photo=[
                    {"photo_id": "photo_1", "observation": "hairline crack at sill", "cited": ["crack-01"]},
                    {"photo_id": "photo_2", "observation": "no visible defect", "no_evidence": True},
                ],
                action_items=[
                    {"priority": "high", "text": "seal the crack", "citations": ["crack-01"], "photo_refs": ["photo_1"]}
                ],
                answers=[
                    {"concern": "is the crack active?", "answer": "monitor width", "citations": ["crack-01"]}
                ],
            ),
        ]
    )
    report = run_walkthrough(
        photos=_photos(2), visit_note="crack?", recognizer=FakeRecognizer(), provider=provider
    )
    assert report.concerns == ["is the crack active?"]
    assert [f.photo_id for f in report.per_photo] == ["photo_1", "photo_2"]
    assert report.per_photo[0].cited == ["crack-01"]
    assert report.per_photo[1].no_evidence is True
    assert report.summary.action_items[0].citations == ["crack-01"]
    assert report.summary.answers[0].citations == ["crack-01"]
    assert report.summary.overall_assessment == "Overall fine."
    assert report.summary.assessment_citations == ["crack-01"]
    assert report.flagged_claims == []
    assert report.disclaimer == "Initial diagnostic - verify before acting."
    # the synthesis call carried BOTH photos (cross-photo reasoning)
    assert provider.calls[-1].n_images == 2


def test_ungrounded_observation_dropped_to_flagged_and_replaced():
    provider = MockProvider(
        responses=[
            _synthesis(
                per_photo=[
                    {"photo_id": "photo_1", "observation": "invented spalling", "cited": ["ghost-99"]}
                ]
            )
        ]
    )
    report = run_walkthrough(
        photos=_photos(1), visit_note=None, recognizer=FakeRecognizer(), provider=provider
    )
    f = report.per_photo[0]
    assert f.no_evidence is True and f.observation == NOT_OBSERVED_PHOTO
    reasons = {c["reason"] for c in report.flagged_claims}
    assert "no_valid_citation" in reasons


def test_photo_cannot_cite_another_photos_card():
    """Per-photo scoping: photo_2 citing a card retrieved only for photo_1 is
    ungrounded for photo_2, even though the id exists in the walkthrough."""
    rec = SequencedRecognizer(
        per_photo_cards=[[FakeCard("crack-01")], [FakeCard("spall-03")]]
    )
    provider = MockProvider(
        responses=[
            _synthesis(
                per_photo=[
                    {"photo_id": "photo_1", "observation": "crack", "cited": ["crack-01"]},
                    {"photo_id": "photo_2", "observation": "borrowed card", "cited": ["crack-01"]},
                ],
                overall_citations=("crack-01",),
            )
        ]
    )
    report = run_walkthrough(
        photos=_photos(2), visit_note=None, recognizer=rec, provider=provider
    )
    assert report.per_photo[0].cited == ["crack-01"]
    assert report.per_photo[1].no_evidence is True
    assert any(
        c.get("photo_id") == "photo_2" and c["reason"] == "no_valid_citation"
        for c in report.flagged_claims
    )


def test_uncited_assessment_falls_back_and_is_flagged():
    provider = MockProvider(
        responses=[
            _synthesis(
                per_photo=[{"photo_id": "photo_1", "observation": "x", "cited": ["crack-01"]}],
                overall="Everything is fine, no action needed.",
                overall_citations=(),
            )
        ]
    )
    report = run_walkthrough(
        photos=_photos(1), visit_note=None, recognizer=FakeRecognizer(), provider=provider
    )
    assert "guidance cards" in report.summary.overall_assessment  # deterministic fallback
    assert report.summary.assessment_citations == []
    assert any(
        c.get("field") == "overall_assessment" and c["reason"] == "no_valid_citation"
        for c in report.flagged_claims
    )


def test_bare_string_assessment_treated_as_uncited():
    raw = json.dumps(
        {
            "per_photo": [{"photo_id": "photo_1", "observation": "x", "cited": ["crack-01"]}],
            "summary": {
                "overall_assessment": "legacy string narrative",
                "action_items": [],
                "answers": [],
            },
        }
    )
    provider = MockProvider(responses=[raw])
    report = run_walkthrough(
        photos=_photos(1), visit_note=None, recognizer=FakeRecognizer(), provider=provider
    )
    assert report.summary.assessment_citations == []
    assert "legacy string narrative" not in report.summary.overall_assessment
    assert any(c.get("field") == "overall_assessment" for c in report.flagged_claims)


def test_missing_photo_finding_synthesized_and_flagged():
    provider = MockProvider(responses=[_synthesis(per_photo=[])])
    report = run_walkthrough(
        photos=_photos(1), visit_note=None, recognizer=FakeRecognizer(), provider=provider
    )
    assert report.per_photo[0].no_evidence is True
    assert any(c["reason"] == "missing_photo_finding" for c in report.flagged_claims)


def test_unanswered_concern_becomes_not_observed_and_flagged():
    provider = MockProvider(responses=['["concern A", "concern B"]', _synthesis(
        per_photo=[{"photo_id": "photo_1", "observation": "x", "cited": ["crack-01"]}],
        answers=[{"concern": "concern A", "answer": "ok", "citations": ["crack-01"]}],
    )])
    report = run_walkthrough(
        photos=_photos(1), visit_note="two concerns", recognizer=FakeRecognizer(), provider=provider
    )
    answers = {a.concern: a for a in report.summary.answers}
    assert set(answers) == {"concern A", "concern B"}
    assert answers["concern B"].not_observed is True
    assert answers["concern B"].answer == NOT_OBSERVED_ANSWER
    assert any(c["reason"] == "missing_answer" for c in report.flagged_claims)


def test_ungrounded_answer_flips_to_not_observed():
    provider = MockProvider(responses=['["concern A"]', _synthesis(
        per_photo=[{"photo_id": "photo_1", "observation": "x", "cited": ["crack-01"]}],
        answers=[{"concern": "concern A", "answer": "made-up advice", "citations": []}],
    )])
    report = run_walkthrough(
        photos=_photos(1), visit_note="n", recognizer=FakeRecognizer(), provider=provider
    )
    a = report.summary.answers[0]
    assert a.not_observed is True and a.citations == []
    assert any(
        c["reason"] == "no_valid_citation" and c["text"] == "made-up advice"
        for c in report.flagged_claims
    )


def test_action_item_with_unknown_card_dropped():
    provider = MockProvider(responses=[_synthesis(
        per_photo=[{"photo_id": "photo_1", "observation": "x", "cited": ["crack-01"]}],
        action_items=[{"priority": "high", "text": "invented fix", "citations": ["ghost-99"]}],
    )])
    report = run_walkthrough(
        photos=_photos(1), visit_note=None, recognizer=FakeRecognizer(), provider=provider
    )
    assert report.summary.action_items == []
    assert any(c["text"] == "invented fix" for c in report.flagged_claims)


def test_concern_overflow_is_flagged_not_silent():
    many = json.dumps([f"concern {i}" for i in range(MAX_CONCERNS + 3)])
    provider = MockProvider(responses=[many, _synthesis(
        per_photo=[{"photo_id": "photo_1", "observation": "x", "cited": ["crack-01"]}],
    )])
    report = run_walkthrough(
        photos=_photos(1), visit_note="a long list", recognizer=FakeRecognizer(), provider=provider
    )
    assert len(report.concerns) == MAX_CONCERNS
    overflow = [c for c in report.flagged_claims if c["reason"] == "concern_overflow"]
    assert [c["concern"] for c in overflow] == [
        f"concern {i}" for i in range(MAX_CONCERNS, MAX_CONCERNS + 3)
    ]


def test_visit_and_photo_notes_are_length_capped():
    provider = MockProvider(responses=["not json", _synthesis(
        per_photo=[{"photo_id": "photo_1", "observation": "x", "cited": ["crack-01"]}],
    )])
    rec = SequencedRecognizer(per_photo_cards=[[FakeCard("crack-01")]])
    photos = [{"photo_id": "photo_1", "image_bytes": _png_bytes(), "note": "n" * 5000}]
    report = run_walkthrough(
        photos=photos, visit_note="v" * (MAX_VISIT_NOTE_CHARS + 500),
        recognizer=rec, provider=provider,
    )
    # concern extraction fell back to the whole note - already truncated
    assert report.concerns == ["v" * MAX_VISIT_NOTE_CHARS]
    # the photo note reached retrieval truncated to the /analyze note cap (500)
    assert rec.calls == ["n" * 500]


def test_synthesis_parse_failure_retries_once_then_raises():
    provider = MockProvider(responses=["garbage", "also garbage"])
    with pytest.raises(ValueError):
        run_walkthrough(
            photos=_photos(1), visit_note=None, recognizer=FakeRecognizer(), provider=provider
        )
    assert len(provider.calls) == 2


def test_photo_cap_enforced():
    with pytest.raises(ValueError):
        run_walkthrough(
            photos=_photos(MAX_PHOTOS + 1),
            visit_note=None,
            recognizer=FakeRecognizer(),
            provider=MockProvider(responses=[]),
        )


def test_empty_photos_rejected():
    with pytest.raises(ValueError):
        run_walkthrough(
            photos=[], visit_note=None, recognizer=FakeRecognizer(), provider=MockProvider(responses=[])
        )


def test_concern_retrieval_cards_are_citable_by_answers():
    """A card retrieved for a CONCERN (not any photo) is a valid citation."""
    provider = MockProvider(responses=['["damp smell"]', _synthesis(
        per_photo=[{"photo_id": "photo_1", "observation": "stain", "cited": ["crack-01"]}],
        answers=[{"concern": "damp smell", "answer": "check drainage", "citations": ["damp-02"]}],
    )])
    report = run_walkthrough(
        photos=_photos(1), visit_note="damp smell", recognizer=FakeRecognizer(), provider=provider
    )
    assert report.summary.answers[0].citations == ["damp-02"]


def test_concern_cards_citable_by_photo_observations():
    """Concern-retrieved cards are visit-level context: valid on any photo."""
    provider = MockProvider(responses=['["damp smell"]', _synthesis(
        per_photo=[{"photo_id": "photo_1", "observation": "stain", "cited": ["damp-02"]}],
        overall_citations=("damp-02",),
    )])
    report = run_walkthrough(
        photos=_photos(1), visit_note="damp smell", recognizer=FakeRecognizer(), provider=provider
    )
    assert report.per_photo[0].cited == ["damp-02"]
