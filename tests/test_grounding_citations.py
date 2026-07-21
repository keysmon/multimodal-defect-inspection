"""grounding.citations: the shared citation-validity logic (agent + report)."""
from defectlens.grounding.citations import (
    citation_is_class_relevant,
    on_class_citations,
    validate_citations,
)

CITES = [
    {"card_id": "crack-01", "title": "Crack card", "class_tags": ["crack"]},
    {"card_id": "spall-02", "title": "Spalling card", "class_tags": ["spalling"]},
    {"card_id": "multi-03", "title": "Multi card", "class_tags": ["crack", "spalling"]},
]


def test_on_class_keeps_only_matching_tags():
    kept = on_class_citations(CITES, "crack")
    assert [c["card_id"] for c in kept] == ["crack-01", "multi-03"]


def test_on_class_missing_tags_key_drops():
    assert on_class_citations([{"card_id": "x"}], "crack") == []


def test_class_relevance_predicate():
    tags = {"crack-01": ["crack"], "spall-02": ["spalling"]}
    assert citation_is_class_relevant("crack-01", "crack", tags)
    assert not citation_is_class_relevant("spall-02", "crack", tags)
    assert not citation_is_class_relevant("ghost-99", "crack", tags)


def test_validate_keeps_grounded_and_records_stripped_invalid_ids():
    claims = [{"text": "crack near sill", "citations": ["crack-01", "ghost-99"], "photo_id": "photo_1"}]
    kept, flagged = validate_citations(claims, {"crack-01", "spall-02"})
    assert kept == [{"text": "crack near sill", "citations": ["crack-01"], "photo_id": "photo_1"}]
    # the hallucinated id is recorded, not silently stripped
    assert flagged == [
        {
            "text": "crack near sill",
            "citations": ["ghost-99"],
            "photo_id": "photo_1",
            "reason": "invalid_citation_stripped",
        }
    ]


def test_validate_all_valid_flags_nothing():
    kept, flagged = validate_citations(
        [{"text": "ok", "citations": ["crack-01"]}], {"crack-01"}
    )
    assert kept == [{"text": "ok", "citations": ["crack-01"]}] and flagged == []


def test_validate_drops_ungrounded_to_flagged_with_original_citations():
    claims = [{"text": "invented advice", "citations": ["ghost-99"]}]
    kept, flagged = validate_citations(claims, {"crack-01"})
    assert kept == []
    assert flagged == [
        {"text": "invented advice", "citations": ["ghost-99"], "reason": "no_valid_citation"}
    ]


def test_validate_empty_citations_is_ungrounded():
    kept, flagged = validate_citations([{"text": "claim", "citations": []}], {"crack-01"})
    assert kept == [] and flagged[0]["reason"] == "no_valid_citation"
