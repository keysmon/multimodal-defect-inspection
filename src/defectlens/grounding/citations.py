"""Citation validity, extracted from the inspection-agent (design 2026-07-21).

Three callers, one rule set:
- on_class_citations: the agent workflow's on-class filter (an off-class
  citation is worse than none; baseline measured citation_validity 0.741
  before this filter, 1.0 after).
- citation_is_class_relevant: the agent eval's validity predicate.
- validate_citations: the walkthrough report's citation gate - every claim
  must cite a real card from the retrieved set; ungrounded claims are
  dropped and recorded (flagged_claims), never silently kept.
"""
from __future__ import annotations


def on_class_citations(citations: list[dict], class_tag: str) -> list[dict]:
    """Keep only citation dicts whose class_tags include class_tag."""
    return [c for c in citations if class_tag in c.get("class_tags", [])]


def citation_is_class_relevant(
    card_id: str, class_tag: str, card_tags: dict[str, list[str]]
) -> bool:
    """True when the cited card exists and carries the claim's class tag."""
    return class_tag in card_tags.get(card_id, [])


def validate_citations(
    claims: list[dict], allowed_ids: set[str]
) -> tuple[list[dict], list[dict]]:
    """The citation gate: claims must cite cards from the retrieved set.

    Each claim: {"text": str, "citations": [card_id], **extra}. Citations not
    in allowed_ids are stripped; a claim left with none is moved to flagged
    (original citations preserved, reason recorded) instead of shipping
    ungrounded. Extra keys pass through untouched on both sides.
    """
    kept: list[dict] = []
    flagged: list[dict] = []
    for claim in claims:
        valid = [c for c in claim.get("citations", []) if c in allowed_ids]
        if valid:
            kept.append({**claim, "citations": valid})
        else:
            flagged.append({**claim, "reason": "no_valid_citation"})
    return kept, flagged
