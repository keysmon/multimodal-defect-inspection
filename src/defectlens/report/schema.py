"""Walkthrough diagnostic report schema (design 2026-07-21).

The honesty rules are enforced by types, mirroring agent/schema.py's
two-tier rule: a grounded claim MUST cite retrieved cards; an explicit
no-evidence claim MUST NOT pretend to. Raw LLM output is parsed to a plain
dict first (parse_synthesis_json) because the model may violate these rules;
the citation gate in report.synthesize normalizes, then these models
validate the final report - so an ungrounded claim can never ship.
"""
from __future__ import annotations

import json
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

from defectlens.llm_json import balanced_json_candidates

DISCLAIMER = "Initial diagnostic - verify before acting."


class Enrichment(BaseModel):
    """P4 fine-tuned-Qwen structural label, merged only when consistent."""

    label: str = Field(min_length=1)
    confidence: float
    consistent: bool


class PhotoFinding(BaseModel):
    photo_id: str = Field(min_length=1)
    observation: str = Field(min_length=1)
    cited: list[str] = Field(default_factory=list)
    no_evidence: bool = False
    enrichment: Optional[Enrichment] = None

    @model_validator(mode="after")
    def _grounded_xor_no_evidence(self) -> "PhotoFinding":
        if self.no_evidence and self.cited:
            raise ValueError("a no-evidence finding cannot carry citations")
        if not self.no_evidence and not self.cited:
            raise ValueError("a grounded observation needs at least one citation")
        return self


class ActionItem(BaseModel):
    priority: Literal["high", "medium", "low"]
    text: str = Field(min_length=1)
    citations: list[str] = Field(min_length=1)
    photo_refs: list[str] = Field(default_factory=list)


class ConcernAnswer(BaseModel):
    concern: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    citations: list[str] = Field(default_factory=list)
    not_observed: bool = False

    @model_validator(mode="after")
    def _grounded_xor_not_observed(self) -> "ConcernAnswer":
        if self.not_observed and self.citations:
            raise ValueError("a not-observed answer cannot carry citations")
        if not self.not_observed and not self.citations:
            raise ValueError("a grounded answer needs at least one citation")
        return self


class WalkthroughSummary(BaseModel):
    overall_assessment: str = Field(min_length=1)
    # Citations backing the LLM-written assessment narrative. Empty ONLY when
    # the assessment is the deterministic fallback derived from already-gated
    # content - an uncited LLM narrative never ships (gate rule, C+ design).
    assessment_citations: list[str] = Field(default_factory=list)
    action_items: list[ActionItem]
    answers: list[ConcernAnswer]


class WalkthroughReport(BaseModel):
    concerns: list[str]
    per_photo: list[PhotoFinding]
    summary: WalkthroughSummary
    disclaimer: str = DISCLAIMER
    flagged_claims: list[dict] = Field(default_factory=list)
    # Metadata (title/passage/citation/source) for every card the report
    # cites, keyed by card_id - so the UI and the markdown export can render
    # citations without a second corpus lookup. Only CITED ids appear here.
    cards: dict[str, dict] = Field(default_factory=dict)


def parse_synthesis_json(raw: str) -> dict:
    """Parse the synthesis LLM response into a plain dict (pre-gate).

    Bare JSON first, then brace-balanced candidates last-to-first (models
    emit the real object after prose/fences/examples). Raises ValueError
    when nothing parses - callers retry once, then fail the walkthrough.
    """
    try:
        data = json.loads(raw.strip())
    except json.JSONDecodeError:
        pass
    else:
        if isinstance(data, dict):
            return data
    for candidate in reversed(balanced_json_candidates(raw)):
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    raise ValueError(f"no parseable JSON object in synthesis response: {raw[:120]!r}")
