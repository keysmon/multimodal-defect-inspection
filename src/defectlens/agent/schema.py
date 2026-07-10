"""Inspection-report schema: the contract between workflow, LLM, and eval.

Measured findings must use the trained vocabulary; observations are
free-text. This split is the two-tier honesty rule, enforced by types.
"""
from __future__ import annotations

import json
from typing import Literal, Optional

from pydantic import BaseModel, Field, ValidationError, model_validator

from defectlens.taxonomy import UNIFIED_CLASSES

SEVERITIES = ("cosmetic", "monitor", "moderate", "structural", "unknown")


class Citation(BaseModel):
    card_id: str = Field(min_length=1)
    title: str = ""


class Finding(BaseModel):
    finding: str = Field(min_length=1)
    tier: Literal["measured", "observation"]
    defect_class: Optional[str] = None
    severity: Literal[*SEVERITIES] = "unknown"
    evidence_photo: str = Field(min_length=1)
    citations: list[Citation] = Field(default_factory=list)
    notes: str = ""

    @model_validator(mode="after")
    def _measured_needs_trained_class(self) -> "Finding":
        if self.tier == "measured":
            if self.defect_class not in UNIFIED_CLASSES:
                raise ValueError(
                    f"measured finding requires a trained class, got {self.defect_class!r}"
                )
        return self


class InspectionReport(BaseModel):
    property_id: str = Field(min_length=1)
    findings: list[Finding]
    summary: str = Field(min_length=1)
    audio_band: Optional[str] = None


def _balanced_json_candidates(raw: str) -> list[str]:
    """Top-level brace-balanced {...} substrings, in order of appearance.

    A minimal in-string flag (with backslash escapes) keeps braces inside
    double-quoted JSON strings from confusing the depth counter. Quotes are
    only tracked inside a candidate; prose quotes outside braces are ignored.
    """
    candidates: list[str] = []
    depth = 0
    start = 0
    in_string = False
    escaped = False
    for i, ch in enumerate(raw):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if depth > 0 and ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0:
                candidates.append(raw[start : i + 1])
    return candidates


def parse_report_json(raw: str) -> InspectionReport:
    """Parse an LLM response into a validated report.

    Accepts bare JSON directly. Otherwise scans for brace-balanced JSON
    objects and validates them last to first, since models tend to emit the
    real report after any prose, fences, or example/schema blocks.
    """
    try:
        data = json.loads(raw.strip())
    except json.JSONDecodeError:
        pass
    else:
        return InspectionReport.model_validate(data)
    for candidate in reversed(_balanced_json_candidates(raw)):
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        try:
            return InspectionReport.model_validate(data)
        except ValidationError:
            continue
    raise ValueError(f"no parseable JSON report in response: {raw[:120]!r}")
