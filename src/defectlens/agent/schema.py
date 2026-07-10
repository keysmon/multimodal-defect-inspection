"""Inspection-report schema: the contract between workflow, LLM, and eval.

Measured findings must use the trained vocabulary; observations are
free-text. This split is the two-tier honesty rule, enforced by types.
"""
from __future__ import annotations

import json
import re
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

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


_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)


def parse_report_json(raw: str) -> InspectionReport:
    """Parse an LLM response into a validated report.

    Accepts bare JSON or a fenced ```json block (small local models fence
    reliably but add prose around it).
    """
    m = _FENCE.search(raw)
    candidate = m.group(1) if m else raw.strip()
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(f"no parseable JSON report in response: {exc}") from exc
    return InspectionReport.model_validate(data)
