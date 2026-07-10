"""LLM-orchestrated inspection workflow (v1: structured steps, LLM judgment).

Skeleton is code; the LLM handles open-vocab observation and summary
synthesis. Findings are assembled deterministically so the LLM can neither
invent nor drop them - the two-tier labels stay trustworthy.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from defectlens.agent.providers import LLMProvider, Usage
from defectlens.agent.schema import Citation, Finding, InspectionReport
from defectlens.agent.tools import (
    Trace,
    classify_image,
    observe_image,
    retrieve_guidance,
    score_audio,
)

MEASURED_THRESHOLD = 0.5

SUMMARY_PROMPT = """You are drafting the summary paragraph of a building inspection report.
Findings (JSON): {findings}
Write 2-4 plain sentences summarizing overall condition and priorities.
Respond with the paragraph only."""


def _default_load_image(path):
    from PIL import Image

    return Image.open(path).convert("RGB")


def run_inspection(
    *,
    property_id: str,
    image_paths: list,
    describer,
    recognizer,
    provider: LLMProvider,
    audio_analyzer=None,
    audio_bytes: bytes | None = None,
    out_dir: Path,
    load_image=_default_load_image,
) -> tuple[InspectionReport, Usage, Path]:
    out_dir = Path(out_dir)
    trace = Trace(out_dir / f"trace_{property_id}_{int(time.time())}.jsonl")

    findings: list[Finding] = []
    for path in image_paths:
        image = load_image(path)

        ranking = classify_image(describer, image, trace)
        measured_class = None
        if ranking:
            top_class, top_prob = ranking[0]
            if top_prob >= MEASURED_THRESHOLD and top_class != "no_defect":
                measured_class = top_class
                citations = retrieve_guidance(
                    recognizer, f"{top_class} building defect remediation", trace
                )
                findings.append(
                    Finding(
                        finding=top_class,
                        tier="measured",
                        defect_class=top_class,
                        severity="unknown",
                        evidence_photo=str(path),
                        citations=[Citation(card_id=c["card_id"], title=c["title"]) for c in citations],
                        notes=f"classifier p={top_prob:.2f}",
                    )
                )

        for obs in observe_image(provider, image, trace):
            text = str(obs.get("finding", "")).strip()
            if not text:
                continue
            if measured_class and measured_class.replace("_", " ") in text.lower():
                continue  # dedup against the measured finding for this photo
            citations = retrieve_guidance(recognizer, f"{text} remediation", trace)
            severity = obs.get("severity", "unknown")
            findings.append(
                Finding(
                    finding=text,
                    tier="observation",
                    defect_class=None,
                    severity=severity if severity in ("cosmetic", "monitor", "moderate", "structural") else "unknown",
                    evidence_photo=str(path),
                    citations=[Citation(card_id=c["card_id"], title=c["title"]) for c in citations],
                    notes="open-vocabulary VLM observation, not benchmarked",
                )
            )

    audio_band = None
    if audio_analyzer is not None and audio_bytes:
        audio_finding = score_audio(audio_analyzer, audio_bytes, trace)
        audio_band = getattr(audio_finding, "band", None)

    findings_json = json.dumps(
        [{"finding": f.finding, "tier": f.tier, "severity": f.severity} for f in findings]
    )
    summary = ""
    for _attempt in range(2):
        with trace.span("synthesize_summary", {"findings": len(findings)}) as span:
            summary = provider.complete(
                SUMMARY_PROMPT.format(findings=findings_json), max_tokens=1024
            ).strip()
            span["result_digest"] = summary[:80]
        if summary:
            break
    if not summary:
        classes = sorted({f.finding for f in findings}) or ["no findings"]
        summary = f"{len(findings)} finding(s): {', '.join(classes)}."

    report = InspectionReport(
        property_id=property_id, findings=findings, summary=summary, audio_band=audio_band
    )
    (out_dir / f"report_{property_id}.json").write_text(report.model_dump_json(indent=2))
    return report, provider.usage(), trace.path
