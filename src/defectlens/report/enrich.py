"""Fine-tuned-Qwen enrichment gate (P4, design 2026-07-21).

The Phase-3 fine-tuned Qwen2.5-VL is genuinely better than CLIP/general-Haiku
at its ~9-class concrete/structural taxonomy, but it is NARROW: it forces
every photo into those classes, so it confidently mislabels out-of-scope
items (HVAC, roofing, plumbing). The gate therefore merges a label into a
photo's finding ONLY when it is confident AND consistent with what Haiku
actually observed in that photo. Everything dropped is logged (the design's
"how often was the label kept vs dropped" eval surface). Enrichment NEVER
blocks or alters the report's claims - it only annotates them.
"""
from __future__ import annotations

import copy

from defectlens.report.schema import WalkthroughReport

CONFIDENCE_THRESHOLD = 0.5  # mirrors the agent's MEASURED_THRESHOLD

# Words/phrases whose presence in Haiku's observation makes the fine-tuned
# class plausible for that photo. Deliberately generous per class (Haiku
# paraphrases), but empty overlap = the two models saw different things.
CLASS_KEYWORDS: dict[str, tuple[str, ...]] = {
    "crack": ("crack", "fissure", "hairline"),
    "spalling": ("spall", "missing concrete", "chunks", "delaminat", "concrete loss"),
    "efflorescence": ("efflorescence", "white deposit", "mineral deposit", "chalky", "powdery"),
    "exposed_rebar": ("rebar", "exposed steel", "reinforcement", "exposed bar"),
    "corrosion_stain": ("corrosion", "rust"),
    "mold_algae": ("mold", "algae", "biological growth", "green growth", "moss"),
    "water_damage": ("water", "moisture", "damp", "stain", "ingress", "leak"),
    "peeling_paint": ("peeling", "flaking", "paint"),
    "no_defect": ("no defect", "no visible", "sound condition", "clean", "unremarkable"),
}


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def is_consistent(label: str, observation: str) -> bool:
    """True when the fine-tuned class is plausibly present in the observation."""
    haystack = _norm(observation)
    return any(keyword in haystack for keyword in CLASS_KEYWORDS.get(label, ()))


def merge_enrichment(
    report: dict, labels: dict[str, tuple[str, float]]
) -> tuple[dict, dict]:
    """Merge gated fine-tuned labels into a stored report dict.

    labels: {photo_id: (label, confidence)}. Returns (new report dict, gate
    log). The input is not mutated (the caller persists the returned dict).
    Gate log: {"kept": n, "dropped": [{photo_id, label, confidence, reason}]}
    with reasons low_confidence / no_evidence_photo /
    inconsistent_with_observation / unknown_photo_id.
    """
    merged = copy.deepcopy(report)
    findings = {f["photo_id"]: f for f in merged.get("per_photo", [])}
    kept = 0
    dropped: list[dict] = []

    def drop(pid: str, label: str, confidence: float, reason: str) -> None:
        dropped.append(
            {"photo_id": pid, "label": label, "confidence": confidence, "reason": reason}
        )

    for pid, (label, confidence) in labels.items():
        finding = findings.get(pid)
        if finding is None:
            drop(pid, label, confidence, "unknown_photo_id")
        elif confidence < CONFIDENCE_THRESHOLD:
            drop(pid, label, confidence, "low_confidence")
        elif finding.get("no_evidence"):
            # Haiku saw nothing to ground here; a confident narrow-taxonomy
            # label would contradict the report's own no-evidence honesty.
            drop(pid, label, confidence, "no_evidence_photo")
        elif not is_consistent(label, finding.get("observation", "")):
            drop(pid, label, confidence, "inconsistent_with_observation")
        else:
            finding["enrichment"] = {
                "label": label,
                "confidence": confidence,
                "consistent": True,
            }
            kept += 1

    WalkthroughReport.model_validate(merged)  # enrichment must keep schema validity
    return merged, {"kept": kept, "dropped": dropped}
