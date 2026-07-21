"""Walkthrough synthesis: the Haiku-vision PRIMARY engine (design 2026-07-21).

Flow: extract concerns from the visit note -> retrieve candidate cards per
photo (fused CLIP image retrieval) and per concern (text retrieval) -> ONE
multi-image provider call that sees every photo, every candidate card, and
the concern checklist -> deterministic citation gate.

The gate guarantees, by construction: every input photo appears exactly
once; every concern gets an answer (cited, or an explicit not-observed);
every kept claim cites only cards retrieved for THIS walkthrough; every
dropped claim is recorded in flagged_claims. The LLM proposes; the gate
disposes.
"""
from __future__ import annotations

import json
import logging
from io import BytesIO

from PIL import Image

from defectlens.grounding.citations import validate_citations
from defectlens.grounding.retrieval import retrieve_for_photo, retrieve_for_text
from defectlens.report.concerns import extract_concerns
from defectlens.report.schema import (
    ActionItem,
    ConcernAnswer,
    PhotoFinding,
    WalkthroughReport,
    WalkthroughSummary,
    parse_synthesis_json,
)

logger = logging.getLogger(__name__)

MAX_PHOTOS = 10  # multi-image context cap (design "Risks")
PHOTO_K = 5      # candidate cards per photo (matches /analyze's k)
CONCERN_K = 3    # candidate cards per extracted concern (matches agent's k)
_PRIORITIES = ("high", "medium", "low")

NOT_OBSERVED_PHOTO = (
    "Not observed - no defect matched to guidance in this photo; verify on-site."
)
NOT_OBSERVED_ANSWER = "Not observed in these photos - verify on-site."

SYNTHESIS_PROMPT = """You are drafting an INITIAL diagnostic report for a building
technician after a first site visit. You are given {n_photos} photos (in order:
{photo_ids}), the technician's concerns, and a set of guidance cards retrieved
for these photos and concerns.

Photos:
{photo_lines}

Technician concerns (answer EVERY one):
{concerns_json}

Guidance cards (the ONLY cards you may cite, by exact card_id):
{cards_block}

Respond with ONLY a JSON object, exactly this shape:
{{"per_photo": [{{"photo_id": "...", "observation": "<what is visible>",
   "cited": ["<card_id>"], "no_evidence": false}}],
 "summary": {{
   "overall_assessment": "<2-4 sentences, may reason ACROSS photos>",
   "action_items": [{{"priority": "high|medium|low", "text": "<check or action>",
      "citations": ["<card_id>"], "photo_refs": ["<photo_id>"]}}],
   "answers": [{{"concern": "<concern verbatim>", "answer": "<grounded answer>",
      "citations": ["<card_id>"]}}]}}}}

Hard rules:
- Every observation, action item, and answer MUST cite at least one card_id
  from the list above. NEVER invent a card_id.
- If a photo shows no defect, or nothing in the cards applies to what you see,
  set "no_evidence": true for that photo and cite nothing.
- If the photos cannot answer a concern, answer it with "citations": [] - it
  will be reported as "not observed, verify on-site". Do NOT guess.
- Describe only what is visible. This is a draft to verify, not a verdict."""


def _card_line(card) -> str:
    tags = ",".join(card.class_tags)
    return f"- {card.id} [{tags}] {card.title}: {card.passage}"


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def run_walkthrough(
    *,
    photos: list[dict],
    visit_note: str | None,
    recognizer,
    provider,
    max_photos: int = MAX_PHOTOS,
) -> WalkthroughReport:
    if not photos:
        raise ValueError("a walkthrough needs at least one photo")
    if len(photos) > max_photos:
        raise ValueError(f"walkthrough capped at {max_photos} photos, got {len(photos)}")

    # 1. Concerns: the coverage checklist, extracted from the note.
    concerns = extract_concerns(provider, visit_note)

    # 2. Retrieval fan-out (CLIP retrieval-only): per photo + per concern.
    allowed: dict[str, object] = {}
    photo_ids: list[str] = []
    images: list[Image.Image] = []
    photo_lines: list[str] = []
    for photo in photos:
        pid = photo["photo_id"]
        photo_ids.append(pid)
        images.append(Image.open(BytesIO(photo["image_bytes"])).convert("RGB"))
        for card in retrieve_for_photo(
            recognizer, photo["image_bytes"], k=PHOTO_K, note=photo.get("note")
        ):
            allowed[card.id] = card
        note = photo.get("note")
        photo_lines.append(f"- {pid}" + (f" (technician note: {note})" if note else ""))
    for concern in concerns:
        for card in retrieve_for_text(recognizer, concern, k=CONCERN_K):
            allowed[card.id] = card

    # 3. ONE multi-image synthesis call (cross-photo reasoning happens here).
    prompt = SYNTHESIS_PROMPT.format(
        n_photos=len(photo_ids),
        photo_ids=", ".join(photo_ids),
        photo_lines="\n".join(photo_lines),
        concerns_json=json.dumps(concerns) if concerns else "[] (none given)",
        cards_block="\n".join(_card_line(c) for c in allowed.values()),
    )
    data = None
    for attempt in range(2):
        raw = provider.complete(prompt, images=images, max_tokens=2048)
        try:
            data = parse_synthesis_json(raw)
            break
        except ValueError:
            logger.warning("synthesis parse failed (attempt %d)", attempt + 1)
    if data is None:
        raise ValueError("synthesis response was not parseable JSON after retry")

    # 4. The citation gate (deterministic; the LLM proposed, this disposes).
    return _gate(data, concerns=concerns, photo_ids=photo_ids, allowed_ids=set(allowed))


def _gate(
    data: dict, *, concerns: list[str], photo_ids: list[str], allowed_ids: set[str]
) -> WalkthroughReport:
    flagged: list[dict] = []
    summary_raw = data.get("summary")
    if not isinstance(summary_raw, dict):
        summary_raw = {}

    # --- per-photo findings: every input photo, exactly once, in input order.
    raw_by_pid: dict[str, dict] = {}
    for entry in data.get("per_photo", []) if isinstance(data.get("per_photo"), list) else []:
        if not isinstance(entry, dict):
            continue
        pid = str(entry.get("photo_id", ""))
        if pid in photo_ids and pid not in raw_by_pid:
            raw_by_pid[pid] = entry
        elif pid not in photo_ids:
            flagged.append({"photo_id": pid, "reason": "unknown_photo_id"})

    per_photo: list[PhotoFinding] = []
    for pid in photo_ids:
        entry = raw_by_pid.get(pid)
        if entry is None:
            flagged.append({"photo_id": pid, "reason": "missing_photo_finding"})
            per_photo.append(
                PhotoFinding(photo_id=pid, observation=NOT_OBSERVED_PHOTO, no_evidence=True)
            )
            continue
        observation = str(entry.get("observation", "")).strip() or NOT_OBSERVED_PHOTO
        if entry.get("no_evidence") is True:
            per_photo.append(
                PhotoFinding(photo_id=pid, observation=observation, no_evidence=True)
            )
            continue
        cites = [str(c) for c in entry.get("cited", []) if isinstance(c, str)]
        kept, dropped = validate_citations(
            [{"text": observation, "citations": cites, "photo_id": pid}], allowed_ids
        )
        if kept:
            per_photo.append(
                PhotoFinding(photo_id=pid, observation=observation, cited=kept[0]["citations"])
            )
        else:
            flagged.extend(dropped)
            per_photo.append(
                PhotoFinding(photo_id=pid, observation=NOT_OBSERVED_PHOTO, no_evidence=True)
            )

    # --- action items: grounded or gone.
    action_items: list[ActionItem] = []
    raw_items = summary_raw.get("action_items", [])
    for item in raw_items if isinstance(raw_items, list) else []:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        cites = [str(c) for c in item.get("citations", []) if isinstance(c, str)]
        kept, dropped = validate_citations([{"text": text, "citations": cites}], allowed_ids)
        if not kept:
            flagged.extend(dropped)
            continue
        priority = str(item.get("priority", "")).lower()
        if priority not in _PRIORITIES:
            priority = "medium"  # presentation default; the advice itself is gated above
        refs = [str(r) for r in item.get("photo_refs", []) if str(r) in photo_ids]
        action_items.append(
            ActionItem(priority=priority, text=text, citations=kept[0]["citations"], photo_refs=refs)
        )

    # --- answers: coverage by construction - every concern, exactly once.
    raw_answers: dict[str, dict] = {}
    concern_by_norm = {_norm(c): c for c in concerns}
    answers_list = summary_raw.get("answers", [])
    for ans in answers_list if isinstance(answers_list, list) else []:
        if not isinstance(ans, dict):
            continue
        matched = concern_by_norm.get(_norm(str(ans.get("concern", ""))))
        if matched is None:
            flagged.append(
                {"text": str(ans.get("answer", "")), "concern": str(ans.get("concern", "")),
                 "reason": "unknown_concern"}
            )
        elif matched not in raw_answers:
            raw_answers[matched] = ans

    answers: list[ConcernAnswer] = []
    for concern in concerns:
        ans = raw_answers.get(concern)
        if ans is None:
            flagged.append({"concern": concern, "reason": "missing_answer"})
            answers.append(
                ConcernAnswer(concern=concern, answer=NOT_OBSERVED_ANSWER, not_observed=True)
            )
            continue
        text = str(ans.get("answer", "")).strip() or NOT_OBSERVED_ANSWER
        cites = [str(c) for c in ans.get("citations", []) if isinstance(c, str)]
        kept, dropped = validate_citations(
            [{"text": text, "citations": cites, "concern": concern}], allowed_ids
        )
        if kept:
            answers.append(
                ConcernAnswer(concern=concern, answer=text, citations=kept[0]["citations"])
            )
        else:
            flagged.extend(dropped)
            answers.append(
                ConcernAnswer(concern=concern, answer=NOT_OBSERVED_ANSWER, not_observed=True)
            )

    # --- overall assessment: a summary of gated content; deterministic fallback
    # mirrors the agent's (a provider hiccup must not sink assembled findings).
    overall = str(summary_raw.get("overall_assessment", "")).strip()
    if not overall:
        grounded_n = sum(1 for f in per_photo if not f.no_evidence)
        overall = (
            f"{grounded_n} of {len(per_photo)} photos show observations matched to "
            "guidance cards; review the action items."
        )

    return WalkthroughReport(
        concerns=concerns,
        per_photo=per_photo,
        summary=WalkthroughSummary(
            overall_assessment=overall, action_items=action_items, answers=answers
        ),
        flagged_claims=flagged,
    )
