"""Extract discrete concerns from the technician's free-text visit note.

The extracted list IS the coverage checklist (design 2026-07-21): every
concern gets its own card retrieval and must receive an answer - cited or
an explicit "not observed". Failure degrades honestly: if extraction cannot
be parsed (or the provider errors), the whole note becomes one concern so
its signal still drives retrieval and an answer, rather than being dropped.
"""
from __future__ import annotations

import logging

from defectlens.llm_json import parse_string_array

logger = logging.getLogger(__name__)

MAX_CONCERNS = 8

CONCERN_PROMPT = """You are triaging a building technician's site-visit note before a
photo review. Extract the distinct concerns or questions the technician wants
answered. Respond with ONLY a JSON array of short strings, one per concern,
in the note's order. No commentary.

Note:
{note}"""


def extract_concerns(provider, visit_note: str | None, max_concerns: int = MAX_CONCERNS) -> list[str]:
    if not visit_note or not visit_note.strip():
        return []
    note = visit_note.strip()
    try:
        raw = provider.complete(CONCERN_PROMPT.format(note=note), max_tokens=512)
        parsed = parse_string_array(raw)
    except Exception:
        logger.warning("concern extraction failed; using the whole note", exc_info=True)
        parsed = None
    if not parsed:
        return [note]
    seen: list[str] = []
    for concern in parsed:
        if concern not in seen:
            seen.append(concern)
        if len(seen) == max_concerns:
            break
    return seen
