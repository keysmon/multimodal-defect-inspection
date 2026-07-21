"""Workflow tools: thin wrappers over existing components + JSONL tracing."""
from __future__ import annotations

import json
import re
import time
from contextlib import contextmanager
from pathlib import Path

from defectlens.llm_json import balanced_array_candidates as _balanced_array_candidates

OBSERVE_PROMPT = """You are assisting a building inspector. Look at this photo and list any
visible defects or maintenance concerns. Respond with ONLY a JSON array; each
element: {"finding": "<short description>", "severity": "cosmetic|monitor|moderate|structural"}.
If nothing is wrong, respond with []."""

_FENCE = re.compile(r"```(?:json)?\s*(\[.*?\]|\{.*?\})\s*```", re.S)


def _parse_observation_list(raw: str) -> list[dict] | None:
    """Extract a JSON array of observation dicts from LLM output, or None.

    Tries the fenced block first, then the bare response, then falls back to
    bracket-balanced scanning (last candidate first: models emit the real
    array after any prose) so an unclosed fence still parses.
    """
    m = _FENCE.search(raw)
    candidates = ([m.group(1)] if m else []) + [raw.strip()]
    candidates += reversed(_balanced_array_candidates(raw))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list):
            return [o for o in parsed if isinstance(o, dict)]
    return None


class Trace:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def span(self, step: str, args: dict):
        record: dict = {"step": step, "args": args, "ts": time.time()}
        start = time.perf_counter()
        try:
            yield record
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            record["elapsed_ms"] = round((time.perf_counter() - start) * 1000, 1)
            with self.path.open("a") as f:
                f.write(json.dumps(record, default=str) + "\n")


def classify_image(describer, image, trace: Trace, note: str | None = None):
    """Tier-1 measured ranking via the fine-tuned model ([] if no adapter)."""
    with trace.span("classify_image", {"note": bool(note)}) as span:
        ranking = describer.rank_classes(image, note=note)
        span["result_digest"] = ranking[:3]
    return ranking


def observe_image(provider, image, trace: Trace) -> list[dict]:
    """Tier-2 open-vocabulary observations. Unparseable output -> [] (logged)."""
    with trace.span("observe_image", {"provider": getattr(provider, "name", "?")}) as span:
        raw = provider.complete(OBSERVE_PROMPT, image=image, max_tokens=512)
        observations = _parse_observation_list(raw)
        if observations is None:
            observations = []
            span["parse_error"] = raw[:200]
        span["result_digest"] = [o.get("finding") for o in observations][:5]
    return observations


def retrieve_guidance(recognizer, query: str, trace: Trace, k: int = 3) -> list[dict]:
    """Cited cards for a finding via the existing text retrieval.

    Recognizer.search_text returns rag.retrieve.Hit objects: Card metadata
    lives behind ``hit.card`` (``id``, ``title``, ``class_tags``).
    """
    with trace.span("retrieve_guidance", {"query": query, "k": k}) as span:
        hits = recognizer.search_text(query, k=k)
        citations = [
            {
                "card_id": h.card.id,
                "title": getattr(h.card, "title", ""),
                "class_tags": list(getattr(h.card, "class_tags", [])),
            }
            for h in hits
        ]
        span["result_digest"] = [c["card_id"] for c in citations]
    return citations


def score_audio(analyzer, wav_bytes: bytes, trace: Trace):
    """Existing CLAP anomaly banding; returns the AudioFinding or None."""
    with trace.span("score_audio", {"bytes": len(wav_bytes)}) as span:
        finding = analyzer.analyze(wav_bytes)
        span["result_digest"] = getattr(finding, "band", None)
    return finding
