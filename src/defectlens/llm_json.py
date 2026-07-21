"""Balanced-scan JSON extraction from LLM output, shared across layers.

Models wrap their JSON in prose, fences, or example blocks; these helpers
find top-level brace/bracket-balanced candidates without being confused by
braces inside JSON strings. Lifted from agent/schema.py + agent/tools.py so
the report layer can parse without importing agent internals.
"""
from __future__ import annotations

import json
import re

_FENCE = re.compile(r"```(?:json)?\s*(\[.*?\]|\{.*?\})\s*```", re.S)


def balanced_json_candidates(raw: str) -> list[str]:
    """Top-level brace-balanced {...} substrings, in order of appearance."""
    return _balanced_candidates(raw, "{", "}")


def balanced_array_candidates(raw: str) -> list[str]:
    """Top-level bracket-balanced [...] substrings, in order of appearance."""
    return _balanced_candidates(raw, "[", "]")


def _balanced_candidates(raw: str, open_ch: str, close_ch: str) -> list[str]:
    """One string-escape-aware depth scanner behind both public shapes.

    A minimal in-string flag (with backslash escapes) keeps delimiters inside
    double-quoted JSON strings from confusing the depth counter. Quotes are
    only tracked inside a candidate; prose quotes outside are ignored.
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
        elif ch == open_ch:
            if depth == 0:
                start = i
            depth += 1
        elif ch == close_ch and depth > 0:
            depth -= 1
            if depth == 0:
                candidates.append(raw[start : i + 1])
    return candidates


def parse_string_array(raw: str) -> list[str] | None:
    """Extract a JSON array of non-empty strings from LLM output, or None.

    Tries a fenced block first, then the bare response, then bracket-balanced
    scanning (last candidate first: models emit the real array after prose).
    Non-string and blank elements are dropped, not errors.
    """
    m = _FENCE.search(raw)
    candidates = ([m.group(1)] if m else []) + [raw.strip()]
    candidates += reversed(balanced_array_candidates(raw))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list):
            return [s.strip() for s in parsed if isinstance(s, str) and s.strip()]
    return None
