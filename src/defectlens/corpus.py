"""Guidance-card corpus: YAML schema, loader, validation (spec §6)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from defectlens.taxonomy import UNIFIED_CLASSES

SEVERITIES = ("structural", "urgent", "monitor", "cosmetic")
REQUIRED_CARD_KEYS = (
    "id", "title", "class_tags", "severity", "index_sentence", "passage", "citation",
)


@dataclass(frozen=True)
class Card:
    id: str
    title: str
    class_tags: list[str]
    severity: str
    index_sentence: str
    passage: str
    citation: str
    source_name: str
    source_url: str
    source_license: str


def load_corpus_file(path: Path) -> list[Card]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "source" not in raw or "cards" not in raw:
        raise ValueError(f"{path}: expected top-level 'source' and 'cards'")
    src = raw["source"]
    if not isinstance(src, dict):
        raise ValueError(f"{path}: 'source' must be a mapping")
    for k in ("name", "url", "license"):
        if not src.get(k):
            raise ValueError(f"{path}: source missing {k!r}")
    if not isinstance(raw["cards"], list):
        raise ValueError(f"{path}: 'cards' must be a list")
    cards: list[Card] = []
    for i, entry in enumerate(raw["cards"]):
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: card {i} is not a mapping: {entry!r}")
        missing = [k for k in REQUIRED_CARD_KEYS if not entry.get(k)]
        if missing:
            raise ValueError(f"{path}: card {i} missing/empty {missing}: {entry.get('id', '?')}")
        for k in ("id", "title", "index_sentence", "passage", "citation"):
            if not isinstance(entry[k], str):
                raise ValueError(f"{path}: card {i} field {k!r} must be a string")
        tags = entry["class_tags"]
        if not isinstance(tags, list):
            raise ValueError(f"{path}: card {entry['id']} class_tags must be a list")
        bad = [t for t in tags if t not in UNIFIED_CLASSES]
        if bad or not tags:
            raise ValueError(f"{path}: card {entry['id']} has invalid class_tags {bad}")
        if entry["severity"] not in SEVERITIES:
            raise ValueError(
                f"{path}: card {entry['id']} invalid severity {entry['severity']!r}"
            )
        cards.append(
            Card(
                id=entry["id"],
                title=entry["title"],
                class_tags=list(tags),
                severity=entry["severity"],
                index_sentence=entry["index_sentence"],
                passage=entry["passage"].strip(),
                citation=entry["citation"],
    source_name=src["name"],
                source_url=src["url"],
                source_license=src["license"],
            )
        )
    return cards


def load_corpus_dir(dir_path: Path) -> list[Card]:
    cards: list[Card] = []
    seen: dict[str, Path] = {}
    for f in sorted(Path(dir_path).glob("*.yaml")):
        for c in load_corpus_file(f):
            if c.id in seen:
                raise ValueError(f"duplicate card id {c.id} in {f} (first in {seen[c.id]})")
            seen[c.id] = f
            cards.append(c)
    return cards
