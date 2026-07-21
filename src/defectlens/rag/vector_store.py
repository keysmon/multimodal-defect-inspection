"""In-memory vector store for no-DB serving (Phase 5.5a, spec decision 7).

The cloud deploy has no pgvector: the ~457 card vectors (410 visual x768 +
47 audio x512) are baked into the Lambda image as one ``.npz`` and answered by
brute-force cosine over normalized vectors — microseconds at this corpus size.

``VectorStore`` is the seam both serving paths share:

- ``visual_top_k(embedding, k, kinds)`` mirrors ``rag.db.top_k`` (Recognizer).
- ``audio_top_k(embedding, k)`` mirrors ``rag.audio_db.top_k`` (AudioAnalyzer).
- ``search_top_k(embedding, k)`` serves the /search box over ALL cards,
  including the hvac-* audio cards absent from the visual index (format v2).

All three return the same ``[(card_id, class_tags, cosine_distance)]`` row shape
the pgvector functions do, so ``rag.retrieve.hits_from_rows`` reuses unchanged.
``PgVectorStore`` (the local dev path) stays the two ``rag.db`` / ``rag.audio_db``
modules; ``ArrayVectorStore`` is the cloud impl. Recognizer / AudioAnalyzer take
an optional store and fall back to the pgvector ``conn`` when none is injected,
so local dev is unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

# npz key names — the export script (scripts/export_vector_artifacts.py) writes
# exactly these, so both sides read from one source of truth.
VISUAL_IDS = "visual_ids"
VISUAL_TAGS = "visual_tags_json"
VISUAL_TEXT = "visual_embeddings_text"
VISUAL_CENTROID = "visual_embeddings_centroid"
AUDIO_IDS = "audio_ids"
AUDIO_TAGS = "audio_tags_json"
AUDIO_EMB = "audio_embeddings"
# Search-scoped index (format v2): CLIP text embeddings of index_sentences for
# ALL cards, including the hvac-* audio cards absent from the visual
# card_vectors table. Serves the free-text /search box so a query like "pump
# grinding noise" can surface bearing_wear guidance. A v1 npz lacks these keys
# and load() rejects it loudly.
SEARCH_IDS = "search_ids"
SEARCH_TEXT = "search_embeddings_text"
# Exemplar-image index (format v3, plan 2026-07-21): CLIP IMAGE embeddings of
# the served exemplar derivatives (frontend/public/exemplars/<id>.jpg) plus a
# JSON metadata row per exemplar ({id, card_ids, class_tags, license, credit,
# source_url, caption, image_url, thumb_url}). Serves both the per-card thumb
# strips (joined by card_ids) and /analyze's "similar documented cases"
# (query-image cosine over these vectors).
EXEMPLAR_IDS = "exemplar_ids"
EXEMPLAR_META = "exemplar_meta_json"
EXEMPLAR_EMB = "exemplar_embeddings"
FORMAT = "format_version"
FORMAT_VERSION = 3

# Visual ``kinds`` map to the two per-card embedding matrices. Mirrors the
# ``kind`` CHECK constraint in rag/db.py's card_vectors table.
_VISUAL_KIND_KEY = {"text": VISUAL_TEXT, "image_centroid": VISUAL_CENTROID}

Row = tuple[str, list[str], float]


@runtime_checkable
class VectorStore(Protocol):
    """The retrieval seam Recognizer + AudioAnalyzer query.

    A single store answers both the visual (CLIP, 768-d, text/centroid kinds)
    and audio (CLAP, 512-d) queries so one injected object serves the whole
    request path.
    """

    def visual_top_k(self, embedding, k: int, kinds: tuple[str, ...]) -> list[Row]:
        ...

    def audio_top_k(self, embedding, k: int) -> list[Row]:
        ...

    def search_top_k(self, embedding, k: int) -> list[Row]:
        ...


def _normalize_rows(m: np.ndarray) -> np.ndarray:
    """L2-normalize each row; zero rows are left as zeros (guarded division).

    Matches rag.embed.normalize so a query embedded there and a matrix exported
    from there compare on the same footing.
    """
    m = np.asarray(m, dtype=np.float32)
    if m.ndim == 1:
        n = float(np.linalg.norm(m))
        return m / n if n > 0 else m
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return m / n


class ArrayVectorStore:
    """Brute-force cosine over card vectors held in memory (the cloud path).

    Cosine distance is ``1 - cos_sim`` to match pgvector's ``<=>`` operator, so
    a query that ranked cards one way against the DB ranks them identically
    here. Vectors are normalized once at construction; queries are normalized
    per call.
    """

    def __init__(
        self,
        visual_ids: list[str],
        visual_tags: list[list[str]],
        visual_text: np.ndarray,
        visual_centroid: np.ndarray,
        audio_ids: list[str],
        audio_tags: list[list[str]],
        audio_emb: np.ndarray,
        search_ids: list[str],
        search_text: np.ndarray,
        exemplar_ids: list[str] | None = None,
        exemplar_meta: list[dict] | None = None,
        exemplar_emb: np.ndarray | None = None,
    ) -> None:
        self.visual_ids = list(visual_ids)
        self.visual_tags = list(visual_tags)
        self._visual_tag_by_id = dict(zip(self.visual_ids, self.visual_tags))
        self._visual = {
            "text": _normalize_rows(visual_text),
            "image_centroid": _normalize_rows(visual_centroid),
        }
        self.audio_ids = list(audio_ids)
        self.audio_tags = list(audio_tags)
        self._audio = _normalize_rows(audio_emb)
        # Search-scoped index over ALL cards (visual + hvac audio).
        self.search_ids = list(search_ids)
        self._search = _normalize_rows(search_text)
        # Exemplar-image index (format v3) + card_id -> exemplar-meta join.
        self.exemplar_ids = list(exemplar_ids or [])
        self.exemplar_meta = list(exemplar_meta or [])
        self._exemplar = _normalize_rows(
            exemplar_emb if exemplar_emb is not None else np.zeros((0, 768), np.float32)
        )
        self._exemplars_by_card: dict[str, list[dict]] = {}
        for meta in self.exemplar_meta:
            for card_id in meta.get("card_ids", []):
                self._exemplars_by_card.setdefault(card_id, []).append(meta)

    @classmethod
    def load(cls, path: str | Path) -> "ArrayVectorStore":
        data = np.load(path, allow_pickle=False)
        version = int(data[FORMAT]) if FORMAT in data.files else 1
        if version < FORMAT_VERSION or EXEMPLAR_IDS not in data.files:
            raise ValueError(
                f"{path}: card_vectors.npz is format v{version} but v{FORMAT_VERSION} "
                "is required (exemplar-image vectors + metadata). "
                "Re-run scripts/export_vector_artifacts.py."
            )
        return cls(
            visual_ids=[str(x) for x in data[VISUAL_IDS]],
            visual_tags=[json.loads(s) for s in data[VISUAL_TAGS]],
            visual_text=data[VISUAL_TEXT],
            visual_centroid=data[VISUAL_CENTROID],
            audio_ids=[str(x) for x in data[AUDIO_IDS]],
            audio_tags=[json.loads(s) for s in data[AUDIO_TAGS]],
            audio_emb=data[AUDIO_EMB],
            search_ids=[str(x) for x in data[SEARCH_IDS]],
            search_text=data[SEARCH_TEXT],
            exemplar_ids=[str(x) for x in data[EXEMPLAR_IDS]],
            exemplar_meta=[json.loads(s) for s in data[EXEMPLAR_META]],
            exemplar_emb=data[EXEMPLAR_EMB],
        )

    def visual_count(self) -> int:
        """Number of indexed visual cards — the /health cards_indexed figure."""
        return len(self.visual_ids)

    def visual_top_k(self, embedding, k: int, kinds: tuple[str, ...]) -> list[Row]:
        """Nearest visual cards, deduped per card across ``kinds``.

        Reproduces rag.db.top_k: ``SELECT DISTINCT ON (card_id) ... embedding
        <=> %s ORDER BY card_id, dist`` keeps each card's nearest vector among
        the requested kinds; the outer sort then orders those by distance and
        caps at k.
        """
        q = _normalize_rows(np.asarray(embedding, dtype=np.float32))
        best: dict[str, float] = {}
        for kind in kinds:
            dists = 1.0 - self._visual[kind] @ q  # cosine distance per card
            for i, cid in enumerate(self.visual_ids):
                d = float(dists[i])
                if cid not in best or d < best[cid]:
                    best[cid] = d
        rows = [(cid, self._visual_tag_by_id[cid], dist) for cid, dist in best.items()]
        rows.sort(key=lambda r: r[2])
        return rows[:k]

    def audio_top_k(self, embedding, k: int) -> list[Row]:
        """Nearest audio cards (one CLAP vector per card), nearest-first."""
        q = _normalize_rows(np.asarray(embedding, dtype=np.float32))
        dists = 1.0 - self._audio @ q
        rows = [
            (cid, self.audio_tags[i], float(dists[i]))
            for i, cid in enumerate(self.audio_ids)
        ]
        rows.sort(key=lambda r: r[2])
        return rows[:k]

    def search_count(self) -> int:
        """Number of cards in the search-scoped index (visual + hvac audio)."""
        return len(self.search_ids)

    def exemplar_count(self) -> int:
        """Number of indexed exemplar images."""
        return len(self.exemplar_ids)

    def exemplar_top_k(self, embedding, k: int) -> list[tuple[str, dict, float]]:
        """Nearest exemplar images to a CLIP image embedding, nearest-first.

        Returns ``[(exemplar_id, meta, cosine_distance)]`` — meta is the full
        manifest-derived dict (card_ids, credit, caption, image/thumb urls) so
        callers can render "similar documented cases" without another lookup.
        """
        if not self.exemplar_ids:
            return []
        q = _normalize_rows(np.asarray(embedding, dtype=np.float32))
        dists = 1.0 - self._exemplar @ q
        rows = [
            (eid, self.exemplar_meta[i], float(dists[i]))
            for i, eid in enumerate(self.exemplar_ids)
        ]
        rows.sort(key=lambda r: r[2])
        return rows[:k]

    def exemplars_for_card(self, card_id: str, limit: int = 3) -> list[dict]:
        """Exemplar metadata joined by card_ids (manifest order), capped."""
        return self._exemplars_by_card.get(card_id, [])[:limit]

    def search_top_k(self, embedding, k: int) -> list[Row]:
        """Nearest cards for a free-text /search query, over ALL cards.

        Covers the hvac-* audio cards absent from the visual index, so equipment
        queries surface audible-symptom guidance. Rows carry empty class_tags —
        the caller joins to full Card metadata by id (via search_lookup), so the
        row tags are never read (mirrors how hits_from_rows consumes them).
        """
        q = _normalize_rows(np.asarray(embedding, dtype=np.float32))
        dists = 1.0 - self._search @ q
        rows = [(cid, [], float(dists[i])) for i, cid in enumerate(self.search_ids)]
        rows.sort(key=lambda r: r[2])
        return rows[:k]
