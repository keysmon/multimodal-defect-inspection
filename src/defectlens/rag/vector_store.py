"""In-memory vector store for no-DB serving (Phase 5.5a, spec decision 7).

The cloud deploy has no pgvector: the ~457 card vectors (410 visual x768 +
47 audio x512) are baked into the Lambda image as one ``.npz`` and answered by
brute-force cosine over normalized vectors — microseconds at this corpus size.

``VectorStore`` is the seam both serving paths share. Its two methods match the
two ``top_k`` call shapes already in the codebase, minus the pgvector ``conn``:

- ``visual_top_k(embedding, k, kinds)`` mirrors ``rag.db.top_k`` (Recognizer).
- ``audio_top_k(embedding, k)`` mirrors ``rag.audio_db.top_k`` (AudioAnalyzer).

Both return the same ``[(card_id, class_tags, cosine_distance)]`` row shape the
pgvector functions do, so ``rag.retrieve.hits_from_rows`` reuses unchanged.
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
    ) -> None:
        self.visual_ids = list(visual_ids)
        self.visual_tags = list(visual_tags)
        self._visual = {
            "text": _normalize_rows(visual_text),
            "image_centroid": _normalize_rows(visual_centroid),
        }
        self.audio_ids = list(audio_ids)
        self.audio_tags = list(audio_tags)
        self._audio = _normalize_rows(audio_emb)

    @classmethod
    def load(cls, path: str | Path) -> "ArrayVectorStore":
        data = np.load(path, allow_pickle=False)
        return cls(
            visual_ids=[str(x) for x in data[VISUAL_IDS]],
            visual_tags=[json.loads(s) for s in data[VISUAL_TAGS]],
            visual_text=data[VISUAL_TEXT],
            visual_centroid=data[VISUAL_CENTROID],
            audio_ids=[str(x) for x in data[AUDIO_IDS]],
            audio_tags=[json.loads(s) for s in data[AUDIO_TAGS]],
            audio_emb=data[AUDIO_EMB],
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
        tag_by_id = dict(zip(self.visual_ids, self.visual_tags))
        rows = [(cid, tag_by_id[cid], dist) for cid, dist in best.items()]
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
