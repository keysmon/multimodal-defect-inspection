"""pgvector storage for audio (CLAP) card vectors. One row per card.

Separate from rag/db.py because CLAP projects to 512 dims while CLIP projects
to 768 — a distinct table rather than a new ``kind`` on ``card_vectors``. A card
carries a single CLAP text vector (its audible-symptom index_sentence), so
``card_id`` is the primary key and there is no ``kind`` column. ``top_k`` returns
the same ``(card_id, class_tags, dist)`` row shape as ``db.top_k`` so
``rag.retrieve.hits_from_rows`` reuses cleanly.
"""
from __future__ import annotations

import os

import psycopg
from pgvector.psycopg import register_vector

DSN = os.environ.get(
    "DEFECTLENS_DSN", "postgresql://defectlens:defectlens@localhost:5433/defectlens"
)
DIM = 512  # CLAP (laion/clap-htsat-unfused) projected dim

SCHEMA = f"""
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS audio_card_vectors (
    card_id TEXT PRIMARY KEY,
    class_tags TEXT[] NOT NULL,
    embedding vector({DIM}) NOT NULL
);
CREATE INDEX IF NOT EXISTS audio_card_vectors_embedding_idx
    ON audio_card_vectors USING hnsw (embedding vector_cosine_ops);
"""


def connect(dsn: str = DSN) -> psycopg.Connection:
    conn = psycopg.connect(dsn, autocommit=True, connect_timeout=3)
    register_vector(conn)
    return conn


def ensure_schema(conn: psycopg.Connection) -> None:
    conn.execute(SCHEMA)


def upsert(conn, card_id: str, class_tags: list[str], embedding) -> None:
    conn.execute(
        """
        INSERT INTO audio_card_vectors (card_id, class_tags, embedding)
        VALUES (%s, %s, %s)
        ON CONFLICT (card_id)
        DO UPDATE SET class_tags = EXCLUDED.class_tags, embedding = EXCLUDED.embedding
        """,
        (card_id, class_tags, embedding),
    )


def top_k(conn, embedding, k: int) -> list[tuple[str, list[str], float]]:
    """Return [(card_id, class_tags, cosine_distance)] nearest-first."""
    # ORDER BY <=> uses the HNSW index: approximate NN, but effectively exact at
    # the corpus's ~50 cards. Revisit (exact scan / ef_search tuning) if it grows.
    return conn.execute(
        """
        SELECT card_id, class_tags, embedding <=> %s AS dist
        FROM audio_card_vectors
        ORDER BY dist
        LIMIT %s
        """,
        (embedding, k),
    ).fetchall()


def clear(conn) -> None:
    conn.execute("DELETE FROM audio_card_vectors")
