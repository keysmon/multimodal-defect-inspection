"""pgvector storage for card vectors. One row per (card, vector kind)."""
from __future__ import annotations

import os

import psycopg
from pgvector.psycopg import register_vector

DSN = os.environ.get(
    "DEFECTLENS_DSN", "postgresql://defectlens:defectlens@localhost:5433/defectlens"
)
DIM = 768  # CLIP ViT-L/14 projected dim

SCHEMA = f"""
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS card_vectors (
    id SERIAL PRIMARY KEY,
    card_id TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('text', 'image_centroid')),
    class_tags TEXT[] NOT NULL,
    embedding vector({DIM}) NOT NULL,
    UNIQUE (card_id, kind)
);
CREATE INDEX IF NOT EXISTS card_vectors_embedding_idx
    ON card_vectors USING hnsw (embedding vector_cosine_ops);
"""


def connect(dsn: str = DSN) -> psycopg.Connection:
    conn = psycopg.connect(dsn, autocommit=True, connect_timeout=3)
    register_vector(conn)
    return conn


def init_schema(conn: psycopg.Connection) -> None:
    conn.execute(SCHEMA)


def upsert_vector(conn, card_id: str, kind: str, class_tags: list[str], embedding) -> None:
    conn.execute(
        """
        INSERT INTO card_vectors (card_id, kind, class_tags, embedding)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (card_id, kind)
        DO UPDATE SET class_tags = EXCLUDED.class_tags, embedding = EXCLUDED.embedding
        """,
        (card_id, kind, class_tags, embedding),
    )


def top_k(conn, embedding, k: int, kinds: tuple[str, ...]) -> list[tuple[str, list[str], float]]:
    """Return [(card_id, class_tags, cosine_distance)] nearest-first, deduped by card."""
    rows = conn.execute(
        """
        SELECT DISTINCT ON (card_id) card_id, class_tags, embedding <=> %s AS dist
        FROM card_vectors
        WHERE kind = ANY(%s)
        ORDER BY card_id, dist
        """,
        (embedding, list(kinds)),
    ).fetchall()
    return sorted(rows, key=lambda r: r[2])[:k]


def clear(conn) -> None:
    conn.execute("DELETE FROM card_vectors")
