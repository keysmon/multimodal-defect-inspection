import numpy as np
import pytest

from defectlens.rag import audio_db


# Dedicated test database, mirroring tests/test_rag_db.py: audio_db.clear() in
# the fixture must never touch a production table (the real card index once got
# wiped from the shared DB — 2026-07-07).
TEST_DSN = "postgresql://defectlens:defectlens@localhost:5433/defectlens_test"


def get_conn():
    try:
        return audio_db.connect(TEST_DSN)
    except Exception:
        return None


conn = get_conn()
pytestmark = pytest.mark.skipif(conn is None, reason="pgvector DB not running")


@pytest.fixture(autouse=True)
def fresh_schema():
    audio_db.ensure_schema(conn)
    audio_db.clear(conn)
    yield
    audio_db.clear(conn)  # don't leave toy vectors behind


def unit(i: int) -> np.ndarray:
    v = np.zeros(audio_db.DIM, dtype=np.float32)
    v[i] = 1.0
    return v


def at_45deg() -> np.ndarray:
    """Unit vector 45° from unit(0): [1/√2, 1/√2, 0, ...]."""
    v = np.zeros(audio_db.DIM, dtype=np.float32)
    v[0] = v[1] = 1.0
    return v / np.linalg.norm(v)


def test_upsert_and_topk_roundtrip():
    audio_db.upsert(conn, "hvac-1", ["fan_imbalance"], unit(0))
    hits = audio_db.top_k(conn, unit(0), k=1)
    assert hits[0][0] == "hvac-1"
    assert hits[0][1] == ["fan_imbalance"]
    assert hits[0][2] < 1e-6


def test_topk_orders_by_distance():
    audio_db.upsert(conn, "a", ["fan_imbalance"], unit(0))
    audio_db.upsert(conn, "b", ["bearing_wear"], unit(1))
    hits = audio_db.top_k(conn, unit(0), k=2)
    assert hits[0][0] == "a" and hits[0][2] < 1e-6
    assert hits[1][0] == "b"


def test_upsert_replaces_on_primary_key():
    audio_db.upsert(conn, "a", ["fan_imbalance"], unit(0))
    audio_db.upsert(conn, "a", ["bearing_wear"], unit(2))  # same card_id -> replace
    hits = audio_db.top_k(conn, unit(2), k=5)
    assert [h[0] for h in hits] == ["a"]  # single row per card_id (PK)
    assert hits[0][1] == ["bearing_wear"]


def test_topk_truncates_to_k():
    audio_db.upsert(conn, "a", ["fan_imbalance"], unit(0))
    audio_db.upsert(conn, "b", ["bearing_wear"], unit(1))
    audio_db.upsert(conn, "c", ["pump_cavitation"], unit(2))
    hits = audio_db.top_k(conn, unit(0), k=2)
    assert len(hits) == 2


def test_topk_uses_cosine_distance_operator():
    # Two unit vectors 45° apart: cosine distance = 1 - cos(45°) ≈ 0.293.
    # (L2 distance would be ≈ 0.765.) Pins the <=> cosine operator so a future
    # swap to <-> (L2) or <#> (inner product) fails this test.
    audio_db.upsert(conn, "a", ["fan_imbalance"], unit(0))
    hits = audio_db.top_k(conn, at_45deg(), k=1)
    assert hits[0][0] == "a"
    assert hits[0][2] == pytest.approx(0.2929, abs=1e-3)
