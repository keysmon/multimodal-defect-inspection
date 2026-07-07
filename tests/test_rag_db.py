import numpy as np
import pytest

from defectlens.rag import db


def get_conn():
    try:
        return db.connect()
    except Exception:
        return None


conn = get_conn()
pytestmark = pytest.mark.skipif(conn is None, reason="pgvector DB not running")


@pytest.fixture(autouse=True)
def fresh_schema():
    db.init_schema(conn)
    db.clear(conn)
    yield


def unit(i: int) -> np.ndarray:
    v = np.zeros(db.DIM, dtype=np.float32)
    v[i] = 1.0
    return v


def test_upsert_and_topk_orders_by_distance():
    db.upsert_vector(conn, "a", "text", ["crack"], unit(0))
    db.upsert_vector(conn, "b", "text", ["spalling"], unit(1))
    hits = db.top_k(conn, unit(0), k=2, kinds=("text",))
    assert hits[0][0] == "a" and hits[0][2] < 1e-6
    assert hits[1][0] == "b"


def test_upsert_is_idempotent():
    db.upsert_vector(conn, "a", "text", ["crack"], unit(0))
    db.upsert_vector(conn, "a", "text", ["crack"], unit(2))  # replaces
    hits = db.top_k(conn, unit(2), k=1, kinds=("text",))
    assert hits[0][0] == "a"


def test_dedup_by_card_across_kinds():
    db.upsert_vector(conn, "a", "text", ["crack"], unit(0))
    db.upsert_vector(conn, "a", "image_centroid", ["crack"], unit(1))
    hits = db.top_k(conn, unit(1), k=5, kinds=("text", "image_centroid"))
    assert [h[0] for h in hits] == ["a"]
