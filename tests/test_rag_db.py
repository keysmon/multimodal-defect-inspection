import numpy as np
import pytest

from defectlens.rag import db


TEST_DSN = "postgresql://defectlens:defectlens@localhost:5433/defectlens_test"


def get_conn():
    # Dedicated test database: db.clear() in the fixture must never touch the
    # production card_vectors table (it wiped the real index once — 2026-07-07).
    try:
        return db.connect(TEST_DSN)
    except Exception:
        return None


conn = get_conn()
pytestmark = pytest.mark.skipif(conn is None, reason="pgvector DB not running")


@pytest.fixture(autouse=True)
def fresh_schema():
    db.init_schema(conn)
    db.clear(conn)
    yield
    db.clear(conn)  # don't leave toy vectors in the shared table (real index coexists)


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
