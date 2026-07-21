"""ArrayVectorStore: no-DB serving path (Phase 5.5a).

Synthetic vectors only — no CLIP/CLAP, no DB. Locks the behaviours the cloud
Lambda depends on: brute-force cosine that matches pgvector's `<=>` distance and
DISTINCT-ON-per-card dedup, and the three query shapes (visual/audio/search) the
Recognizer and AudioAnalyzer call — including the search index that covers hvac
audio cards absent from the visual index.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from defectlens.rag.vector_store import ArrayVectorStore


def _unit(i: int, dim: int) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    v[i] = 1.0
    return v


def _write_npz(tmp_path, *, visual, audio, search, exemplar=(), version=3):
    """visual/audio/search/exemplar are lists of tuples; version omits keys
    below the corresponding format to simulate stale (v1/v2) artifacts."""
    path = tmp_path / "card_vectors.npz"
    arrays = dict(
        visual_ids=np.array([v[0] for v in visual]),
        visual_tags_json=np.array([json.dumps(v[1]) for v in visual]),
        visual_embeddings_text=np.stack([v[2] for v in visual]).astype(np.float32),
        visual_embeddings_centroid=np.stack([v[3] for v in visual]).astype(np.float32),
        audio_ids=np.array([a[0] for a in audio]),
        audio_tags_json=np.array([json.dumps(a[1]) for a in audio]),
        audio_embeddings=np.stack([a[2] for a in audio]).astype(np.float32),
    )
    if version >= 2:
        arrays["search_ids"] = np.array([s[0] for s in search])
        arrays["search_embeddings_text"] = np.stack([s[1] for s in search]).astype(np.float32)
        arrays["format_version"] = np.array(version)
    if version >= 3:
        # (id, meta_dict, vec) tuples; empty exemplar index is a valid v3 state.
        arrays["exemplar_ids"] = np.array([e[0] for e in exemplar])
        arrays["exemplar_meta_json"] = np.array([json.dumps(e[1]) for e in exemplar])
        arrays["exemplar_embeddings"] = (
            np.stack([e[2] for e in exemplar]).astype(np.float32)
            if exemplar else np.zeros((0, 4), np.float32)
        )
    np.savez(path, **arrays)
    return path


def _sample_store(tmp_path):
    visual = [
        # (id, tags, text_vec, centroid_vec) in dim 4
        ("v_a", ["crack"], _unit(0, 4), _unit(1, 4)),
        ("v_b", ["spalling"], _unit(1, 4), _unit(2, 4)),
        ("v_c", ["mold_algae"], _unit(2, 4), _unit(3, 4)),
    ]
    audio = [
        ("h_a", ["bearing_wear"], _unit(0, 3)),
        ("h_b", ["fan_imbalance"], _unit(1, 3)),
    ]
    # search index covers ALL cards: the 3 visual + one hvac audio card (dim 4).
    search = [
        ("v_a", _unit(0, 4)),
        ("v_b", _unit(1, 4)),
        ("v_c", _unit(2, 4)),
        ("hvac-1", _unit(3, 4)),
    ]
    exemplar = [
        ("ex_1", {"id": "ex_1", "card_ids": ["v_a"], "class_tags": ["crack"], "caption": "one"}, _unit(0, 4)),
        ("ex_2", {"id": "ex_2", "card_ids": ["v_a", "v_b"], "class_tags": ["crack"], "caption": "two"}, _unit(1, 4)),
        ("ex_3", {"id": "ex_3", "card_ids": ["v_a"], "class_tags": ["crack"], "caption": "three"}, _unit(2, 4)),
        ("ex_4", {"id": "ex_4", "card_ids": ["v_a"], "class_tags": ["crack"], "caption": "four"}, _unit(3, 4)),
        ("ex_5", {"id": "ex_5", "card_ids": [], "class_tags": ["mold_algae"], "caption": "five"}, _unit(0, 4)),
    ]
    path = _write_npz(tmp_path, visual=visual, audio=audio, search=search, exemplar=exemplar)
    return ArrayVectorStore.load(path)


# ---------------------------------------------------------------------------
# loading + shape
# ---------------------------------------------------------------------------


def test_load_roundtrips_ids_tags_and_count(tmp_path):
    store = _sample_store(tmp_path)
    assert store.visual_count() == 3
    rows = store.visual_top_k(_unit(1, 4), k=3, kinds=("image_centroid",))
    ids = [cid for cid, _tags, _dist in rows]
    # centroid of v_a is unit(1); a query of unit(1) is nearest to v_a.
    assert ids[0] == "v_a"
    # tags come back as the parsed JSON list, not the raw string.
    assert rows[0][1] == ["crack"]


# ---------------------------------------------------------------------------
# visual_top_k — cosine distance + ordering
# ---------------------------------------------------------------------------


def test_visual_top_k_distance_matches_cosine(tmp_path):
    store = _sample_store(tmp_path)
    # image_centroid vectors: v_a=unit1, v_b=unit2, v_c=unit3.
    rows = store.visual_top_k(_unit(1, 4), k=3, kinds=("image_centroid",))
    by_id = {cid: dist for cid, _tags, dist in rows}
    # cosine distance = 1 - cos_sim: identical unit vectors -> 0, orthogonal -> 1.
    assert by_id["v_a"] == pytest.approx(0.0, abs=1e-6)
    assert by_id["v_b"] == pytest.approx(1.0, abs=1e-6)
    assert by_id["v_c"] == pytest.approx(1.0, abs=1e-6)


def test_visual_top_k_orders_nearest_first_and_caps_at_k(tmp_path):
    store = _sample_store(tmp_path)
    # text vectors: v_a=unit0, v_b=unit1, v_c=unit2. Query near unit0.
    q = _unit(0, 4) * 0.9 + _unit(1, 4) * 0.1
    rows = store.visual_top_k(q, k=2, kinds=("text",))
    assert [cid for cid, _t, _d in rows] == ["v_a", "v_b"]
    assert len(rows) == 2


def test_visual_top_k_text_and_centroid_dedup_keeps_min_distance(tmp_path):
    """Multi-kind query mirrors pgvector DISTINCT ON (card_id): one row per card
    at its nearest vector across the requested kinds."""
    store = _sample_store(tmp_path)
    # Query = unit(1). v_a: text=unit0 (dist 1), centroid=unit1 (dist 0) -> 0.
    #                   v_b: text=unit1 (dist 0), centroid=unit2 (dist 1) -> 0.
    #                   v_c: text=unit2 (dist 1), centroid=unit3 (dist 1) -> 1.
    rows = store.visual_top_k(_unit(1, 4), k=5, kinds=("text", "image_centroid"))
    by_id = {cid: dist for cid, _t, dist in rows}
    assert set(by_id) == {"v_a", "v_b", "v_c"}  # one row per card, not two
    assert by_id["v_a"] == pytest.approx(0.0, abs=1e-6)
    assert by_id["v_b"] == pytest.approx(0.0, abs=1e-6)
    assert by_id["v_c"] == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# audio_top_k
# ---------------------------------------------------------------------------


def test_audio_top_k_orders_nearest_first(tmp_path):
    store = _sample_store(tmp_path)
    rows = store.audio_top_k(_unit(0, 3), k=2)
    assert rows[0][0] == "h_a"
    assert rows[0][1] == ["bearing_wear"]
    assert rows[0][2] == pytest.approx(0.0, abs=1e-6)
    assert rows[1][0] == "h_b"


def test_audio_top_k_caps_at_k(tmp_path):
    store = _sample_store(tmp_path)
    assert len(store.audio_top_k(_unit(0, 3), k=1)) == 1


# ---------------------------------------------------------------------------
# normalization — unnormalized query and stored vectors still give cosine
# ---------------------------------------------------------------------------


def test_query_is_normalized_before_cosine(tmp_path):
    store = _sample_store(tmp_path)
    # A scaled query vector must give the same distances as the unit query.
    scaled = store.visual_top_k(_unit(1, 4) * 7.3, k=3, kinds=("image_centroid",))
    unit = store.visual_top_k(_unit(1, 4), k=3, kinds=("image_centroid",))
    assert [r[0] for r in scaled] == [r[0] for r in unit]
    for a, b in zip(scaled, unit):
        assert a[2] == pytest.approx(b[2], abs=1e-6)


# ---------------------------------------------------------------------------
# Recognizer + AudioAnalyzer use the injected store instead of the pgvector conn
# ---------------------------------------------------------------------------


class _SpyStore:
    """Records queries; returns the fixed cards it was built with. search_cards
    (defaults to cards) stands in for the all-cards search index."""

    def __init__(self, cards, search_cards=None):
        self.cards = cards
        self.search_cards = cards if search_cards is None else search_cards
        self.visual_kinds = []
        self.audio_calls = 0
        self.search_calls = 0

    def visual_top_k(self, embedding, k, kinds):
        self.visual_kinds.append(kinds)
        return [(c.id, c.class_tags, 0.0) for c in self.cards]

    def audio_top_k(self, embedding, k):
        self.audio_calls += 1
        return [(c.id, c.class_tags, 0.0) for c in self.cards]

    def search_top_k(self, embedding, k):
        self.search_calls += 1
        return [(c.id, [], 0.0) for c in self.search_cards]

    def visual_count(self):
        return len(self.cards)


def _make_card(cid, tags):
    from defectlens.corpus import Card

    return Card(
        id=cid, title="t", class_tags=tags, severity="monitor",
        index_sentence="s", passage="p", citation="c",
        source_name="n", source_url="u", source_license="l",
    )


def test_recognizer_routes_retrieval_through_injected_store(monkeypatch):
    """A Recognizer given a vector_store must query it (not the pgvector conn)
    for both the image-centroid and note-text retrievals."""
    import numpy as _np
    import torch

    from defectlens.serve import recognizer as recognizer_mod
    from defectlens.serve.recognizer import Recognizer
    from defectlens.taxonomy import UNIFIED_CLASSES

    cards = [_make_card("v_a", ["crack"]), _make_card("v_b", ["spalling"])]
    spy = _SpyStore(cards)

    # db.top_k must never be reached on the store path.
    def _boom(*a, **k):
        raise AssertionError("db.top_k must not be called when a store is injected")

    monkeypatch.setattr(recognizer_mod.db, "top_k", _boom)
    monkeypatch.setattr(
        recognizer_mod, "embed_texts", lambda *a, **k: _np.ones((1, 4), dtype=_np.float32)
    )

    class _FakeInputs(dict):
        def to(self, device):
            return self

    class _FakeProcessor:
        def __call__(self, images=None, return_tensors=None, **kwargs):
            return _FakeInputs()

    class _FakeCLIP:
        def get_image_features(self, **kwargs):
            return torch.ones(1, 4)

    rec = Recognizer(vector_store=spy)
    rec.cards = cards
    rec.lookup = {c.id: c for c in cards}
    rec.device = "cpu"
    rec.prompt_feats = _np.zeros((len(UNIFIED_CLASSES), 4), dtype=_np.float32)
    rec.processor = _FakeProcessor()
    rec.model = _FakeCLIP()
    assert rec.conn is None

    buf = np.zeros((8, 8, 3), dtype=np.uint8)
    from io import BytesIO

    from PIL import Image

    png = BytesIO()
    Image.fromarray(buf).save(png, "PNG")
    data = png.getvalue()

    rec.analyze_image_bytes(data, note="musty smell")
    # centroid retrieval + note-text retrieval both went to the store.
    assert ("image_centroid",) in spy.visual_kinds
    assert ("text",) in spy.visual_kinds


def test_audio_analyzer_routes_retrieval_through_injected_store():
    from defectlens.serve.audio_analyzer import AudioAnalyzer

    cards = [_make_card("h_a", ["bearing_wear"])]
    spy = _SpyStore(cards)

    a = AudioAnalyzer(vector_store=spy)
    a.vector_store = spy
    rows = a._audio_top_k([0.0], 5)
    assert spy.audio_calls == 1
    assert rows[0][0] == "h_a"


def test_audio_analyzer_top_k_empty_without_store_or_conn():
    from defectlens.serve.audio_analyzer import AudioAnalyzer

    a = AudioAnalyzer()  # no store, conn stays None
    assert a._audio_top_k([0.0], 5) == []


def test_search_text_routes_through_injected_store(monkeypatch):
    """Cloud /search (Recognizer.search_text) must query the store's search
    index — which covers hvac audio cards absent from the visual index — and
    join full Card metadata via search_lookup, NOT the pgvector conn and NOT the
    visual index. Regression for the E2E finding that equipment queries never
    returned hvac guidance."""
    import numpy as _np

    from defectlens.serve import recognizer as recognizer_mod
    from defectlens.serve.recognizer import Recognizer

    visual = [_make_card("v_a", ["crack"])]
    hvac = _make_card("hvac-001", ["bearing_wear"])
    spy = _SpyStore(visual, search_cards=[hvac])  # search returns the hvac card

    def _boom(*a, **k):
        raise AssertionError("db.top_k must not be called when a store is injected")

    monkeypatch.setattr(recognizer_mod.db, "top_k", _boom)
    monkeypatch.setattr(
        recognizer_mod, "embed_texts", lambda *a, **k: _np.ones((1, 4), dtype=_np.float32)
    )

    rec = Recognizer(vector_store=spy)
    rec.lookup = {"v_a": visual[0]}          # visual-only (fusion)
    rec.search_lookup = {"hvac-001": hvac}   # all cards incl. hvac (/search)
    rec.device = "cpu"
    rec.model = object()
    rec.processor = object()
    assert rec.conn is None

    hits = rec.search_text("pump making grinding noise", k=5)
    assert spy.search_calls == 1
    assert spy.visual_kinds == []  # did NOT use the visual index
    assert [h.card.id for h in hits] == ["hvac-001"]
    assert hits[0].card.class_tags == ["bearing_wear"]  # full Card metadata joined


def test_search_text_uses_db_conn_when_no_store(monkeypatch):
    """Local pgvector path unchanged: search_text routes to db.top_k('text')."""
    import numpy as _np

    from defectlens.serve import recognizer as recognizer_mod
    from defectlens.serve.recognizer import Recognizer

    cards = [_make_card("v_a", ["crack"])]
    captured = []

    def fake_top_k(conn, emb, k, kinds):
        captured.append((conn, kinds))
        return [(c.id, c.class_tags, 0.0) for c in cards]

    monkeypatch.setattr(recognizer_mod.db, "top_k", fake_top_k)
    monkeypatch.setattr(
        recognizer_mod, "embed_texts", lambda *a, **k: _np.ones((1, 4), dtype=_np.float32)
    )

    rec = Recognizer()  # no store
    rec.lookup = {c.id: c for c in cards}
    rec.device = "cpu"
    rec.model = object()
    rec.processor = object()
    rec.conn = object()

    hits = rec.search_text("query", k=3)
    assert captured == [(rec.conn, ("text",))]
    assert [h.card.id for h in hits] == ["v_a"]


# ---------------------------------------------------------------------------
# search_top_k + format version
# ---------------------------------------------------------------------------


def test_search_top_k_returns_hvac_card_nearest_its_vector(tmp_path):
    """The whole point of the search index: an hvac card (absent from the visual
    index) is retrievable when the query is nearest its search vector."""
    store = _sample_store(tmp_path)
    assert store.search_count() == 4  # 3 visual + 1 hvac
    rows = store.search_top_k(_unit(3, 4), k=1)  # hvac-1's search vector is unit(3)
    assert rows[0][0] == "hvac-1"
    assert rows[0][2] == pytest.approx(0.0, abs=1e-6)


def test_search_top_k_orders_by_cosine_and_caps_at_k(tmp_path):
    store = _sample_store(tmp_path)
    rows = store.search_top_k(_unit(0, 4), k=2)  # nearest v_a (unit0), then ties
    assert rows[0][0] == "v_a"
    assert len(rows) == 2


def test_load_rejects_stale_v1_npz_loudly(tmp_path):
    """A pre-search (v1) artifact lacking the search keys must fail loudly, not
    silently serve a search index that can't see hvac cards."""
    visual = [("v_a", ["crack"], _unit(0, 4), _unit(1, 4))]
    audio = [("h_a", ["bearing_wear"], _unit(0, 3))]
    path = _write_npz(tmp_path, visual=visual, audio=audio, search=[], version=1)
    with pytest.raises(ValueError, match="format v1|export_vector_artifacts"):
        ArrayVectorStore.load(path)


def test_load_rejects_stale_v2_npz_loudly(tmp_path):
    """A pre-exemplar (v2) artifact must fail loudly: serving would silently
    drop the exemplar thumb strips and similar-cases section."""
    visual = [("v_a", ["crack"], _unit(0, 4), _unit(1, 4))]
    audio = [("h_a", ["bearing_wear"], _unit(0, 3))]
    search = [("v_a", _unit(0, 4))]
    path = _write_npz(tmp_path, visual=visual, audio=audio, search=search, version=2)
    with pytest.raises(ValueError, match="format v2|export_vector_artifacts"):
        ArrayVectorStore.load(path)


# ---------------------------------------------------------------------------
# exemplar index (format v3)
# ---------------------------------------------------------------------------


def test_exemplar_top_k_orders_by_cosine_and_returns_meta(tmp_path):
    store = _sample_store(tmp_path)
    rows = store.exemplar_top_k(_unit(1, 4), 2)
    assert [r[0] for r in rows] == ["ex_2", "ex_1"] or [r[0] for r in rows][0] == "ex_2"
    top_id, meta, dist = rows[0]
    assert top_id == "ex_2"
    assert meta["caption"] == "two"
    assert dist == pytest.approx(0.0, abs=1e-6)
    assert store.exemplar_count() == 5


def test_exemplar_top_k_empty_when_no_exemplars(tmp_path):
    visual = [("v_a", ["crack"], _unit(0, 4), _unit(1, 4))]
    audio = [("h_a", ["bearing_wear"], _unit(0, 3))]
    search = [("v_a", _unit(0, 4))]
    path = _write_npz(tmp_path, visual=visual, audio=audio, search=search)
    store = ArrayVectorStore.load(path)
    assert store.exemplar_top_k(_unit(0, 4), 3) == []
    assert store.exemplar_count() == 0


def test_exemplars_for_card_joins_and_caps_at_three(tmp_path):
    store = _sample_store(tmp_path)
    # v_a is linked from four exemplars; the join caps at 3 (manifest order).
    metas = store.exemplars_for_card("v_a")
    assert [m["id"] for m in metas] == ["ex_1", "ex_2", "ex_3"]
    assert [m["id"] for m in store.exemplars_for_card("v_b")] == ["ex_2"]


def test_exemplars_for_card_falls_back_to_class_tags(tmp_path):
    """Cards without a curated join get class-tag-matched exemplars; unknown
    ids (e.g. audio cards absent from the visual index) get none."""
    store = _sample_store(tmp_path)
    # v_c (mold_algae) has no curated join; ex_5 shares its class tag.
    assert [m["id"] for m in store.exemplars_for_card("v_c")] == ["ex_5"]
    assert store.exemplars_for_card("hvac-999") == []


def test_fusion_never_returns_hvac_even_with_search_index(tmp_path):
    """Regression: hvac cards live only in the search index, never the visual
    fusion path. analyze_image_bytes must return visual cards only — the
    KeyError guard (fusion lookup is visual-only) holds under the v2 npz."""
    from io import BytesIO

    import numpy as _np
    import torch
    from PIL import Image

    from defectlens.serve.recognizer import Recognizer
    from defectlens.taxonomy import UNIFIED_CLASSES

    visual = [
        ("v_a", ["crack"], _unit(0, 4), _unit(1, 4)),
        ("v_b", ["spalling"], _unit(1, 4), _unit(2, 4)),
    ]
    audio = [("h_a", ["bearing_wear"], _unit(0, 3))]
    # search index deliberately includes an hvac card absent from the visual set.
    search = [("v_a", _unit(0, 4)), ("v_b", _unit(1, 4)), ("hvac-9", _unit(3, 4))]
    store = ArrayVectorStore.load(
        _write_npz(tmp_path, visual=visual, audio=audio, search=search)
    )

    cards = [_make_card("v_a", ["crack"]), _make_card("v_b", ["spalling"])]

    class _FakeInputs(dict):
        def to(self, device):
            return self

    class _FakeProcessor:
        def __call__(self, images=None, return_tensors=None, **kwargs):
            return _FakeInputs()

    class _FakeCLIP:
        def get_image_features(self, **kwargs):
            return torch.ones(1, 4)

    rec = Recognizer(vector_store=store)
    rec.cards = cards
    rec.lookup = {c.id: c for c in cards}  # visual-only — no hvac
    rec.device = "cpu"
    rec.prompt_feats = _np.zeros((len(UNIFIED_CLASSES), 4), dtype=_np.float32)
    rec.processor = _FakeProcessor()
    rec.model = _FakeCLIP()

    buf = np.zeros((8, 8, 3), dtype=np.uint8)
    png = BytesIO()
    Image.fromarray(buf).save(png, "PNG")

    result = rec.analyze_image_bytes(png.getvalue())  # must not KeyError
    hit_ids = [h.card.id for h in result.hits]
    assert all(not cid.startswith("hvac") for cid in hit_ids)
    assert set(hit_ids) <= {"v_a", "v_b"}
