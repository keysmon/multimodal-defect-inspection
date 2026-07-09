from io import BytesIO

import numpy as np
import pytest
import torch
from PIL import Image

from defectlens.corpus import Card
from defectlens.eval.rag_recall import rrf_fuse
from defectlens.serve import recognizer as recognizer_mod
from defectlens.serve.recognizer import (
    Recognizer,
    class_ranking_from_cards,
    fused_card_ranking,
    severity_for,
)
from defectlens.taxonomy import UNIFIED_CLASSES


def make_card(cid, tags, severity="monitor"):
    return Card(
        id=cid, title="t", class_tags=tags, severity=severity,
        index_sentence="s", passage="p", citation="c",
        source_name="n", source_url="u", source_license="l",
    )


# ---------------------------------------------------------------------------
# fused_card_ranking
# ---------------------------------------------------------------------------


def test_fused_card_ranking_tied_scores_keep_centroid_insertion_order():
    cards_by_id = {
        "A": make_card("A", ["crack"]),
        "B": make_card("B", ["spalling"]),
        "C": make_card("C", ["mold_algae"]),
    }
    centroid_ranked_ids = ["A", "B", "C"]
    prompt_class_sims = dict.fromkeys(UNIFIED_CLASSES, 0.0)
    prompt_class_sims.update({"crack": 0.5, "spalling": 0.9, "mold_algae": 0.1})

    # max-tag-sim ranking (desc): B(0.9), A(0.5), C(0.1) -> ranking_prompt = ["B", "A", "C"]
    # RRF (k=60): A = 1/61 (centroid rank0) + 1/62 (prompt rank1) = 0.0325224748...
    #             B = 1/62 (centroid rank1) + 1/61 (prompt rank0) = 0.0325224748...  (tie w/ A)
    #             C = 1/63 (centroid rank2) + 1/63 (prompt rank2) = 0.0317460317...
    # A and B tie exactly; stable sort keeps A first (inserted first, from centroid ranking).
    expected = rrf_fuse([centroid_ranked_ids, ["B", "A", "C"]])
    assert expected == ["A", "B", "C"]
    assert (
        fused_card_ranking(centroid_ranked_ids, prompt_class_sims, cards_by_id)
        == expected
    )


def test_fused_card_ranking_prompt_can_beat_centroid_top_pick():
    cards_by_id = {
        "A": make_card("A", ["crack"]),
        "B": make_card("B", ["spalling"]),
        "C": make_card("C", ["mold_algae"]),
    }
    centroid_ranked_ids = ["A", "B", "C"]
    prompt_class_sims = dict.fromkeys(UNIFIED_CLASSES, 0.0)
    prompt_class_sims.update({"crack": 0.1, "spalling": 0.9, "mold_algae": 0.5})

    # max-tag-sim ranking (desc): B(0.9), C(0.5), A(0.1) -> ranking_prompt = ["B", "C", "A"]
    # RRF (k=60): A = 1/61 (centroid rank0) + 1/63 (prompt rank2) = 0.0322664585...
    #             B = 1/62 (centroid rank1) + 1/61 (prompt rank0) = 0.0325224748...
    #             C = 1/63 (centroid rank2) + 1/62 (prompt rank1) = 0.0320020481...
    # => B > A > C: B (1st prompt / 2nd centroid) beats A (1st centroid / 3rd prompt).
    expected = rrf_fuse([centroid_ranked_ids, ["B", "C", "A"]])
    assert expected == ["B", "A", "C"]
    assert (
        fused_card_ranking(centroid_ranked_ids, prompt_class_sims, cards_by_id)
        == expected
    )


# ---------------------------------------------------------------------------
# class_ranking_from_cards
# ---------------------------------------------------------------------------


def test_class_ranking_from_cards_first_card_rank_scoring():
    cards_by_id = {
        "c1": make_card("c1", ["crack"]),
        "c2": make_card("c2", ["spalling", "crack"]),
        "c3": make_card("c3", ["exposed_rebar"]),
    }
    fused_ids = ["c1", "c2", "c3"]

    ranking = class_ranking_from_cards(fused_ids, cards_by_id)

    assert len(ranking) == 9
    assert {c for c, _ in ranking} == set(UNIFIED_CLASSES)

    # crack: first fused card carrying it is c1 at rank 0 -> 1/(0+1) = 1.0
    assert ranking[0] == ("crack", 1.0)
    # spalling: first fused card carrying it is c2 at rank 1 -> 1/(1+1) = 0.5
    assert ranking[1] == ("spalling", 0.5)
    # exposed_rebar: first fused card carrying it is c3 at rank 2 -> 1/(2+1)
    assert ranking[2][0] == "exposed_rebar"
    assert ranking[2][1] == pytest.approx(1 / 3)

    # classes with no tagged card in the fused list get 0.0, ordered last,
    # preserving UNIFIED_CLASSES relative order (stable sort).
    zero_classes_expected = [
        c for c in UNIFIED_CLASSES if c not in ("crack", "spalling", "exposed_rebar")
    ]
    tail = ranking[3:]
    assert [c for c, _ in tail] == zero_classes_expected
    assert all(score == 0.0 for _, score in tail)


# ---------------------------------------------------------------------------
# severity_for
# ---------------------------------------------------------------------------


def test_severity_for_exposed_rebar_base_is_structural():
    # structural is already the most severe band -> a monitor-severity card
    # tagged with it cannot de-escalate.
    cards = [make_card("a", ["exposed_rebar"], severity="monitor")]
    assert severity_for("exposed_rebar", cards) == "structural"


def test_severity_for_monitor_class_escalated_by_urgent_card():
    cards = [make_card("a", ["crack"], severity="urgent")]
    assert severity_for("crack", cards) == "urgent"


def test_severity_for_no_defect_stays_cosmetic():
    cards = [make_card("a", ["no_defect"], severity="cosmetic")]
    assert severity_for("no_defect", cards) == "cosmetic"


def test_severity_for_ignores_cards_not_tagged_with_top_class():
    # card is severity="structural" but tagged "spalling", not the top_class
    # "crack" -> must be ignored, leaving the default "monitor" band.
    cards = [make_card("a", ["spalling"], severity="structural")]
    assert severity_for("crack", cards) == "monitor"


# ---------------------------------------------------------------------------
# Recognizer.analyze_image_bytes — inspector note conditions card retrieval
#
# test_recognizer.py had no Recognizer / DB / model fixture (only make_card +
# pure-function tests), so this builds the minimal fakes needed to drive
# analyze_image_bytes far enough to reach the fusion call, following the
# module's plain-helper style. embed_texts + db.top_k are monkeypatched (the
# note-embedding correctness is out of scope); rrf_fuse is a spy that records
# how many rankings the fusion received.
# ---------------------------------------------------------------------------


def _make_png_bytes() -> bytes:
    buf = BytesIO()
    Image.new("RGB", (8, 8)).save(buf, "PNG")
    return buf.getvalue()


class _FakeInputs(dict):
    def to(self, device):
        return self


class _FakeProcessor:
    def __call__(self, images=None, return_tensors=None, **kwargs):
        return _FakeInputs()


class _FakeCLIP:
    def get_image_features(self, **kwargs):
        return torch.ones(1, 4)


def _wire_recognizer(cards):
    rec = Recognizer()
    rec.cards = cards
    rec.lookup = {c.id: c for c in cards}
    rec.device = "cpu"
    rec.prompt_feats = np.zeros((len(UNIFIED_CLASSES), 4), dtype=np.float32)
    rec.processor = _FakeProcessor()
    rec.model = _FakeCLIP()
    rec.conn = object()
    return rec


def test_note_adds_third_ranking_to_card_fusion(monkeypatch):
    cards = [make_card("A", ["crack"]), make_card("B", ["spalling"])]
    rec = _wire_recognizer(cards)
    card_ids = [c.id for c in cards]

    # db.top_k backs both the image-centroid and the note-text retrievals;
    # return the same card rows regardless of embedding, but record the
    # `kinds` filter so we can prove the note path queries the text vectors.
    captured_kinds = []

    def fake_top_k(conn, emb, k, kinds):
        captured_kinds.append(kinds)
        return [(c.id, c.class_tags, 0.0) for c in cards]

    monkeypatch.setattr(recognizer_mod.db, "top_k", fake_top_k)
    # Note path needs only *some* fixed embedding; correctness is out of scope.
    monkeypatch.setattr(
        recognizer_mod,
        "embed_texts",
        lambda *a, **k: np.ones((1, 4), dtype=np.float32),
    )

    captured_ranking_counts = []

    def rrf_spy(rankings, k=60):
        captured_ranking_counts.append(len(rankings))
        return card_ids

    monkeypatch.setattr(recognizer_mod, "rrf_fuse", rrf_spy)

    png = _make_png_bytes()
    rec.analyze_image_bytes(png)  # no note -> centroid + prompt rankings only
    rec.analyze_image_bytes(png, note="musty smell")  # + note-text ranking

    assert captured_ranking_counts == [2, 3]
    # the note retrieval must query the text-vector kind, not image centroids
    assert ("text",) in captured_kinds


def test_fused_card_ranking_rejects_unknown_tags_loudly():
    """Regression lock for the E2E-found KeyError: an audio-tagged card in the
    visual fusion dict must never occur — Recognizer.load() filters hvac-*
    cards; this asserts the failure mode stays loud if that filter regresses."""
    import pytest

    from defectlens.serve.recognizer import fused_card_ranking

    audio_card = make_card("hvac-999", ["fan_imbalance"])
    with pytest.raises(KeyError):
        fused_card_ranking(["hvac-999"], {"crack": 0.5}, {"hvac-999": audio_card})
