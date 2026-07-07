import pytest

from defectlens.corpus import Card
from defectlens.eval.rag_recall import rrf_fuse
from defectlens.serve.recognizer import (
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
