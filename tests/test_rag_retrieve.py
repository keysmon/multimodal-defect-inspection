import math

import pytest

from defectlens.corpus import Card
from defectlens.eval.rag_recall import per_class_recall, recall_at_k
from defectlens.rag.retrieve import Hit, card_lookup, hits_from_rows


def make_card(cid, tags):
    return Card(
        id=cid, title="t", class_tags=tags, severity="monitor",
        index_sentence="s", passage="p", citation="c",
        source_name="n", source_url="u", source_license="l",
    )


def test_hits_from_rows_joins_and_raises_on_drift():
    lookup = card_lookup([make_card("a", ["crack"])])
    hits = hits_from_rows([("a", ["crack"], 0.1)], lookup)
    assert hits[0].card.id == "a" and hits[0].distance == 0.1
    with pytest.raises(KeyError, match="re-run indexing"):
        hits_from_rows([("ghost", [], 0.2)], lookup)


def test_recall_at_k():
    results = [
        ("crack", [["spalling"], ["crack"]]),   # relevant at rank 2
        ("crack", [["spalling"], ["no_defect"]]),  # miss
    ]
    assert recall_at_k(results, k=2) == 0.5
    assert recall_at_k(results, k=1) == 0.0
    assert math.isnan(recall_at_k([], k=5))


def test_per_class_recall():
    results = [
        ("crack", [["crack"]]),
        ("spalling", [["crack"]]),
    ]
    per = per_class_recall(results, ["crack", "spalling", "no_defect"], k=1)
    assert per["crack"] == 1.0
    assert per["spalling"] == 0.0
    assert math.isnan(per["no_defect"])
