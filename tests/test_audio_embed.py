import numpy as np

from defectlens.audio.embed import CLAP_MODEL, batched


def test_batched_covers_all_items_in_order():
    items = list(range(10))
    batches = list(batched(items, size=4))
    assert batches == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]


def test_clap_model_constant():
    assert CLAP_MODEL == "laion/clap-htsat-unfused"
