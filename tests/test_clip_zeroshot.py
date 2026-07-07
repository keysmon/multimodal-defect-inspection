import numpy as np

from defectlens.eval.clip_zeroshot import _nan_to_none, expand_prompts, rank_from_similarity


def test_nan_to_none():
    assert _nan_to_none(float("nan")) is None
    assert _nan_to_none(0.5) == 0.5


def test_expand_prompts():
    phrases = {"crack": "a crack", "no_defect": "a clean wall"}
    templates = ["a photo of {}", "{}"]
    prompts = expand_prompts(phrases, templates)
    assert prompts["crack"] == ["a photo of a crack", "a crack"]
    assert prompts["no_defect"] == ["a photo of a clean wall", "a clean wall"]


def test_rank_from_similarity():
    classes = ["a", "b", "c"]
    # image 0 most similar to c, then a, then b
    sims = np.array([[0.2, 0.1, 0.9], [0.8, 0.7, 0.1]])
    ranked = rank_from_similarity(sims, classes)
    assert ranked[0] == ["c", "a", "b"]
    assert ranked[1] == ["a", "b", "c"]
