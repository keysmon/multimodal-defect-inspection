import numpy as np
import pytest

from defectlens.audio.anomaly import KNNAnomalyScorer


def _cluster(center, n=50, seed=0, scale=0.01):
    rng = np.random.default_rng(seed)
    embs = center + rng.normal(0, scale, size=(n, len(center)))
    return embs / np.linalg.norm(embs, axis=1, keepdims=True)


def test_far_points_score_higher_than_near_points():
    normal = _cluster(np.array([1.0, 0.0, 0.0]))
    scorer = KNNAnomalyScorer(k=5).fit(normal)
    near = _cluster(np.array([1.0, 0.0, 0.0]), n=10, seed=1)
    far = _cluster(np.array([0.0, 1.0, 0.0]), n=10, seed=2)
    assert scorer.score(far).min() > scorer.score(near).max()


def test_k_larger_than_fit_set_is_capped():
    normal = _cluster(np.array([1.0, 0.0, 0.0]), n=3)
    scorer = KNNAnomalyScorer(k=10).fit(normal)
    assert np.isfinite(scorer.score(normal)).all()


def test_score_before_fit_raises():
    with pytest.raises(RuntimeError):
        KNNAnomalyScorer().score(np.zeros((1, 3)))
