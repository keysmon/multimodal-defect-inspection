import numpy as np

from defectlens.ingest import ManifestRow
from defectlens.rag.embed import normalize, sample_train_rows, tag_centroid


def rows(label, n):
    return [ManifestRow(f"data/raw/d/{label}/{i}.jpg", "d", label, label) for i in range(n)]


def test_normalize_1d_and_2d():
    v = normalize(np.array([3.0, 4.0]))
    assert np.allclose(np.linalg.norm(v), 1.0)
    m = normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert np.allclose(np.linalg.norm(m[0]), 1.0)
    assert np.allclose(m[1], 0.0)  # zero vector stays zero, no NaN


def test_sample_train_rows_deterministic_and_capped():
    data = rows("crack", 50) + rows("spalling", 3)
    a = sample_train_rows(data, per_class_cap=10, seed=7)
    b = sample_train_rows(data, per_class_cap=10, seed=7)
    assert a == b
    assert len(a["crack"]) == 10 and len(a["spalling"]) == 3


def test_sample_stable_when_other_class_added():
    base = rows("crack", 50)
    more = base + rows("algae", 20)
    a = sample_train_rows(base, per_class_cap=10, seed=7)
    b = sample_train_rows(more, per_class_cap=10, seed=7)
    assert a["crack"] == b["crack"]


def test_tag_centroid_mean_and_norm():
    c = {"a": np.array([1.0, 0.0]), "b": np.array([0.0, 1.0])}
    v = tag_centroid(c, ["a", "b"])
    assert np.allclose(v, np.array([1.0, 1.0]) / np.sqrt(2))
    assert np.allclose(tag_centroid(c, ["a"]), [1.0, 0.0])
