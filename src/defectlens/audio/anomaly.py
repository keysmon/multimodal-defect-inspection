"""k-NN anomaly scoring over normalized embeddings (DCASE unsupervised protocol).

Fit on NORMAL clips only; score = mean cosine distance to the k nearest
normal embeddings. Higher = more anomalous. This embeddings+density shape is
the modern replacement for the DCASE AE-reconstruction baseline.
"""
from __future__ import annotations

import numpy as np


class KNNAnomalyScorer:
    def __init__(self, k: int = 5) -> None:
        self.k = k
        self._bank: np.ndarray | None = None

    def fit(self, normal_embeddings: np.ndarray) -> "KNNAnomalyScorer":
        self._bank = np.asarray(normal_embeddings, dtype=np.float32)
        return self

    def score(self, embeddings: np.ndarray) -> np.ndarray:
        if self._bank is None:
            raise RuntimeError("fit() before score()")
        emb = np.asarray(embeddings, dtype=np.float32)
        # cosine distance on L2-normalized vectors: 1 - dot
        sims = emb @ self._bank.T                     # [n, bank]
        k = min(self.k, self._bank.shape[0])
        top = np.sort(sims, axis=1)[:, -k:]           # k most-similar normals
        return 1.0 - top.mean(axis=1)
