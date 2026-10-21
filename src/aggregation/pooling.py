"""Selected patches -> slide embedding (README section 12). MVP = plain mean."""
from __future__ import annotations

import numpy as np


def mean_pool(embeddings: np.ndarray) -> np.ndarray:
    if len(embeddings) == 0:
        raise ValueError("cannot pool an empty evidence set")
    return np.asarray(embeddings, dtype=np.float32).mean(axis=0)


def weighted_mean_pool(embeddings: np.ndarray, select_prob: np.ndarray, vote_conf: np.ndarray) -> np.ndarray:
    """Ablation only: weight = p(SELECT) * (0.5 + 0.5 * vote_confidence)."""
    w = np.asarray(select_prob, dtype=np.float32) * (0.5 + 0.5 * np.asarray(vote_conf, dtype=np.float32))
    w = w / max(float(w.sum()), 1e-8)
    return (np.asarray(embeddings, dtype=np.float32) * w[:, None]).sum(axis=0)
