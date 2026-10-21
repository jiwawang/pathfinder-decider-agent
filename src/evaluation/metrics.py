"""Slide-level metrics (README section 16)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score


def slide_metrics(y_true: np.ndarray, p1: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    p1 = np.asarray(p1, dtype=float)
    pred = (p1 >= threshold).astype(int)
    return dict(
        auroc=float(roc_auc_score(y_true, p1)) if len(np.unique(y_true)) == 2 else float("nan"),
        balanced_accuracy=float(balanced_accuracy_score(y_true, pred)),
        f1=float(f1_score(y_true, pred, zero_division=0)),
        accuracy=float(accuracy_score(y_true, pred)),
    )
