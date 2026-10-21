"""Leakage-safe cross-validation splits (README sections 7 and 14).

Patient-level grouping is always used (StratifiedGroupKFold with groups=patient_id). When every
patient has exactly one slide this degenerates to ordinary StratifiedKFold, which is what the
cohort currently is, but the grouping is kept so additional slides per patient can never leak.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def make_folds(meta: pd.DataFrame, n_folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    y = meta["label"].values.astype(int)
    groups = meta["patient_id"].astype(str).values
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = list(sgkf.split(np.zeros(len(y)), y, groups))
    for tr, te in folds:   # hard assertion: no patient on both sides
        assert not set(groups[tr]) & set(groups[te]), "patient leakage across folds"
    return folds
