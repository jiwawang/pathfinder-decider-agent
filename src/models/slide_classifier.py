"""Lightweight slide classifier (README section 13): standardise -> optional PCA -> L2 logistic regression.

Every preprocessing step is inside one sklearn Pipeline, so it is fit on the training fold only.
"""
from __future__ import annotations

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def build_classifier(cfg: dict) -> Pipeline:
    steps = []
    if cfg.get("standardize", True):
        steps.append(("scaler", StandardScaler()))
    n_pca = int(cfg.get("pca_components", 0) or 0)
    if n_pca > 0:
        steps.append(("pca", PCA(n_components=n_pca, random_state=0)))
    kind = cfg.get("type", "logistic_regression")
    if kind == "logistic_regression":
        steps.append(("clf", LogisticRegression(C=float(cfg.get("C", 1.0)),
                                                max_iter=5000, class_weight="balanced")))
    elif kind == "linear_svm":
        steps.append(("clf", SVC(kernel="linear", C=float(cfg.get("C", 1.0)), probability=True,
                                 class_weight="balanced", random_state=0)))
    else:
        raise ValueError(f"unknown classifier type {kind}")
    return Pipeline(steps)
