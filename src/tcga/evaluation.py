"""Multiclass evaluation for Task 0 (registered deviation from the binary protocol).
Same pipeline (StandardScaler + L2 LogisticRegression, multinomial by sklearn default), same per-seed OOF
aggregation; metrics: macro one-vs-rest AUROC (+ per class), balanced accuracy, macro F1, accuracy."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             recall_score, roc_auc_score)

from src.models.slide_classifier import build_classifier


def fit_predict_multi(cfg_clf: dict, X_tr, y_tr, X_te) -> np.ndarray:
    clf = build_classifier(cfg_clf)
    clf.fit(X_tr, y_tr)
    return clf.predict_proba(X_te)            # [n_te, n_classes], columns = clf.classes_ (sorted labels)


def multiclass_metrics(y_true: np.ndarray, P: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int); P = np.asarray(P, dtype=float)
    pred = P.argmax(1)
    if P.shape[1] == 2:   # binary task: ROC-AUC on p(class 1), positive-class F1, PR-AUC, per-class recall
        p1 = P[:, 1]
        cm = confusion_matrix(y_true, pred, labels=[0, 1])
        return dict(auroc=float(roc_auc_score(y_true, p1)), balanced_accuracy=float(balanced_accuracy_score(y_true, pred)),
                    f1=float(f1_score(y_true, pred)), accuracy=float(accuracy_score(y_true, pred)), pr_auc=float(average_precision_score(y_true, p1)),
                    recall_class0=float(recall_score(y_true, pred, pos_label=0)), recall_class1=float(recall_score(y_true, pred, pos_label=1)),
                    tn=int(cm[0, 0]), fp=int(cm[0, 1]), fn=int(cm[1, 0]), tp=int(cm[1, 1]))
    out = dict(auroc=float(roc_auc_score(y_true, P, multi_class="ovr", average="macro")),
               balanced_accuracy=float(balanced_accuracy_score(y_true, pred)),
               f1=float(f1_score(y_true, pred, average="macro")), accuracy=float(accuracy_score(y_true, pred)))
    for c in range(P.shape[1]):
        out[f"auroc_class{c}"] = float(roc_auc_score((y_true == c).astype(int), P[:, c]))
    return out


def summarise_multi(pred: pd.DataFrame, n_classes: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = [f"p{c}" for c in range(n_classes)]
    rows = []
    for (m, seed), g in pred.groupby(["method", "seed"]):
        met = multiclass_metrics(g.true_label.values, g[cols].values)
        rows.append(dict(method=m, seed=seed, **met, mean_patches_used=float(g.num_selected.mean())))
    per_run = pd.DataFrame(rows)
    agg = per_run.groupby("method").agg(["mean", "std"]); agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    return agg.reset_index(), per_run


def patient_bootstrap_ci(pred: pd.DataFrame, n_classes: int, n_boot: int = 1000, seed: int = 0) -> dict[str, tuple[float, float]]:
    """95% CI of macro AUROC per method, resampling patients (slides) with replacement; each resampled patient
    contributes all of its per-seed OOF predictions, and the AUROC is averaged over seeds inside each replicate."""
    rng = np.random.default_rng(seed); cols = [f"p{c}" for c in range(n_classes)]
    out = {}
    for m, g in pred.groupby("method"):
        sids = g.slide_id.unique(); by_seed = {s: gg.set_index("slide_id") for s, gg in g.groupby("seed")}
        vals = []
        for _ in range(n_boot):
            samp = rng.choice(sids, size=len(sids), replace=True)
            aucs = []
            for s, gg in by_seed.items():
                yy = gg.loc[samp, "true_label"].values; PP = gg.loc[samp, cols].values
                if len(np.unique(yy)) < n_classes:
                    continue
                aucs.append(roc_auc_score(yy, PP[:, 1]) if n_classes == 2 else roc_auc_score(yy, PP, multi_class="ovr", average="macro"))
            if aucs:
                vals.append(np.mean(aucs))
        out[m] = (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))
    return out
