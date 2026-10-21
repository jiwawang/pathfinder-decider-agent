"""Leakage-safe CV driver shared by run_cv.py (README sections 7, 14, 15).

For every (seed, fold):
  1. training-fold candidate prototypes -> KNN patch bank
  2. votes: test slides use the full training bank; training slides use bank minus themselves
  3. decider episodes for every slide -> evidence set -> mean-pooled slide embedding
  4. logistic regression fit on training slides only -> out-of-fold prediction for test slides
  5. baselines computed in the same fold with the same classifier
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.aggregation.pooling import mean_pool
from src.evaluation.splits import make_folds
from src.models.slide_classifier import build_classifier
from src.voting.knn_vote import PatchBank


def load_candidates(cand_dir: Path, slide_ids: list[str]) -> dict[str, dict]:
    out = {}
    for sid in slide_ids:
        z = np.load(cand_dir / f"{sid}.npz", allow_pickle=True)
        out[sid] = {k: z[k] for k in z.files}
    return out


def fit_predict(cfg_clf: dict, X_tr, y_tr, X_te) -> np.ndarray:
    clf = build_classifier(cfg_clf)
    clf.fit(X_tr, y_tr)
    return clf.predict_proba(X_te)[:, 1]


def run_fold(meta: pd.DataFrame, cands: dict, pools: dict | None, tr: np.ndarray, te: np.ndarray, *,
             cfg: dict, seed: int, fold: int, episode_fn: Callable[[str, dict, dict, bool], dict] | None,
             methods: list[str], trace_dir: Path | None) -> tuple[list[dict], list[dict]]:
    """Return (prediction rows, decider stat rows) for this fold."""
    sids = meta["slide_id"].tolist()
    y = meta["label"].values.astype(int)
    train_ids = [sids[i] for i in tr]
    train_set = set(train_ids)
    bank = PatchBank.from_candidates(cands, train_ids, int(cfg["knn"]["k"]))

    votes, episodes, stat_rows = {}, {}, []
    for i, sid in enumerate(sids):
        votes[sid] = bank.vote(cands[sid]["embedding"], exclude_slide_id=sid if sid in train_set else None)
    if "decider" in methods:
        assert episode_fn is not None
        for sid in sids:
            cached = trace_dir / f"{sid}.json" if trace_dir else None
            if cached and cached.exists():
                ep = json.loads(cached.read_text())
            else:
                ep = episode_fn(sid, cands[sid], votes[sid], sid in train_set)
                if cached:
                    cached.parent.mkdir(parents=True, exist_ok=True)
                    cached.write_text(json.dumps(ep, indent=1))
            episodes[sid] = ep
            stat_rows.append({"seed": seed, "fold": fold, "slide_id": sid, "is_test": sid not in train_set,
                              **{k: v for k, v in ep["stats"].items() if k != "action_counts"},
                              **{f"n_{a.lower()}": c for a, c in ep["stats"]["action_counts"].items()}})
        k_fold = int(round(np.mean([episodes[s]["stats"]["num_selected"] for s in train_ids])))
    else:
        k_fold = int(cfg["decider"]["target_selected"])
    k_fold = max(1, k_fold)

    def embed(method: str, sid: str, rep: int = 0) -> tuple[np.ndarray, int, int]:
        c, v = cands[sid], votes[sid]
        n_cand = len(c["patch_index"])
        if method == "all_mean":
            return c["all_patch_mean"], int(c["n_patches"]), int(c["n_patches"])
        if method == "candidates_mean":
            return mean_pool(c["embedding"]), n_cand, n_cand
        if method == "kmeans_topk":
            idx = np.argsort(-c["cluster_fraction"], kind="stable")[:k_fold]
            return mean_pool(c["embedding"][idx]), len(idx), n_cand
        if method == "vote_topk":
            idx = np.argsort(-v["vote_confidence"], kind="stable")[:k_fold]
            return mean_pool(c["embedding"][idx]), len(idx), n_cand
        if method == "random_k":
            p = pools[sid]
            rng = np.random.default_rng(seed * 1000 + fold * 10 + rep)
            idx = rng.choice(len(p["embedding"]), size=min(k_fold, len(p["embedding"])), replace=False)
            return mean_pool(p["embedding"][idx]), len(idx), int(p["n_patches"])
        if method == "decider":
            ep = episodes[sid]
            return mean_pool(c["embedding"][ep["selected_indices"]]), ep["stats"]["num_selected"], ep["stats"]["num_examined"]
        raise ValueError(method)

    rows = []
    for method in methods:
        reps = int(cfg["evaluation"].get("random_k_repeats", 1)) if method == "random_k" else 1
        for rep in range(reps):
            feats = {sid: embed(method, sid, rep) for sid in sids}
            X = np.stack([feats[s][0] for s in sids])
            p = fit_predict(cfg["classifier"], X[tr], y[tr], X[te])
            for j, i in enumerate(te):
                sid = sids[i]
                rows.append(dict(method=method, seed=seed, fold=fold, rep=rep, slide_id=sid,
                                 true_label=int(y[i]), pred_probability=float(p[j]),
                                 pred_label=int(p[j] >= 0.5), num_selected=int(feats[sid][1]),
                                 num_examined=int(feats[sid][2]), k_fold=k_fold))
    return rows, stat_rows


def summarise(pred: pd.DataFrame, metric_fn) -> pd.DataFrame:
    """Per-method metrics computed on each (seed, rep) OOF set, then mean ± std across them."""
    out = []
    for (m, seed, rep), g in pred.groupby(["method", "seed", "rep"]):
        met = metric_fn(g["true_label"].values, g["pred_probability"].values)
        out.append(dict(method=m, seed=seed, rep=rep, **met,
                        mean_patches_used=float(g["num_selected"].mean()),
                        mean_examined=float(g["num_examined"].mean())))
    per_run = pd.DataFrame(out)
    agg = per_run.groupby("method").agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    return agg.reset_index(), per_run
