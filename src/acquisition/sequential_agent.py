"""One slide = one acquisition episode (WP001 sections 8-13).

    cheap pool -> shortlist (A-D) -> router (Decider) -> ACQUIRE one patch (UNI call) -> update state -> ...

The agent state only ever contains: budget counters, coverage statistics, redundancy among the
embeddings that were ACQUIRED, and cheap stats of the shortlisted candidates. No label, no UNI
information about unacquired patches.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from decision_patterns.adaptive_expensive_observation_router import STOP, AdaptiveObservationRouter, Option
from src.acquisition.lazy_uni_store import AcquisitionBudget
from src.acquisition.shortlist import ShortlistBuilder


def _pairwise_mean_cos(E: list[np.ndarray]) -> float:
    if len(E) < 2:
        return 0.0
    M = np.stack(E)
    M = M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-8)
    S = M @ M.T
    return float(S[np.triu_indices(len(E), 1)].mean())


def acquisition_state(budget: AcquisitionBudget, pool: dict, acquired_pool_idx: list[int], embeddings: list[np.ndarray],
                      cfg: dict) -> dict[str, Any]:
    xy = np.stack([pool["norm_x"], pool["norm_y"]], 1)
    if acquired_pool_idx:
        d = np.linalg.norm(xy[:, None, :] - xy[acquired_pool_idx][None, :, :], axis=-1).min(1)
        spatial_cov = float((d <= float(cfg["spatial_coverage_radius"])).mean())
    else:
        spatial_cov = 0.0
    sim = _pairwise_mean_cos(embeddings)
    return {
        "uni_patches_acquired": budget.uni_calls,
        "minimum_budget": budget.min_calls, "target_budget": budget.target_calls, "maximum_budget": budget.max_calls,
        "cheap_clusters_covered": len(set(int(pool["cluster_id"][i]) for i in acquired_pool_idx)),
        "total_cheap_clusters": int(pool["pool_size"]),
        "spatial_coverage_percent": round(100 * spatial_cov, 1),
        "mean_pairwise_uni_similarity_of_acquired": round(sim, 3),
        "target_budget_reached": bool(budget.target_reached),
        "acquired_evidence_is_diverse": bool(len(embeddings) >= 2 and sim < float(cfg["diverse_similarity_threshold"])),
    }


def run_acquisition(slide_id: str, pool: dict, store, router: AdaptiveObservationRouter, cfg: dict,
                    *, shortlist_size: int = 4, seed: int = 0) -> dict[str, Any]:
    budget = AcquisitionBudget(router.min_calls, router.target_calls, router.max_calls)
    sb = ShortlistBuilder(pool)
    acquired_pool_idx: list[int] = []
    embeddings: list[np.ndarray] = []
    trace: list[dict[str, Any]] = []
    stopped_by_model = False
    rnd = 0
    while True:
        options = sb.build(acquired_pool_idx, shortlist_size)
        state = {"acquisition_state": acquisition_state(budget, pool, acquired_pool_idx, embeddings, cfg),
                 "candidates": {o.id: {**sb.stats(o.payload, acquired_pool_idx), "why_shortlisted": o.description.split(";")[0],
                                       "exploration_score": round(o.score, 3)} for o in options}}
        d = router.decide(state, options, budget.uni_calls)
        rec = {"slide_id": slide_id, "round": rnd, "shortlist": {o.id: int(pool["patch_index"][o.payload]) for o in options},
               "state": state, "options_shown": d.options, "decision": d.choice, "raw_decision": d.raw_choice,
               "overridden_by": d.overridden_by, "allowed": list(d.allowed), "option_probabilities": d.probabilities,
               "confidence": d.confidence, "halt_probabilities": d.halt_probabilities, "halt_confidence": d.halt_confidence,
               "decider_latency_ms": round(d.latency_ms, 3),
               "model_queried": d.overridden_by != "max_budget" and len(d.options) >= 2}
        if d.choice == STOP:
            rec["uni_call"] = None
            trace.append(rec)
            stopped_by_model = d.overridden_by is None
            break
        pick = next(o for o in options if o.id == d.choice)
        patch_index = int(pool["patch_index"][pick.payload])
        emb = store.acquire(slide_id, patch_index)          # the ONLY place a UNI embedding is revealed
        budget.record(patch_index)
        acquired_pool_idx.append(int(pick.payload))
        embeddings.append(emb)
        rec["uni_call"] = budget.uni_calls
        rec["acquired_patch_index"] = patch_index
        rec["acquired_coord"] = [int(pool["coord_x"][pick.payload]), int(pool["coord_y"][pick.payload])]
        trace.append(rec)
        rnd += 1
        if budget.must_stop or not sb.build(acquired_pool_idx, 1):
            break
    lat = [t["decider_latency_ms"] for t in trace if t["model_queried"]]
    conf = [t["confidence"] for t in trace if t["model_queried"]]
    return {
        "slide_id": slide_id, "acquired_patch_index": budget.acquired_patch_ids, "acquired_pool_index": acquired_pool_idx,
        "embedding": np.stack(embeddings).astype(np.float32) if embeddings else np.zeros((0, 1)),
        "trace": trace,
        "stats": {"uni_calls": budget.uni_calls, "rounds": len(trace), "stopped_by_model": stopped_by_model,
                  "clusters_covered": len(set(int(pool["cluster_id"][i]) for i in acquired_pool_idx)),
                  "final_mean_pairwise_similarity": _pairwise_mean_cos(embeddings),
                  "mean_decider_latency_ms": float(np.mean(lat)) if lat else 0.0,
                  "mean_confidence": float(np.mean(conf)) if conf else 0.0,
                  "low_confidence_rate": float(np.mean([c < router.low_conf for c in conf])) if conf else 0.0,
                  "choice_counts": {k: sum(t["decision"] == k for t in trace) for k in ("A", "B", "C", "D", STOP)}},
    }


# ----------------------------------------------------------------------------------------------
# Fixed-budget baselines. Every one of them goes through store.acquire, so UNI calls are counted
# identically and nothing can peek at unacquired embeddings.
# ----------------------------------------------------------------------------------------------

def acquire_indices(slide_id: str, store, patch_indices: list[int]) -> np.ndarray:
    return np.stack([store.acquire(slide_id, int(i)) for i in patch_indices]).astype(np.float32)


def random_patches(n_patches: int, k: int, seed: int) -> list[int]:
    return np.random.default_rng(seed).choice(n_patches, size=min(k, n_patches), replace=False).tolist()


def uniform_spatial_patches(coords: np.ndarray, k: int, seed: int = 0) -> list[int]:
    """K-means on patch coordinates -> the real patch nearest each centre (even spatial coverage)."""
    from sklearn.cluster import KMeans
    k = min(k, len(coords))
    km = KMeans(n_clusters=k, random_state=seed, n_init=1).fit(coords.astype(np.float32))
    out = []
    for c in km.cluster_centers_:
        out.append(int(np.argmin(np.linalg.norm(coords - c, axis=1))))
    return sorted(set(out))


def cheap_cluster_topk(pool: dict, k: int) -> list[int]:
    """Largest cheap clusters' representatives (non-sequential cheap heuristic)."""
    order = np.argsort(-pool["cluster_fraction"], kind="stable")[:k]
    return [int(pool["patch_index"][i]) for i in order]


def cheap_greedy(pool: dict, k: int) -> list[int]:
    """Sequential cheap heuristic: each step takes the highest unified exploration score (novelty +
    spatial distance + cluster size + content). No model, no UNI information."""
    sb = ShortlistBuilder(pool)
    acquired: list[int] = []
    for _ in range(min(k, int(pool["pool_size"]))):
        remaining = [i for i in range(int(pool["pool_size"])) if i not in acquired]
        acquired.append(max(remaining, key=lambda i: sb.exploration_score(i, acquired)))
    return [int(pool["patch_index"][i]) for i in acquired]
