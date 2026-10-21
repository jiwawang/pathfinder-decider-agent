"""V2.1 (docs/WP/WP002.md): target-free, information-gain driven sequential acquisition.

Per round, two typed questions about the same state (one model call):
  halt   "Would another expensive UNI observation still add meaningful new evidence?"  CONTINUE / STOP
  route  "Which candidate should receive the next UNI inference?"                    A / B / C / D
STOP ends the episode; otherwise the routed candidate is acquired. The state carries NO target or
preferred count: only the calls used, the hard maximum, evidence-saturation statistics (information
gain of each acquisition and its recent trend, pairwise similarity, coverage) and the cheap stats of
the shortlist. Option letters are shuffled every round (seeded) so position bias can be measured.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from decision_patterns.adaptive_expensive_observation_router import STOP, Option
from src.acquisition.lazy_uni_store import AcquisitionBudget
from src.acquisition.shortlist import STRATEGIES, ShortlistBuilder

STRATEGY_KEYS = {"A": "largest_cluster", "B": "cheap_novelty", "C": "spatial_far", "D": "undercovered_region"}
FORBIDDEN_STATE_TOKENS = ("target", "expected_count", "preferred", "label", "responder", "effective", "invalid", "vote")

# Halting question (WP002 sections 7-8). Measured on 2,212 real states: a 5-way choice never stops; a
# CONTINUE/STOP question with a categorical "trend" word in the state stops on the word alone; a question
# that asks the model to COMPARE recent gains with the initial ones, over numbers only, tracks the gain
# ratio (corr -0.86) and yields slide-specific stopping. No threshold is coded anywhere.
HALT_QUESTION = ("Is the evidence from expensive UNI observations saturating? Compare the recent information gains "
                 "with the initial ones.")
HALT_CRITERIA = {
    "CONTINUE": "not saturating: recent gains are comparable to the initial gains",
    "STOP": "saturating: recent gains are much lower than the initial gains",
}
ROUTE_QUESTION = ("Which candidate should receive the next expensive UNI inference? Prefer a candidate that is novel and "
                  "far from the evidence acquired so far, representative of a large unexplored tissue region, and rich in "
                  "tissue content.")


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def information_gain(new: np.ndarray, previous: list[np.ndarray]) -> float:
    """1 - max cosine(new, previous). First acquisition -> 1.0."""
    if not previous:
        return 1.0
    P = _unit(np.stack(previous))
    return float(1.0 - np.max(P @ _unit(new)))


def gain_window_stats(gains: list[float], window: int) -> dict[str, Any]:
    """Numeric saturation statistics only (no categorical trend word: the model keys on such labels)."""
    if not gains:
        return {"information_gain_of_each_acquisition_so_far": [], "latest_information_gain": None, "recent_information_gains": [],
                "recent_gain_mean": None, "recent_gain_min": None, "recent_gain_max": None, "recent_gain_slope": None,
                "average_gain_of_first_acquisitions": None, "average_gain_of_last_acquisitions": None,
                "recent_gain_as_percent_of_initial_gains": None}
    g = np.array(gains[-window:], dtype=float)
    init = float(np.mean(gains[1:1 + window])) if len(gains) > 1 else 1.0   # first acquisition (gain 1.0) excluded
    slope = float(np.polyfit(np.arange(len(g)), g, 1)[0]) if len(g) >= 2 else 0.0
    return {"information_gain_of_each_acquisition_so_far": [round(float(v), 2) for v in gains],
            "latest_information_gain": round(float(g[-1]), 3), "recent_information_gains": [round(float(v), 3) for v in g],
            "recent_gain_mean": round(float(g.mean()), 3), "recent_gain_min": round(float(g.min()), 3),
            "recent_gain_max": round(float(g.max()), 3), "recent_gain_slope": round(slope, 4),
            "average_gain_of_first_acquisitions": round(init, 3), "average_gain_of_last_acquisitions": round(float(g.mean()), 3),
            "recent_gain_as_percent_of_initial_gains": round(100.0 * float(g.mean()) / max(init, 1e-6), 1)}


def _pairwise(E: list[np.ndarray]) -> tuple[float, float]:
    if len(E) < 2:
        return 0.0, 1.0
    M = _unit(np.stack(E)); S = M @ M.T
    v = S[np.triu_indices(len(E), 1)]
    return float(v.mean()), float(1.0 - v.mean())


def saturation_state(budget: AcquisitionBudget, pool: dict, acquired_pool_idx: list[int], embeddings: list[np.ndarray],
                     gains: list[float], cfg: dict) -> dict[str, Any]:
    window = int(cfg["information_gain_window"])
    xy = np.stack([pool["norm_x"], pool["norm_y"]], 1)
    if acquired_pool_idx:
        d = np.linalg.norm(xy[:, None, :] - xy[acquired_pool_idx][None, :, :], axis=-1).min(1)
        spatial_cov = float((d <= float(cfg["spatial_coverage_radius"])).mean())
    else:
        spatial_cov = 0.0
    sim, dist = _pairwise(embeddings)
    clusters = [int(pool["cluster_id"][i]) for i in acquired_pool_idx]
    seen_before = set(clusters[:-window]) if len(clusters) > window else set()
    new_last_n = len(set(clusters[-window:]) - seen_before) if clusters else 0
    st = {"uni_calls_used": budget.uni_calls, "maximum_uni_calls": budget.max_calls,
          "remaining_candidate_count": int(pool["pool_size"]) - len(acquired_pool_idx),
          "mean_pairwise_uni_similarity": round(sim, 3), "mean_pairwise_uni_distance": round(dist, 3),
          **gain_window_stats(gains, window),
          f"new_cheap_clusters_found_in_last_{window}": new_last_n,
          "cheap_cluster_coverage_percent": round(100 * len(set(clusters)) / int(pool["pool_size"]), 1),
          "spatial_coverage_percent": round(100 * spatial_cov, 1)}
    return st


class AdaptiveRouterV21:
    """Target-free router: halt question + routing question per round, STOP gated only by min/max."""

    def __init__(self, backend, *, min_calls: int, max_calls: int, low_confidence_threshold: float = 0.15,
                 halt_question: str = HALT_QUESTION, halt_criteria: dict | None = None, route_question: str = ROUTE_QUESTION):
        self.backend = backend
        self.min_calls, self.max_calls, self.low_conf = min_calls, max_calls, low_confidence_threshold
        self.halt_question, self.halt_criteria, self.route_question = halt_question, dict(halt_criteria or HALT_CRITERIA), route_question

    def decide(self, state: dict, options: list[Option], calls_used: int) -> dict[str, Any]:
        out = {"allowed_stop": self.min_calls <= calls_used < self.max_calls, "forced_stop": calls_used >= self.max_calls or not options}
        if out["forced_stop"]:
            return {**out, "decision": STOP, "halt_probabilities": {STOP: 1.0}, "halt_confidence": 1.0, "route_choice": None,
                    "route_probabilities": {}, "route_confidence": None, "overridden_by": "max_budget", "latency_ms": 0.0}
        shown = {o.id: o.description for o in options}
        import time
        t0 = time.perf_counter()
        ans = {}
        # Step 1 - halting, judged on the saturation state ALONE (WP002 section 3). Letting the halting
        # question see the candidate shortlist made the model markedly more reluctant to stop.
        if out["allowed_stop"]:
            halt_state = {"expensive_observation_state": state["expensive_observation_state"]}
            ans["halt"] = self.backend.choose_many(halt_state, {"halt": (self.halt_question, self.halt_criteria)})["halt"]
        # Step 2 - routing, only if we continue.
        if len(shown) >= 2 and not (ans.get("halt") and ans["halt"][0] == STOP and (ans["halt"][2] is None or ans["halt"][2] >= self.low_conf)):
            ans["route"] = self.backend.choose_many(state, {"route": (self.route_question, shown)})["route"]
        out["latency_ms"] = (time.perf_counter() - t0) * 1000.0
        halt_choice, hp, hc = ans.get("halt", ("CONTINUE", {}, None))
        if hc is not None and hc < self.low_conf:
            halt_choice = "CONTINUE"
        out.update(halt_probabilities=hp, halt_confidence=hc)
        if halt_choice == STOP:
            return {**out, "decision": STOP, "route_choice": None, "route_probabilities": {}, "route_confidence": None, "overridden_by": None}
        if "route" in ans:
            rc, rp, rconf = ans["route"]
        elif len(shown) >= 2:   # halt said STOP below confidence -> continue; route now
            rc, rp, rconf = self.backend.choose_many(state, {"route": (self.route_question, shown)})["route"]
        else:
            rc, rp, rconf = next(iter(shown)), {next(iter(shown)): 1.0}, 1.0
        final, overridden = rc, None
        if rconf < self.low_conf:
            best = max(options, key=lambda o: o.score)
            final, overridden = best.id, ("low_confidence" if best.id != rc else None)
        return {**out, "decision": final, "route_choice": rc, "route_probabilities": rp, "route_confidence": rconf, "overridden_by": overridden}


class SaturationThresholdRouter:
    """Deterministic baseline (WP002 section 11): cheap-score routing; STOP when the mean of the last
    `window` information gains is below `threshold` (threshold chosen on the training fold only)."""

    def __init__(self, threshold: float):
        self.threshold = float(threshold)

    def choose_many(self, state, questions):
        out = {}
        s = state["expensive_observation_state"]
        for name, (q, opts) in questions.items():
            if name == "halt":
                m = s.get("recent_gain_mean")
                c = STOP if (m is not None and m < self.threshold) else "CONTINUE"
                out[name] = (c, {k: (0.9 if k == c else 0.1) for k in opts}, 0.8)
            else:
                cands = state["candidates"]
                c = max(opts, key=lambda k: cands[k]["exploration_score"])
                out[name] = (c, {k: (0.9 if k == c else 0.1 / max(1, len(opts) - 1)) for k in opts}, 0.85)
        return out


def run_adaptive_episode(slide_id: str, pool: dict, store, router: AdaptiveRouterV21, cfg: dict, *, shuffle_seed: int = 0,
                         shortlist_size: int = 4) -> dict[str, Any]:
    budget = AcquisitionBudget(router.min_calls, router.min_calls, router.max_calls)
    sb = ShortlistBuilder(pool)
    rng = np.random.default_rng(shuffle_seed)
    acquired: list[int] = []
    embeddings: list[np.ndarray] = []
    gains: list[float] = []
    trace: list[dict[str, Any]] = []
    stopped_by_model = False
    rnd = 0
    while True:
        raw = sb.build(acquired, shortlist_size)            # options labelled by STRATEGY letter from shortlist.py
        strategies = [STRATEGY_KEYS[o.id] for o in raw]
        perm = rng.permutation(len(raw))                    # shuffle positions
        letters = [chr(ord("A") + i) for i in range(len(raw))]
        options, position_to_strategy = [], {}
        for pos, src in enumerate(perm):
            o = raw[src]
            desc = o.description.split(";", 1)[1].strip() if ";" in o.description else o.description
            options.append(Option(letters[pos], f"{STRATEGIES[o.id]}; {desc}", payload=o.payload, score=o.score))
            position_to_strategy[letters[pos]] = strategies[src]
        sat = saturation_state(budget, pool, acquired, embeddings, gains, cfg)
        state = {"expensive_observation_state": sat,
                 "candidates": {o.id: {**sb.stats(o.payload, acquired), "why_shortlisted": STRATEGIES[raw[perm[i]].id],
                                       "exploration_score": round(o.score, 3)} for i, o in enumerate(options)}}
        d = router.decide(state, options, budget.uni_calls)
        rec = {"slide_id": slide_id, "round": rnd, "uni_calls": budget.uni_calls, "state": sat,
               "recent_information_gains": sat["recent_information_gains"], "recent_gain_mean": sat["recent_gain_mean"],
               "recent_gain_percent_of_initial": sat["recent_gain_as_percent_of_initial_gains"],
               "recent_gain_slope": sat["recent_gain_slope"], "mean_pairwise_uni_similarity": sat["mean_pairwise_uni_similarity"],
               "cheap_cluster_coverage": sat["cheap_cluster_coverage_percent"] / 100, "spatial_coverage": sat["spatial_coverage_percent"] / 100,
               "stop_allowed": d["allowed_stop"], "stop_probability": d["halt_probabilities"].get(STOP) if d["halt_probabilities"] else None,
               "halt_confidence": d["halt_confidence"], "options": position_to_strategy,
               "candidates": state["candidates"], "selected_option": None, "selected_strategy": None,
               "route_probabilities": d["route_probabilities"], "route_confidence": d["route_confidence"],
               "overridden_by": d["overridden_by"], "decision": d["decision"], "latency_ms": round(d["latency_ms"], 2),
               "model_queried": d["latency_ms"] > 0}
        if d["decision"] == STOP:
            trace.append(rec)
            stopped_by_model = d["overridden_by"] is None
            break
        pick = next(o for o in options if o.id == d["decision"])
        patch_index = int(pool["patch_index"][pick.payload])
        emb = store.acquire(slide_id, patch_index)
        g = information_gain(emb, embeddings)
        budget.record(patch_index); acquired.append(int(pick.payload)); embeddings.append(emb); gains.append(g)
        rec.update(selected_option=pick.id, selected_strategy=position_to_strategy[pick.id], acquired_patch_index=patch_index,
                   acquired_coord=[int(pool["coord_x"][pick.payload]), int(pool["coord_y"][pick.payload])], information_gain=round(g, 4))
        trace.append(rec)
        rnd += 1
        if budget.must_stop or not sb.build(acquired, 1):
            if budget.must_stop:
                trace.append({"slide_id": slide_id, "round": rnd, "uni_calls": budget.uni_calls, "decision": STOP, "overridden_by": "max_budget",
                              "stop_allowed": False, "stop_probability": None, "model_queried": False, "latency_ms": 0.0, "options": {},
                              "selected_option": None, "selected_strategy": None, "state": saturation_state(budget, pool, acquired, embeddings, gains, cfg),
                              "recent_information_gains": gains[-int(cfg["information_gain_window"]):]})
            break
    lat = [t["latency_ms"] for t in trace if t.get("model_queried")]
    hconf = [t["halt_confidence"] for t in trace if t.get("halt_confidence") is not None]
    return {"slide_id": slide_id, "acquired_patch_index": budget.acquired_patch_ids, "acquired_pool_index": acquired,
            "embedding": np.stack(embeddings).astype(np.float32) if embeddings else np.zeros((0, 1)), "information_gains": gains, "trace": trace,
            "stats": {"uni_calls": budget.uni_calls, "rounds": len(trace), "stopped_by_model": stopped_by_model,
                      "forced_max_stop": any(t.get("overridden_by") == "max_budget" for t in trace),
                      "clusters_covered": len(set(int(pool["cluster_id"][i]) for i in acquired)),
                      "final_mean_pairwise_similarity": _pairwise(embeddings)[0], "mean_latency_ms": float(np.mean(lat)) if lat else 0.0,
                      "mean_halt_confidence": float(np.mean(hconf)) if hconf else None,
                      "position_counts": {k: sum(t.get("selected_option") == k for t in trace) for k in "ABCD"},
                      "strategy_counts": {k: sum(t.get("selected_strategy") == k for t in trace) for k in STRATEGY_KEYS.values()}}}
