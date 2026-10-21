"""One slide = one episode: walk the candidate list, ask the decider, build the evidence set.

Pathology-specific glue around decision_patterns.evidence_selector: computes vote-based support,
novelty against the selected embeddings, cluster coverage, then records the full trace.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from decision_patterns.evidence_selector import CandidateState, EvidenceSelector, EvidenceState

STATE_LABELS = {
    "evidence_strength": "evidence_strength",            # = vote_confidence (agreement of the weak KNN vote)
    "reliability": "reference_similarity",               # = mean cosine to the k nearest training prototypes
    "representativeness": "cluster_representativeness",
    "coverage": "cluster_coverage_percent",
    "novelty": "novelty_vs_selected_evidence",
}


def _unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def candidate_order(votes: dict, cand: dict, mode: str, rng: np.random.Generator) -> np.ndarray:
    n = len(cand["patch_index"])
    if mode == "random":
        return rng.permutation(n)
    prio = votes["vote_confidence"] * cand["representativeness"]
    return np.argsort(-prio, kind="stable")


def run_episode(selector: EvidenceSelector, cand: dict, votes: dict, *, order_mode: str = "priority",
                seed: int = 0, slide_id: str = "", weak_remaining_threshold: float = 0.3) -> dict[str, Any]:
    """Return {selected_indices, trace, stats}. `cand` is a candidates npz dict, `votes` from PatchBank.vote.

    The evidence state carries two booleans the agent computes itself and that the STOP criterion
    refers to: `target_budget_reached` and `remaining_candidates_are_weak` (no remaining candidate
    has priority = vote_confidence x representativeness above `weak_remaining_threshold`).
    """
    rng = np.random.default_rng(seed)
    order = candidate_order(votes, cand, order_mode, rng)
    priority = votes["vote_confidence"] * cand["representativeness"]
    E = _unit(cand["embedding"].astype(np.float32))
    selected: list[int] = []
    sel_clusters: set[int] = set()
    trace: list[dict[str, Any]] = []
    stopped = False
    for step, i in enumerate(order):
        i = int(i)
        if selected:
            redundancy = float(np.max(E[selected] @ E[i]))
            novelty = 1.0 - redundancy
        else:
            novelty = 1.0
        p1 = float(votes["vote_p1"][i])
        cs = CandidateState(
            evidence_strength=float(votes["vote_confidence"][i]),
            reliability=float(votes["mean_neighbor_similarity"][i]),
            representativeness=float(cand["representativeness"][i]),
            coverage=float(cand["cluster_fraction"][i]),
            novelty=novelty,
            extra={"evidence_leans_towards": "class_1" if p1 >= 0.5 else "class_0", "weak_class_1_vote": p1},
        )
        remaining = order[step + 1:]
        strongest_remaining = float(priority[remaining].max()) if len(remaining) else 0.0
        ev = EvidenceState(
            selected_count=len(selected), min_selected=selector.min_selected,
            target_selected=selector.target_selected, max_selected=selector.max_selected,
            unique_groups=len(sel_clusters), examined_count=step,
            remaining_count=len(order) - step,
            extra={"mean_selected_vote": float(np.mean(votes["vote_p1"][selected])) if selected else 0.5,
                   "target_budget_reached": len(selected) >= selector.target_selected,
                   "remaining_candidates_are_weak": strongest_remaining < weak_remaining_threshold},
        )
        d = selector.decide(cs, ev)
        rec = {"slide_id": slide_id, "step": step, "candidate_index": i,
               "patch_index": int(cand["patch_index"][i]), "cluster_id": int(cand["cluster_id"][i]),
               "coord_x": int(cand["coord_x"][i]), "coord_y": int(cand["coord_y"][i]),
               "state": d.state, "decision": d.action, "raw_decision": d.raw_action,
               "overridden_by": d.overridden_by, "allowed_actions": list(d.allowed_actions),
               "option_probabilities": d.probabilities, "confidence": d.confidence,
               "latency_ms": round(d.latency_ms, 2)}
        trace.append(rec)
        if d.action == "SELECT":
            selected.append(i)
            sel_clusters.add(int(cand["cluster_id"][i]))
        elif d.action == "STOP":
            stopped = True
            break
    if not selected:
        # Degenerate safety net: never return an empty evidence set. Take the top-priority candidate
        # and record that this happened so it can be audited.
        selected = [int(order[0])]
        trace.append({"slide_id": slide_id, "step": len(trace), "candidate_index": selected[0],
                      "decision": "FORCED_SELECT", "note": "empty evidence set after exhausting candidates"})
    lat = [t["latency_ms"] for t in trace if t.get("latency_ms", 0) > 0]
    confs = [t["confidence"] for t in trace if "confidence" in t]
    acts = [t["decision"] for t in trace]
    stats = dict(
        num_selected=len(selected), num_examined=len(trace), num_candidates=len(order),
        stopped_by_model=stopped, clusters_covered=len(sel_clusters),
        mean_latency_ms=float(np.mean(lat)) if lat else 0.0,
        mean_confidence=float(np.mean(confs)) if confs else 0.0,
        low_confidence_rate=float(np.mean([c < selector.low_conf for c in confs])) if confs else 0.0,
        action_counts={a: acts.count(a) for a in ("SELECT", "SKIP", "STOP")},
    )
    return {"selected_indices": selected, "trace": trace, "stats": stats}
