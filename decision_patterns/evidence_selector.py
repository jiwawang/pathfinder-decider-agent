"""Evidence Context Selector — a reusable Strands Decider decision pattern (#context-memory).

Generic problem: many candidate context chunks (document chunks, retrieved memories, sensor
observations, image regions, pathology patches ...) compete for a small evidence budget. For each
candidate, in turn, the agent must decide:

    SELECT  add the candidate to the evidence set
    SKIP    the candidate adds nothing reliable or new
    STOP    the evidence set is already sufficient; stop searching

The pattern is domain-agnostic: it only sees a *structured state* (a few normalised numbers that
describe the candidate and the evidence set so far) and a typed `choice` question. Domain code
is responsible for computing those numbers (see src/decider/episode.py for the pathology case).

Budget guard-rails (README section 10) are applied *outside* the model: STOP is only offered once
`min_selected` is reached, and STOP is forced at `max_selected`. Low-confidence answers fall back
to a configurable default (SKIP), so an unsure model never pollutes the evidence set.

The only dependency on Strands Decider is the `DecisionBackend` protocol below; `ThresholdPolicy`
is a deterministic stand-in used for ablations and for running without a GPU.
"""
from __future__ import annotations

import time

import numpy as np
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

ACTIONS = ("SELECT", "SKIP", "STOP")

DEFAULT_QUESTION = "Which action should the evidence-selection agent take for this candidate?"

# Criteria are the option descriptions the Decider reads next to each option. They state the
# selection policy explicitly (README section 9) in terms of the state fields, so the model's choice
# is a calibrated reading of the policy against the numbers, not a free-form judgement.
DEFAULT_CRITERIA = {
    "SELECT": "evidence_strength is at least 0.5 and novelty_vs_selected_evidence is at least 0.4: "
              "strong, non-redundant evidence worth adding",
    "SKIP": "evidence_strength is below 0.5 or novelty_vs_selected_evidence is below 0.4: "
            "weak or redundant, do not add",
    "STOP": "target_budget_reached is true and remaining_candidates_are_weak is true: "
            "the evidence set is complete, stop searching",
}


@dataclass
class CandidateState:
    """What the decider sees about one candidate. All values are in [0, 1] unless noted."""
    evidence_strength: float   # how strongly the candidate supports a definite conclusion (e.g. KNN vote agreement)
    reliability: float         # how trustworthy that measurement is (e.g. similarity to its nearest references)
    representativeness: float  # how typical the candidate is of its cluster / region
    coverage: float            # fraction of the source covered by the candidate's cluster
    novelty: float             # 1 - max similarity to already selected evidence
    extra: dict[str, Any] = field(default_factory=dict)   # domain extras shown to the model (e.g. leaning)


@dataclass
class EvidenceState:
    selected_count: int
    min_selected: int
    target_selected: int
    max_selected: int
    unique_groups: int        # e.g. distinct clusters represented in the evidence set
    examined_count: int
    remaining_count: int
    extra: dict[str, float] = field(default_factory=dict)


@dataclass
class Decision:
    action: str
    probabilities: dict[str, float]
    confidence: float
    latency_ms: float
    allowed_actions: tuple[str, ...]
    raw_action: str           # model answer before low-confidence / budget overrides
    overridden_by: str | None  # None | "low_confidence" | "max_budget"
    state: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["allowed_actions"] = list(self.allowed_actions)
        return d


class DecisionBackend(Protocol):
    def choose(self, state: dict[str, Any], question: str, options: dict[str, str]) -> tuple[str, dict[str, float], float]:
        """Return (choice, per-option probabilities, confidence)."""


def _fmt(v: Any) -> Any:
    """Booleans stay booleans (JSON true/false); other numbers are rounded; strings pass through."""
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, float, np.integer, np.floating)):
        return round(float(v), 3)
    return v


def render_state(c: CandidateState, e: EvidenceState, labels: dict[str, str] | None = None) -> dict[str, Any]:
    """Structured state handed to the decider. Numbers are rounded so the text is compact and stable."""
    labels = labels or {}
    cand = {
        labels.get("evidence_strength", "evidence_strength"): round(float(c.evidence_strength), 2),
        labels.get("reliability", "reliability"): round(float(c.reliability), 2),
        labels.get("representativeness", "representativeness"): round(float(c.representativeness), 2),
        labels.get("coverage", "coverage_percent"): round(100.0 * float(c.coverage), 1),
        labels.get("novelty", "novelty_vs_selected_evidence"): round(float(c.novelty), 2),
    }
    cand.update({k: _fmt(v) for k, v in c.extra.items()})
    ev = {
        "selected_items": int(e.selected_count),
        "minimum_evidence": int(e.min_selected),
        "target_evidence_budget": int(e.target_selected),
        "maximum_evidence_budget": int(e.max_selected),
        "unique_groups_represented": int(e.unique_groups),
        "candidates_examined": int(e.examined_count),
        "candidates_remaining": int(e.remaining_count),
    }
    ev.update({k: _fmt(v) for k, v in e.extra.items()})
    return {"candidate": cand, "current_evidence_state": ev}


class EvidenceSelector:
    """Reusable SELECT / SKIP / STOP decision step."""

    def __init__(self, backend: DecisionBackend, *, min_selected: int, target_selected: int,
                 max_selected: int, low_confidence_threshold: float = 0.3,
                 low_confidence_action: str = "SKIP", question: str = DEFAULT_QUESTION,
                 criteria: dict[str, str] | None = None, state_labels: dict[str, str] | None = None):
        assert low_confidence_action in ACTIONS
        self.backend = backend
        self.min_selected, self.target_selected, self.max_selected = min_selected, target_selected, max_selected
        self.low_conf = low_confidence_threshold
        self.low_conf_action = low_confidence_action
        self.question = question
        self.criteria = dict(criteria or DEFAULT_CRITERIA)
        self.state_labels = state_labels

    def allowed_actions(self, selected_count: int) -> tuple[str, ...]:
        if selected_count >= self.max_selected:
            return ("STOP",)
        if selected_count < self.min_selected:
            return ("SELECT", "SKIP")
        return ("SELECT", "SKIP", "STOP")

    def decide(self, cand: CandidateState, ev: EvidenceState) -> Decision:
        allowed = self.allowed_actions(ev.selected_count)
        state = render_state(cand, ev, self.state_labels)
        if allowed == ("STOP",):
            return Decision("STOP", {"STOP": 1.0}, 1.0, 0.0, allowed, "STOP", "max_budget", state)
        options = {a: self.criteria[a] for a in allowed}
        t0 = time.perf_counter()
        choice, probs, conf = self.backend.choose(state, self.question, options)
        latency = (time.perf_counter() - t0) * 1000.0
        action, overridden = choice, None
        if conf < self.low_conf:
            fallback = self.low_conf_action if self.low_conf_action in allowed else "SKIP"
            if fallback != choice:
                overridden = "low_confidence"
            action = fallback
        return Decision(action, {k: float(v) for k, v in probs.items()}, float(conf), latency,
                        allowed, choice, overridden, state)


class ThresholdPolicy:
    """Deterministic ablation backend: the same interface, no model."""

    def __init__(self, select_min_support_conf: float = 0.2, select_min_novelty: float = 0.15,
                 stop_min_novelty_seen: float = 0.1):
        self.c, self.n, self.s = select_min_support_conf, select_min_novelty, stop_min_novelty_seen

    def choose(self, state, question, options):
        cand, ev = state["candidate"], state["current_evidence_state"]
        conf = next(v for k, v in cand.items() if "strength" in k)
        nov = next(v for k, v in cand.items() if k.startswith("novelty"))
        if "STOP" in options and ev["selected_items"] >= ev["target_evidence_budget"] and nov < self.s:
            choice = "STOP"
        elif conf >= self.c and nov >= self.n:
            choice = "SELECT"
        else:
            choice = "SKIP"
        probs = {k: (0.9 if k == choice else 0.1 / max(1, len(options) - 1)) for k in options}
        return choice, probs, 0.85
