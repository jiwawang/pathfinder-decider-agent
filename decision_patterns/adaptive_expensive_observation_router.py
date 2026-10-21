"""Adaptive Expensive-Observation Router — reusable Strands Decider 2B pattern (#context-memory).

Generic problem: an agent holds a large set of *cheap* observations and a limited budget of
*expensive* ones (a foundation-model forward pass, a high-resolution scan, a deep analysis, a large
VLM call). Each round a cheap heuristic proposes a short list of candidates; the Decider chooses which
candidate deserves the next expensive observation, or STOP when the evidence collected is enough.

    state   : budget used, coverage, redundancy of the evidence acquired so far, per-candidate cheap stats
    question: "Which candidate should receive the next expensive observation?"
    options : A, B, C, D (shortlist) [+ STOP once the minimum budget is spent]

The heuristic only *proposes*; the model *decides*. Budget guard-rails (minimum before STOP is offered,
forced STOP at the maximum) and the low-confidence fallback are applied outside the model, and every
decision is recorded with option probabilities, confidence and latency.

Only `DecisionBackend.choose` touches Strands Decider; `HeuristicRouter` is a model-free stand-in with
the same interface for ablations and GPU-less runs.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Protocol

STOP = "STOP"

DEFAULT_QUESTION = ("Which candidate should receive the next expensive observation? Prefer a candidate that "
                    "is novel and far from the evidence acquired so far, representative of a large unexplored "
                    "region, and rich in content.")

DEFAULT_STOP_CRITERION = ("target_budget_reached is true and acquired_evidence_is_diverse is true: enough "
                          "evidence has been collected, stop spending the expensive budget")

# "separate" stop mode: the routing question (which candidate?) and the halting question (continue or stop?)
# are two typed questions about the same state, answered in one call. A 5-way choice almost never selects
# STOP (measured P(STOP) <= 0.07 on 1,900 real states), while the 2-way halting question follows the state
# booleans closely (P(STOP) 0.73 when both hold, 0.13 otherwise).
DEFAULT_HALT_QUESTION = ("Should the agent stop acquiring expensive observations now, or continue with one of the "
                         "shortlisted candidates?")
DEFAULT_HALT_CRITERIA = {
    "CONTINUE": "target_budget_reached is false or acquired_evidence_is_diverse is false: keep acquiring evidence",
    "STOP": DEFAULT_STOP_CRITERION,
}


@dataclass
class Option:
    id: str                 # "A", "B", ...
    description: str        # what the Decider reads next to the option id (reason + cheap stats)
    payload: Any = None     # domain object (e.g. candidate index); never shown to the model
    score: float = 0.0      # cheap heuristic score; used only by HeuristicRouter / fallback


@dataclass
class RouterDecision:
    choice: str                      # option id or "STOP"
    probabilities: dict[str, float]
    confidence: float
    latency_ms: float
    allowed: tuple[str, ...]
    raw_choice: str
    overridden_by: str | None        # None | "low_confidence" | "max_budget"
    state: dict[str, Any]
    options: dict[str, str]
    halt_probabilities: dict[str, float] | None = None   # separate stop mode: P(CONTINUE), P(STOP)
    halt_confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["allowed"] = list(self.allowed)
        return d


class DecisionBackend(Protocol):
    def choose(self, state: dict[str, Any], question: str, options: dict[str, str]) -> tuple[str, dict[str, float], float]: ...

    def choose_many(self, state: dict[str, Any], questions: dict[str, tuple[str, dict[str, str]]]) -> dict[str, tuple[str, dict[str, float], float]]: ...


class AdaptiveObservationRouter:
    def __init__(self, backend: DecisionBackend, *, min_calls: int, target_calls: int, max_calls: int,
                 low_confidence_threshold: float = 0.15, question: str = DEFAULT_QUESTION,
                 stop_criterion: str = DEFAULT_STOP_CRITERION, stop_mode: str = "separate",
                 halt_question: str = DEFAULT_HALT_QUESTION, halt_criteria: dict[str, str] | None = None):
        assert stop_mode in ("separate", "joint")
        self.backend = backend
        self.min_calls, self.target_calls, self.max_calls = min_calls, target_calls, max_calls
        self.low_conf = low_confidence_threshold
        self.question, self.stop_criterion, self.stop_mode = question, stop_criterion, stop_mode
        self.halt_question = halt_question or DEFAULT_HALT_QUESTION
        self.halt_criteria = dict(halt_criteria or {**DEFAULT_HALT_CRITERIA, "STOP": stop_criterion})

    def allowed_actions(self, calls_used: int, option_ids: list[str]) -> tuple[str, ...]:
        if calls_used >= self.max_calls or not option_ids:
            return (STOP,)
        if calls_used < self.min_calls:
            return tuple(option_ids)
        return tuple(option_ids) + (STOP,)

    def decide(self, state: dict[str, Any], options: list[Option], calls_used: int) -> RouterDecision:
        ids = [o.id for o in options]
        allowed = self.allowed_actions(calls_used, ids)
        shown = {o.id: o.description for o in options if o.id in allowed}
        if allowed == (STOP,):
            return RouterDecision(STOP, {STOP: 1.0}, 1.0, 0.0, allowed, STOP, "max_budget", state, shown)
        stop_allowed = STOP in allowed
        t0 = time.perf_counter()
        halt_probs = halt_conf = None
        if self.stop_mode == "joint":
            if stop_allowed:
                shown[STOP] = self.stop_criterion
            if len(shown) == 1:
                only = next(iter(shown))
                return RouterDecision(only, {only: 1.0}, 1.0, 0.0, allowed, only, None, state, shown)
            choice, probs, conf = self.backend.choose(state, self.question, shown)
            halt_choice = STOP if choice == STOP else "CONTINUE"
            route_choice = choice
        else:
            questions = {}
            if len(shown) >= 2:
                questions["route"] = (self.question, shown)
            if stop_allowed:
                questions["halt"] = (self.halt_question, self.halt_criteria)
            answers = self.backend.choose_many(state, questions) if questions else {}
            if "route" in answers:
                route_choice, probs, conf = answers["route"]
            else:
                route_choice, probs, conf = next(iter(shown)), {next(iter(shown)): 1.0}, 1.0
            if "halt" in answers:
                halt_choice, halt_probs, halt_conf = answers["halt"]
                if halt_conf < self.low_conf:        # an unsure halt never stops the search
                    halt_choice = "CONTINUE"
                shown = {**shown, STOP: self.halt_criteria[STOP]}
            else:
                halt_choice = "CONTINUE"
            choice = STOP if halt_choice == STOP else route_choice
        latency = (time.perf_counter() - t0) * 1000.0
        final, overridden = choice, None
        if choice != STOP and conf < self.low_conf:
            # Fallback: the heuristic's best-scored option (never STOP), so an unsure model keeps exploring.
            best = max((o for o in options if o.id in allowed), key=lambda o: o.score)
            final, overridden = best.id, ("low_confidence" if best.id != choice else None)
        return RouterDecision(final, {k: float(v) for k, v in probs.items()}, float(conf), latency,
                              allowed, choice, overridden, state, shown, halt_probs, halt_conf)


class HeuristicRouter:
    """Model-free ablation: always the best cheap score; STOP once the target is reached and the
    evidence is already diverse."""

    def choose(self, state, question, options):
        acq = state.get("acquisition_state", {})
        if STOP in options and acq.get("target_budget_reached") and acq.get("acquired_evidence_is_diverse"):
            choice = STOP
        else:
            cands = state.get("candidates", {})
            choice = max((k for k in options if k != STOP), key=lambda k: cands.get(k, {}).get("exploration_score", 0.0))
        probs = {k: (0.9 if k == choice else 0.1 / max(1, len(options) - 1)) for k in options}
        return choice, probs, 0.85

    def choose_many(self, state, questions):
        out = {}
        for name, (q, opts) in questions.items():
            if name == "halt":
                acq = state.get("acquisition_state", {})
                c = STOP if acq.get("target_budget_reached") and acq.get("acquired_evidence_is_diverse") else "CONTINUE"
                out[name] = (c, {k: (0.9 if k == c else 0.1) for k in opts}, 0.8)
            else:
                out[name] = self.choose(state, q, opts)
        return out
