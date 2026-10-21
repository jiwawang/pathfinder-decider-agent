"""Strands Decider 2B backend for the Evidence Context Selector pattern."""
from __future__ import annotations

import os
from typing import Any


class StrandsDeciderBackend:
    """Loads the model once; `choose` asks a single typed `choice` question per candidate."""

    def __init__(self, model_id: str, device: str = "cuda"):
        from strands_decider.infer import load_engine
        self.model_id = model_id
        self.engine = load_engine(model_id, device=device)

    def choose(self, state: dict[str, Any], question: str, options: dict[str, str]):
        return self.choose_many(state, {"action": (question, options)})["action"]

    def choose_many(self, state: dict[str, Any], questions: dict[str, tuple[str, dict[str, str]]]):
        """Several typed questions about ONE state in a single call (the state prefix is encoded once)."""
        from strands_decider.schema import ChoiceQuestion
        qs = {name: ChoiceQuestion(instructions=q, criteria=opts) for name, (q, opts) in questions.items()}
        answers = self.engine.ask(state, qs).answers
        return {name: (a.choice, dict(a.probabilities), float(a.confidence)) for name, a in answers.items()}


def build_backend(cfg: dict):
    """Factory from the `decider` config block."""
    backend = cfg.get("backend", "strands")
    if backend == "strands":
        return StrandsDeciderBackend(os.environ.get("DECIDER_MODEL_ID", cfg["model_id"]), cfg.get("device", "cuda"))
    if backend == "threshold":
        from decision_patterns.evidence_selector import ThresholdPolicy
        tp = cfg.get("threshold_policy", {})
        return ThresholdPolicy(tp.get("select_min_vote_confidence", 0.2), tp.get("select_min_novelty", 0.15),
                               tp.get("stop_min_novelty_seen", 0.1))
    raise ValueError(f"unknown decider backend {backend}")
