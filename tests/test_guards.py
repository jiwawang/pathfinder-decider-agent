"""Fast unit tests for the leakage guards and the decision-pattern guard-rails (no GPU, no data).

Run: python -m pytest tests -q   (pytest is not pinned in requirements; `pip install pytest`)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from decision_patterns.evidence_selector import CandidateState, EvidenceSelector, EvidenceState, ThresholdPolicy, render_state  # noqa: E402
from src.evaluation.splits import make_folds  # noqa: E402
from src.voting.knn_vote import PatchBank  # noqa: E402


def _fake_cands(n_slides=12, n_cand=8, d=16, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for i in range(n_slides):
        sid = f"s{i:02d}"
        out[sid] = {"embedding": rng.normal(size=(n_cand, d)).astype(np.float32), "label": i % 2}
    return out


def test_bank_excludes_own_slide_and_test_slides():
    cands = _fake_cands()
    train = [s for s in cands if s not in ("s10", "s11")]
    bank = PatchBank.from_candidates(cands, train, k=5)
    assert not set(bank.slide_ids) & {"s10", "s11"}, "test slides leaked into the bank"
    v = bank.vote(cands["s03"]["embedding"], exclude_slide_id="s03")
    assert not (v["neighbor_slide_ids"] == "s03").any(), "leave-one-slide-out violated"
    assert v["vote_p1"].shape == (8,) and np.all((v["vote_confidence"] >= 0) & (v["vote_confidence"] <= 1))


def test_folds_are_patient_disjoint_and_stratified():
    meta = pd.DataFrame({"slide_id": [f"s{i}" for i in range(40)], "label": [i % 2 for i in range(40)],
                         "patient_id": [f"p{i // 2}" for i in range(40)]})   # two slides per patient
    for tr, te in make_folds(meta, 5, seed=1):
        assert not set(meta.patient_id[tr]) & set(meta.patient_id[te])
        assert 0 < meta.label.values[te].mean() < 1


def test_selector_guard_rails():
    sel = EvidenceSelector(ThresholdPolicy(), min_selected=2, target_selected=3, max_selected=4,
                           low_confidence_threshold=0.3)
    assert sel.allowed_actions(0) == ("SELECT", "SKIP")
    assert sel.allowed_actions(2) == ("SELECT", "SKIP", "STOP")
    assert sel.allowed_actions(4) == ("STOP",)
    c = CandidateState(0.9, 0.8, 0.9, 0.05, 0.9)
    d = sel.decide(c, EvidenceState(4, 2, 3, 4, 4, 10, 5))
    assert d.action == "STOP" and d.overridden_by == "max_budget"
    d = sel.decide(c, EvidenceState(0, 2, 3, 4, 0, 0, 10))
    assert d.action == "SELECT" and "STOP" not in d.allowed_actions
    st = render_state(c, EvidenceState(0, 2, 3, 4, 0, 0, 10, extra={"target_budget_reached": False}))
    assert st["current_evidence_state"]["target_budget_reached"] is False


class _LowConf:
    def choose(self, state, question, options):
        return "SELECT", {k: 1 / len(options) for k in options}, 0.0


def test_low_confidence_defaults_to_skip():
    sel = EvidenceSelector(_LowConf(), min_selected=1, target_selected=2, max_selected=3, low_confidence_threshold=0.3)
    d = sel.decide(CandidateState(0.9, 0.8, 0.9, 0.05, 0.9), EvidenceState(0, 1, 2, 3, 0, 0, 5))
    assert d.action == "SKIP" and d.raw_action == "SELECT" and d.overridden_by == "low_confidence"
