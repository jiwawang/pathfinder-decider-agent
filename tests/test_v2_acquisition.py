"""WP001 section 23: the five tests that make V2 credible. No GPU, no real data."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from decision_patterns.adaptive_expensive_observation_router import STOP, AdaptiveObservationRouter, HeuristicRouter, Option  # noqa: E402
from src.acquisition.candidate_pool import build_cheap_pool  # noqa: E402
from src.acquisition.lazy_uni_store import AcquisitionBudget, LazyUNIFeatureStore, UnacquiredAccessError  # noqa: E402
from src.acquisition.sequential_agent import cheap_greedy, run_acquisition  # noqa: E402
from src.cheap_features.extract import FEATURE_NAMES  # noqa: E402

CFG = {"spatial_coverage_radius": 0.08, "diverse_similarity_threshold": 0.55}


@pytest.fixture
def slide(tmp_path):
    rng = np.random.default_rng(0)
    n, d = 400, 32
    feats = rng.normal(size=(n, len(FEATURE_NAMES))).astype(np.float32)
    feats[:, FEATURE_NAMES.index("tissue_fraction")] = rng.uniform(0, 1, n)
    feats[:, FEATURE_NAMES.index("entropy")] = rng.uniform(0, 1, n)
    feats[:, FEATURE_NAMES.index("normalized_x")] = rng.uniform(0, 1, n)
    feats[:, FEATURE_NAMES.index("normalized_y")] = rng.uniform(0, 1, n)
    coords = (rng.integers(0, 200, size=(n, 2)) * 224).astype(np.int32)
    uni = rng.normal(size=(n, d)).astype(np.float32)
    npy = tmp_path / "s_features.npy"
    np.save(npy, uni)
    return {"feats": feats, "coords": coords, "uni": uni, "npy": str(npy)}


class CountingStore(LazyUNIFeatureStore):
    """Store that records every row access so tests can prove nothing was read early."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.rows_read = []

    def _row(self, slide_id, patch_index):
        self.rows_read.append(patch_index)
        return super()._row(slide_id, patch_index)


# 1. test_candidate_generation_uses_no_uni
def test_candidate_generation_uses_no_uni(slide):
    store = CountingStore({"s": slide["npy"]})
    pool = build_cheap_pool(slide["feats"], slide["coords"], FEATURE_NAMES, 16)
    cheap_greedy(pool, 8)
    assert store.total_calls == 0 and store.rows_read == []
    assert pool["pool_size"] == 16 and len(set(pool["patch_index"].tolist())) == 16


# 2. test_unacquired_embedding_is_hidden
def test_unacquired_embedding_is_hidden(slide):
    store = LazyUNIFeatureStore({"s": slide["npy"]})
    with pytest.raises(UnacquiredAccessError):
        store.embedding_of("s", 3)
    v = store.acquire("s", 3)
    assert v.shape == (32,) and np.allclose(store.embedding_of("s", 3), v)
    assert not any(isinstance(getattr(store, a), np.ndarray) for a in dir(store) if not a.startswith("_")), \
        "no public attribute may expose the full array"


# 3. test_uni_call_counter
def test_uni_call_counter(slide):
    store = CountingStore({"s": slide["npy"]})
    pool = build_cheap_pool(slide["feats"], slide["coords"], FEATURE_NAMES, 24)
    router = AdaptiveObservationRouter(HeuristicRouter(), min_calls=3, target_calls=5, max_calls=8)
    ep = run_acquisition("s", pool, store, router, CFG)
    assert store.total_calls == ep["stats"]["uni_calls"] == len(ep["acquired_patch_index"]) == len(store.rows_read)
    assert sorted(store.rows_read) == sorted(ep["acquired_patch_index"])      # exactly the acquired rows, nothing else
    assert ep["embedding"].shape == (ep["stats"]["uni_calls"], 32)
    b = AcquisitionBudget(1, 2, 2); b.record(0); b.record(1)
    with pytest.raises(RuntimeError):
        b.record(2)


# 4. test_decider_stop_budget
def test_decider_stop_budget():
    router = AdaptiveObservationRouter(HeuristicRouter(), min_calls=4, target_calls=6, max_calls=8)
    opts = [Option("A", "a", 0, 1.0), Option("B", "b", 1, 0.5)]
    assert router.allowed_actions(0, ["A", "B"]) == ("A", "B")
    assert router.allowed_actions(4, ["A", "B"]) == ("A", "B", STOP)
    assert router.allowed_actions(8, ["A", "B"]) == (STOP,)
    s = {"acquisition_state": {"target_budget_reached": True, "acquired_evidence_is_diverse": True},
         "candidates": {"A": {"exploration_score": 1.0}, "B": {"exploration_score": 0.5}}}
    assert router.decide(s, opts, 2).choice == "A"            # STOP not offered below the minimum
    assert router.decide(s, opts, 6).choice == STOP            # offered and taken once target + diversity hold
    d = router.decide(s, opts, 8)
    assert d.choice == STOP and d.overridden_by == "max_budget"


# 5. test_no_validation_label_in_agent_state
def test_no_validation_label_in_agent_state(slide):
    store = LazyUNIFeatureStore({"s": slide["npy"]})
    pool = build_cheap_pool(slide["feats"], slide["coords"], FEATURE_NAMES, 16)
    router = AdaptiveObservationRouter(HeuristicRouter(), min_calls=2, target_calls=3, max_calls=4)
    ep = run_acquisition("s", pool, store, router, CFG)
    import json
    blob = json.dumps([t["state"] for t in ep["trace"]] + [t["options_shown"] for t in ep["trace"]]).lower()
    for forbidden in ("label", "responder", "effective", "invalid", "class_", "vote", "uni_embedding"):
        assert forbidden not in blob, f"agent state leaks '{forbidden}'"
