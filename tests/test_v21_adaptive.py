"""WP002 section 26 tests (no GPU, no data)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from decision_patterns.adaptive_expensive_observation_router import STOP  # noqa: E402
from src.acquisition.adaptive_agent import (AdaptiveRouterV21, FORBIDDEN_STATE_TOKENS, SaturationThresholdRouter,  # noqa: E402
                                            STRATEGY_KEYS, gain_window_stats, information_gain, run_adaptive_episode)
from src.acquisition.candidate_pool import build_cheap_pool  # noqa: E402
from src.acquisition.lazy_uni_store import LazyUNIFeatureStore  # noqa: E402
from src.cheap_features.extract import FEATURE_NAMES  # noqa: E402

CFG = {"information_gain_window": 4, "spatial_coverage_radius": 0.08}


@pytest.fixture
def slide(tmp_path):
    rng = np.random.default_rng(1)
    n = 300
    feats = rng.normal(size=(n, len(FEATURE_NAMES))).astype(np.float32)
    for k in ("tissue_fraction", "entropy", "normalized_x", "normalized_y"):
        feats[:, FEATURE_NAMES.index(k)] = rng.uniform(0, 1, n)
    coords = (rng.integers(0, 150, size=(n, 2)) * 224).astype(np.int32)
    npy = tmp_path / "s_features.npy"
    np.save(npy, rng.normal(size=(n, 16)).astype(np.float32))
    pool = build_cheap_pool(feats, coords, FEATURE_NAMES, 24)
    return pool, LazyUNIFeatureStore({"s": str(npy)})


class AlwaysContinue:
    def choose_many(self, state, questions):
        return {n: (("CONTINUE" if n == "halt" else "A"), {k: 0.5 for k in o}, 0.9) for n, (q, o) in questions.items()}


class StopWhenAllowed:
    def choose_many(self, state, questions):
        return {n: ((STOP if n == "halt" else "B"), {k: 0.5 for k in o}, 0.9) for n, (q, o) in questions.items()}


def test_target_count_not_in_state(slide):
    pool, store = slide
    ep = run_adaptive_episode("s", pool, store, AdaptiveRouterV21(AlwaysContinue(), min_calls=4, max_calls=10), CFG)
    blob = json.dumps([t["state"] for t in ep["trace"]] + [t.get("candidates", {}) for t in ep["trace"]]).lower()
    for tok in FORBIDDEN_STATE_TOKENS:
        assert tok not in blob, f"state contains '{tok}'"
    assert "maximum_uni_calls" in blob and "uni_calls_used" in blob


def test_stop_state_contains_information_gain(slide):
    pool, store = slide
    ep = run_adaptive_episode("s", pool, store, AdaptiveRouterV21(AlwaysContinue(), min_calls=2, max_calls=8), CFG)
    rounds = [t for t in ep["trace"] if t.get("model_queried")]
    s = rounds[5]["state"]
    for k in ("latest_information_gain", "recent_information_gains", "recent_gain_mean", "recent_gain_slope",
              "recent_gain_as_percent_of_initial_gains", "average_gain_of_first_acquisitions",
              "mean_pairwise_uni_similarity", "mean_pairwise_uni_distance", "cheap_cluster_coverage_percent", "spatial_coverage_percent",
              "remaining_candidate_count", "new_cheap_clusters_found_in_last_4"):
        assert k in s, k
    assert len(s["recent_information_gains"]) == 4
    assert ep["information_gains"][0] == 1.0
    assert abs(information_gain(np.ones(3), [np.ones(3)])) < 1e-6
    st = gain_window_stats([0.31, 0.18, 0.07, 0.02], 4)
    assert st["recent_gain_slope"] < 0 and "trend" not in " ".join(st)


def test_candidate_options_are_shuffled(slide):
    pool, store = slide
    ep = run_adaptive_episode("s", pool, store, AdaptiveRouterV21(AlwaysContinue(), min_calls=4, max_calls=16), CFG, shuffle_seed=3)
    maps = [tuple(t["options"][k] for k in "ABCD") for t in ep["trace"] if len(t.get("options", {})) == 4]
    assert len(set(maps)) > 1, "option positions never change"
    assert all(set(m) == set(STRATEGY_KEYS.values()) for m in maps)


def test_strategy_identity_preserved_after_shuffle(slide):
    pool, store = slide
    ep = run_adaptive_episode("s", pool, store, AdaptiveRouterV21(StopWhenAllowed(), min_calls=6, max_calls=16), CFG, shuffle_seed=5)
    for t in ep["trace"]:
        if t.get("selected_option"):
            assert t["selected_strategy"] == t["options"][t["selected_option"]]
            assert t["candidates"][t["selected_option"]]["why_shortlisted"]   # description travelled with the strategy
    counts = ep["stats"]["strategy_counts"]
    assert sum(counts.values()) == ep["stats"]["uni_calls"]


def test_stop_not_allowed_before_min_budget(slide):
    pool, store = slide
    ep = run_adaptive_episode("s", pool, store, AdaptiveRouterV21(StopWhenAllowed(), min_calls=5, max_calls=16), CFG)
    assert ep["stats"]["uni_calls"] == 5 and ep["stats"]["stopped_by_model"]
    assert all(not t["stop_allowed"] for t in ep["trace"] if t["uni_calls"] < 5)


def test_max_budget_forces_stop(slide):
    pool, store = slide
    ep = run_adaptive_episode("s", pool, store, AdaptiveRouterV21(AlwaysContinue(), min_calls=2, max_calls=7), CFG)
    assert ep["stats"]["uni_calls"] == 7 and ep["stats"]["forced_max_stop"] and not ep["stats"]["stopped_by_model"]
    assert store.total_calls == 7


def test_validation_label_absent_from_decider_state(slide):
    pool, store = slide
    ep = run_adaptive_episode("s", pool, store, AdaptiveRouterV21(SaturationThresholdRouter(0.5), min_calls=3, max_calls=12), CFG)
    blob = json.dumps(ep["trace"], default=str).lower()
    for tok in ("label", "responder", "effective", "invalid", "class_", "vote", "prediction"):
        assert tok not in blob
    assert ep["stats"]["uni_calls"] >= 3
