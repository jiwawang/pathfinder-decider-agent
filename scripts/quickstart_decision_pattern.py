"""Level-1 quick start: run the Adaptive Expensive-Observation Router on SYNTHETIC data (no WSI, no UNI, no AWS).

What this demonstrates — and only this — is the agent wiring:
    synthetic cheap features -> cheap candidate pool -> per-round shortlist (A-D)
    -> decision (Strands Decider 2B, or the model-free HeuristicRouter when --backend heuristic)
    -> acquisition through the lazy store (only the chosen candidate's "embedding" is ever read)
    -> evidence state / information gain -> CONTINUE / STOP -> trace
The "expensive observation" here is a MOCK: a random vector, labelled as such. It is an interface demo,
not a pathology prediction. Real UNI inference and real Bedrock reporting are configured separately (README).

  python scripts/quickstart_decision_pattern.py                     # heuristic router, CPU, ~1 s
  python scripts/quickstart_decision_pattern.py --backend strands   # real Decider 2B (downloads ~4.5 GB, GPU or CPU)
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from decision_patterns.adaptive_expensive_observation_router import HeuristicRouter  # noqa: E402
from src.acquisition.adaptive_agent import AdaptiveRouterV21, run_adaptive_episode  # noqa: E402
from src.acquisition.candidate_pool import build_cheap_pool  # noqa: E402
from src.acquisition.lazy_uni_store import LazyUNIFeatureStore  # noqa: E402
from src.cheap_features.extract import FEATURE_NAMES  # noqa: E402


def synthetic_slide(n_patches: int, seed: int):
    rng = np.random.default_rng(seed)
    feats = rng.normal(size=(n_patches, len(FEATURE_NAMES))).astype(np.float32)
    for k in ("tissue_fraction", "entropy"):
        feats[:, FEATURE_NAMES.index(k)] = rng.uniform(0.3, 1.0, n_patches)
    coords = (rng.integers(0, 120, size=(n_patches, 2)) * 224).astype(np.int32)
    feats[:, FEATURE_NAMES.index("normalized_x")] = coords[:, 0] / coords[:, 0].max()
    feats[:, FEATURE_NAMES.index("normalized_y")] = coords[:, 1] / coords[:, 1].max()
    # MOCK expensive observations: random unit vectors standing in for encoder embeddings
    mock = rng.normal(size=(n_patches, 64)).astype(np.float32)
    return feats, coords, mock


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["heuristic", "strands"], default="heuristic")
    ap.add_argument("--n-patches", type=int, default=2000)
    ap.add_argument("--pool-size", type=int, default=32)
    ap.add_argument("--min-calls", type=int, default=4)
    ap.add_argument("--max-calls", type=int, default=12)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    feats, coords, mock = synthetic_slide(args.n_patches, args.seed)
    pool = build_cheap_pool(feats, coords, FEATURE_NAMES, args.pool_size, seed=args.seed)
    tmp = Path(tempfile.mkdtemp()); np.save(tmp / "mock.npy", mock)
    store = LazyUNIFeatureStore({"synthetic": str(tmp / "mock.npy")})
    if args.backend == "strands":
        from src.decider.client import StrandsDeciderBackend
        backend = StrandsDeciderBackend("StrandsAgents/strands-decider-2B-qwen3.5-v1-2610", args.device)
    else:
        backend = HeuristicRouter()
    router = AdaptiveRouterV21(backend, min_calls=args.min_calls, max_calls=args.max_calls, low_confidence_threshold=0.15)
    cfg = {"information_gain_window": 4, "spatial_coverage_radius": 0.08}
    ep = run_adaptive_episode("synthetic", pool, store, router, cfg, shuffle_seed=args.seed)
    print(f"[quickstart] MOCK expensive observations; backend={args.backend}; pool={int(pool['pool_size'])} candidates from {args.n_patches} synthetic patches")
    for t in ep["trace"]:
        if t.get("acquired_patch_index") is not None:
            print(f"  round {t['round']:2d}: options {t['options']} -> {t['selected_option']} ({t['selected_strategy']}) acquire patch {t['acquired_patch_index']}, information gain {t['information_gain']:.3f}, P(STOP)={t['stop_probability']}")
        else:
            print(f"  round {t['round']:2d}: STOP ({t.get('overridden_by') or 'model'})")
    assert store.total_calls == ep["stats"]["uni_calls"] == len(ep["acquired_patch_index"])
    assert sorted(store.acquired["synthetic"]) == sorted(ep["acquired_patch_index"]), "only chosen candidates may be read"
    print(json.dumps({"expensive_calls": ep["stats"]["uni_calls"], "rounds": ep["stats"]["rounds"], "stopped_by_model": ep["stats"]["stopped_by_model"],
                      "strategy_counts": ep["stats"]["strategy_counts"], "position_counts": ep["stats"]["position_counts"]}, indent=1))
    print("[quickstart] OK — interface demo only; no pathology data, no UNI, no Bedrock involved.")


if __name__ == "__main__":
    main()
