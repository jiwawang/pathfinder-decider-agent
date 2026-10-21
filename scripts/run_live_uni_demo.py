"""End-to-end LIVE sequential acquisition on one WSI: unselected patches never touch UNI (WP001 sections 3, 25).

  cheap features (precomputed, UNI-free) -> cheap pool -> shortlist -> Strands Decider -> crop + UNI on the
  chosen patch only -> ... -> STOP -> mean pool -> classifier trained on the other slides -> Bedrock report

Run: python scripts/run_live_uni_demo.py --slide-id <private-slide> [--router heuristic]
Writes results/live_demo/<slide_id>.json (trace, per-call UNI latency, prediction, report)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from decision_patterns.adaptive_expensive_observation_router import AdaptiveObservationRouter, HeuristicRouter  # noqa: E402
from src.acquisition.lazy_uni_store import LazyUNIFeatureStore, LiveUNIFeatureStore  # noqa: E402
from src.acquisition.sequential_agent import run_acquisition  # noqa: E402
from src.bedrock.reporter import generate_report  # noqa: E402
from src.common.config import load_config, resolve  # noqa: E402
from src.models.slide_classifier import build_classifier  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--slide-id", required=True)
    ap.add_argument("--router", default=None)
    ap.add_argument("--train-run", default="decider_v3_active", help="results run whose acquisitions train the classifier")
    args = ap.parse_args()
    cfg = load_config(args.config)
    acq = cfg["acquisition"]
    meta = pd.read_csv(resolve(cfg["paths"]["metadata_csv"]), dtype={"case_id": str}).sort_values("slide_id").reset_index(drop=True)
    art = resolve(cfg["paths"]["candidates_dir"]).parent
    sid = args.slide_id
    pool = {k: v for k, v in np.load(art / f"candidate_pool_K{int(acq['candidate_pool_size'])}" / f"{sid}.npz", allow_pickle=True).items()}
    coords = {sid: np.load(art / "cheap_features" / f"{sid}.npz")["coords"]}

    t0 = time.time()
    live = LiveUNIFeatureStore({sid: meta.set_index("slide_id").loc[sid, "wsi_path"]}, coords)
    print(f"[live] UNI loaded in {time.time()-t0:.1f}s; slide has {live.n_patches(sid)} tissue patches", flush=True)
    backend_name = args.router or acq["router_backend"]
    if backend_name == "strands":
        from src.decider.client import StrandsDeciderBackend
        backend = StrandsDeciderBackend(cfg["decider"]["model_id"], "cuda")
    else:
        backend = HeuristicRouter()
    router = AdaptiveObservationRouter(backend, min_calls=int(acq["min_uni_calls"]), target_calls=int(acq["target_uni_calls"]),
                                       max_calls=int(acq["max_uni_calls"]), low_confidence_threshold=float(acq["low_confidence_threshold"]),
                                       question=acq["question"], stop_criterion=acq["stop_criterion"],
                                       stop_mode=acq.get("stop_mode", "separate"), halt_question=acq.get("halt_question") or None,
                                       halt_criteria=acq.get("halt_criteria"))
    t0 = time.time()
    ep = run_acquisition(sid, pool, live, router, acq, shortlist_size=int(acq["shortlist_size"]))
    wall = time.time() - t0
    for t in ep["trace"]:
        if t["uni_call"]:
            print(f"[live] round {t['round']:2d}: {t['decision']} (conf {t['confidence']:.2f}) -> UNI call #{t['uni_call']} on patch {t['acquired_patch_index']} at {t['acquired_coord']}", flush=True)
        else:
            print(f"[live] round {t['round']:2d}: STOP (conf {t['confidence']:.2f})", flush=True)
    assert live.total_calls == ep["stats"]["uni_calls"] == len(live.acquired[sid]), "UNI call accounting mismatch"

    # classifier: trained on the OTHER slides' acquisitions from the cached run (never on this slide)
    run_dir = resolve(cfg["paths"]["results_dir"]) / args.train_run
    trace_dir = run_dir / "traces" / "decider"
    store = LazyUNIFeatureStore(dict(zip(meta.slide_id, meta.features_npy)))
    X, y = [], []
    for r in meta.itertuples():
        if r.slide_id == sid:
            continue
        tr = json.loads((trace_dir / f"{r.slide_id}.json").read_text())
        X.append(np.stack([store.acquire(r.slide_id, i) for i in tr["acquired_patch_index"]]).mean(0)); y.append(int(r.label))
    clf = build_classifier(cfg["classifier"]).fit(np.stack(X), np.array(y))
    p1 = float(clf.predict_proba(ep["embedding"].mean(0)[None, :])[0, 1])
    pred = {"pred_label": int(p1 >= 0.5), "pred_probability": p1}
    n_all = live.n_patches(sid)
    summary = {
        "slide_id": sid, "mode": "live_on_demand_uni",
        "prediction": {"predicted_class": pred["pred_label"], "predicted_class_name": cfg["labels"]["class_names"][pred["pred_label"]],
                       "probability_class_1": round(p1, 3)},
        "search": {"tissue_patches_in_slide": n_all, "candidate_pool": int(pool["pool_size"]), "rounds": ep["stats"]["rounds"],
                   "uni_calls": ep["stats"]["uni_calls"], "uni_compute_avoided_percent": round(100 * (1 - ep["stats"]["uni_calls"] / n_all), 3),
                   "cheap_clusters_covered": ep["stats"]["clusters_covered"], "stopped_by_agent": ep["stats"]["stopped_by_model"],
                   "choice_counts": ep["stats"]["choice_counts"], "mean_decision_confidence": round(ep["stats"]["mean_confidence"], 3),
                   "mean_decision_latency_ms": round(ep["stats"]["mean_decider_latency_ms"], 1),
                   "mean_uni_call_latency_ms": round(float(np.mean(live.latency_ms)), 1), "wall_time_s": round(wall, 1),
                   "final_mean_pairwise_uni_similarity": round(ep["stats"]["final_mean_pairwise_similarity"], 3)},
        "true_label_hidden_from_reporter": True,
    }
    report = generate_report(summary, max_tokens=int(cfg["bedrock"]["max_tokens"]))
    out = resolve(cfg["paths"]["results_dir"]) / "live_demo" / f"{sid}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "report": report, "uni_call_latency_ms": live.latency_ms,
                               "trace": ep["trace"], "acquired_patch_index": ep["acquired_patch_index"]}, indent=1, default=float))
    print(json.dumps(summary, indent=1)); print(f"\n[{report['source']}] {report['report']}\n-> {out}")


if __name__ == "__main__":
    main()
