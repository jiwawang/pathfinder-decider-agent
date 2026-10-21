"""V2 experiment: sequential active patch acquisition under a UNI budget (docs/WP/WP001.md).

Acquisition is label-agnostic and slide-local, so every policy runs ONCE per slide (per acquisition seed
for the random baseline); the leakage-safe patient-level CV then only fits the classifier. Every
embedding is obtained through LazyUNIFeatureStore.acquire, which counts UNI calls.

Examples
  python scripts/run_sequential_cv.py --tag decider_v3_active
  python scripts/run_sequential_cv.py --tag heuristic_v3 --router heuristic
  python scripts/run_sequential_cv.py --tag decider_v3_budget --decider-max-calls 8 16 32 64   # fixed-budget Decider points

Outputs under results/<tag>/:
  oof_predictions.csv, per_run_metrics.csv, summary.csv, budget_curve.csv (+ .png), acquisition_stats.csv,
  decider_summary.json, traces/<method>/<slide_id>.json, config_used.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from decision_patterns.adaptive_expensive_observation_router import AdaptiveObservationRouter, HeuristicRouter  # noqa: E402
from src.acquisition.lazy_uni_store import LazyUNIFeatureStore  # noqa: E402
from src.acquisition.sequential_agent import (acquire_indices, cheap_cluster_topk, cheap_greedy,  # noqa: E402
                                              random_patches, run_acquisition, uniform_spatial_patches)
from src.common.config import load_config, resolve  # noqa: E402
from src.evaluation.cross_validation import fit_predict, summarise  # noqa: E402
from src.evaluation.metrics import slide_metrics  # noqa: E402
from src.evaluation.splits import make_folds  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--pool-size", type=int, default=0)
    ap.add_argument("--router", default=None, help="strands | heuristic (overrides config)")
    ap.add_argument("--budgets", type=int, nargs="*", default=None)
    ap.add_argument("--methods", default="full_uni,random,uniform,cheap_topk,cheap_greedy,decider")
    ap.add_argument("--decider-max-calls", type=int, nargs="*", default=None,
                    help="extra Decider runs with min=target=max=K (fixed-budget points on the curve)")
    ap.add_argument("--seeds", type=int, nargs="*", default=None, help="classifier CV seeds")
    ap.add_argument("--acq-seeds", type=int, nargs="*", default=[0, 1, 2], help="random-baseline draws")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = load_config(args.config)
    acq = cfg["acquisition"]
    if args.router:
        acq["router_backend"] = args.router
    pool_k = args.pool_size or int(acq["candidate_pool_size"])
    budgets = args.budgets if args.budgets is not None else list(acq["budgets"])
    methods = [m for m in args.methods.split(",") if m]
    seeds = args.seeds if args.seeds is not None else list(cfg["evaluation"]["seeds"])
    out_dir = resolve(cfg["paths"]["results_dir"]) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config_used.yaml").write_text(yaml.safe_dump({k: v for k, v in cfg.items() if not k.startswith("_")}))

    meta = pd.read_csv(resolve(cfg["paths"]["metadata_csv"]), dtype={"case_id": str, "patient_id": str})
    meta = meta.sort_values("slide_id").reset_index(drop=True)
    sids = meta.slide_id.tolist()
    y = meta.label.values.astype(int)
    art = resolve(cfg["paths"]["candidates_dir"]).parent
    pools = {s: {k: v for k, v in np.load(art / f"candidate_pool_K{pool_k}" / f"{s}.npz", allow_pickle=True).items()} for s in sids}
    coords = {s: np.load(art / "cheap_features" / f"{s}.npz")["coords"] for s in sids}
    cands_v1 = {s: np.load(resolve(cfg["paths"]["candidates_dir"]) / f"{s}.npz", allow_pickle=True) for s in sids} if "full_uni" in methods else {}
    store = LazyUNIFeatureStore(dict(zip(meta.slide_id, meta.features_npy)), l2_normalize=bool(cfg["embeddings"]["l2_normalize"]))

    # ---------------- acquisition: one pass per (method, budget, acq_seed) per slide ----------------
    runs: dict[str, dict] = {}          # run name -> {sid: {"X": slide embedding, "uni_calls": int}}
    acq_rows, trace_dir = [], out_dir / "traces"
    t0 = time.time()

    def register(name: str, sid: str, E: np.ndarray, calls: int, extra: dict | None = None):
        runs.setdefault(name, {})[sid] = {"X": E.mean(0).astype(np.float32), "uni_calls": calls}
        acq_rows.append({"run": name, "slide_id": sid, "uni_calls": calls, **(extra or {})})

    if "full_uni" in methods:
        for s in sids:   # Baseline 0: the V1 all-patch mean (25k UNI calls per slide, cached in the V1 candidates file)
            register("full_uni", s, cands_v1[s]["all_patch_mean"][None, :], int(cands_v1[s]["n_patches"]))
    for K in budgets:
        for s in sids:
            n = store.n_patches(s)
            if "random" in methods:
                for a in args.acq_seeds:
                    idx = random_patches(n, K, seed=1000 * a + K)
                    register(f"random_{K}_s{a}", s, acquire_indices(s, store, idx), len(idx), {"acq_seed": a})
            if "uniform" in methods:
                idx = uniform_spatial_patches(coords[s], K)
                register(f"uniform_{K}", s, acquire_indices(s, store, idx), len(idx))
            if "cheap_topk" in methods:
                idx = cheap_cluster_topk(pools[s], K)
                register(f"cheap_topk_{K}", s, acquire_indices(s, store, idx), len(idx))
            if "cheap_greedy" in methods:
                idx = cheap_greedy(pools[s], K)
                register(f"cheap_greedy_{K}", s, acquire_indices(s, store, idx), len(idx))
        print(f"[acq] fixed budget K={K} done ({time.time()-t0:.0f}s)", flush=True)

    decider_runs = []
    if "decider" in methods:
        decider_runs.append(("decider", int(acq["min_uni_calls"]), int(acq["target_uni_calls"]), int(acq["max_uni_calls"])))
    for K in (args.decider_max_calls or []):
        decider_runs.append((f"decider_fixed_{K}", K, K, K))
    if decider_runs:
        if acq["router_backend"] == "strands":
            from src.decider.client import StrandsDeciderBackend
            backend = StrandsDeciderBackend(cfg["decider"]["model_id"], args.device)
        else:
            backend = HeuristicRouter()
        for name, mn, tg, mx in decider_runs:
            router = AdaptiveObservationRouter(backend, min_calls=mn, target_calls=tg, max_calls=mx,
                                               low_confidence_threshold=float(acq["low_confidence_threshold"]),
                                               question=acq["question"], stop_criterion=acq["stop_criterion"],
                                               stop_mode=acq.get("stop_mode", "separate"), halt_question=acq.get("halt_question", None) or None,
                                               halt_criteria=acq.get("halt_criteria"))
            (trace_dir / name).mkdir(parents=True, exist_ok=True)
            for s in sids:
                ep = run_acquisition(s, pools[s], store, router, acq, shortlist_size=int(acq["shortlist_size"]))
                register(name, s, ep["embedding"], ep["stats"]["uni_calls"], ep["stats"] | {"choice_counts": json.dumps(ep["stats"]["choice_counts"])})
                (trace_dir / name / f"{s}.json").write_text(json.dumps({k: v for k, v in ep.items() if k != "embedding"}, indent=1, default=float))
            calls = [runs[name][s]["uni_calls"] for s in sids]
            print(f"[acq] {name}: mean UNI calls {np.mean(calls):.1f} (min {min(calls)} max {max(calls)}) ({time.time()-t0:.0f}s)", flush=True)

    pd.DataFrame(acq_rows).to_csv(out_dir / "acquisition_stats.csv", index=False)
    assert store.total_calls == sum(r["uni_calls"] for r in acq_rows if r["run"] != "full_uni"), "UNI call accounting mismatch"

    # ---------------- classifier CV (patient-level, leakage-safe) ----------------
    pred_rows = []
    for name, per_slide in runs.items():
        X = np.stack([per_slide[s]["X"] for s in sids])
        for seed in seeds:
            for f, (tr, te) in enumerate(make_folds(meta, int(cfg["evaluation"]["folds"]), seed)):
                p = fit_predict(cfg["classifier"], X[tr], y[tr], X[te])
                for j, i in enumerate(te):
                    pred_rows.append(dict(method=name, seed=seed, fold=f, rep=0, slide_id=sids[i], true_label=int(y[i]),
                                          pred_probability=float(p[j]), pred_label=int(p[j] >= 0.5),
                                          num_selected=per_slide[sids[i]]["uni_calls"], num_examined=per_slide[sids[i]]["uni_calls"]))
    pred = pd.DataFrame(pred_rows)
    # random draws: average the metric over acquisition seeds by treating each draw as a separate method name
    pred.to_csv(out_dir / "oof_predictions.csv", index=False)
    summary, per_run = summarise(pred, slide_metrics)
    summary.to_csv(out_dir / "summary.csv", index=False)
    per_run.to_csv(out_dir / "per_run_metrics.csv", index=False)

    # ---------------- budget curve ----------------
    rows = []
    import re
    for _, r in summary.iterrows():
        base = re.sub(r"_s\d+$", "", r.method)                 # random_16_s2 -> random_16 (draws share a point)
        fam = base.rsplit("_", 1)[0] if base[-1].isdigit() else base
        rows.append(dict(method=r.method, family=fam, uni_calls=r.mean_patches_used_mean, auroc=r.auroc_mean,
                         auroc_std=r.auroc_std, balanced_accuracy=r.balanced_accuracy_mean, f1=r.f1_mean, accuracy=r.accuracy_mean))
    curve = pd.DataFrame(rows)
    # random draws: one point per K = mean over draws (std = std of per-(seed, draw) AUROCs)
    rnd = per_run[per_run.method.str.startswith("random_")].copy()
    if len(rnd):
        rnd["K"] = rnd.method.str.extract(r"random_(\d+)_s")[0].astype(int)
        agg = rnd.groupby("K").agg(auroc=("auroc", "mean"), auroc_std=("auroc", "std"), balanced_accuracy=("balanced_accuracy", "mean"),
                                   f1=("f1", "mean"), accuracy=("accuracy", "mean")).reset_index()
        curve = curve[~curve.method.str.startswith("random_")]
        curve = pd.concat([curve, pd.DataFrame(dict(method=[f"random_{k}" for k in agg.K], family="random", uni_calls=agg.K.astype(float),
                                                    auroc=agg.auroc, auroc_std=agg.auroc_std, balanced_accuracy=agg.balanced_accuracy,
                                                    f1=agg.f1, accuracy=agg.accuracy))], ignore_index=True)
    curve = curve.sort_values(["family", "uni_calls"]).reset_index(drop=True)
    curve.to_csv(out_dir / "budget_curve.csv", index=False)
    print("\n" + curve.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for fam, g in curve.groupby("family"):
            g = g.sort_values("uni_calls")
            style = dict(marker="o") if fam != "full_uni" else dict(marker="*", markersize=14, linestyle="none")
            ax.errorbar(g.uni_calls, g.auroc, yerr=g.auroc_std.fillna(0), label=fam, capsize=3, **style)
        ax.set_xscale("log"); ax.set_xlabel("UNI calls per slide (log)"); ax.set_ylabel("AUROC (5x5-fold CV)")
        ax.set_title("Performance vs UNI compute budget"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(out_dir / "budget_curve.png", dpi=150)
    except Exception as e:  # plotting is a convenience
        print("plot skipped:", e)

    if decider_runs:
        st = pd.read_csv(out_dir / "acquisition_stats.csv")
        d = st[st.run == decider_runs[0][0]]
        lat = []
        for p in (trace_dir / decider_runs[0][0]).glob("*.json"):
            lat += [t["decider_latency_ms"] for t in json.loads(p.read_text())["trace"] if t["model_queried"]]
        lat = np.array(lat) if lat else np.zeros(1)
        ds = dict(run=decider_runs[0][0], slides=int(len(d)), decisions=int(len(lat)),
                  latency_ms={"mean": float(lat.mean()), "median": float(np.median(lat)), "p95": float(np.percentile(lat, 95))},
                  mean_uni_calls=float(d.uni_calls.mean()), min_uni_calls=int(d.uni_calls.min()), max_uni_calls=int(d.uni_calls.max()),
                  mean_rounds=float(d.rounds.mean()), stopped_by_model_rate=float(d.stopped_by_model.mean()),
                  mean_confidence=float(d.mean_confidence.mean()), low_confidence_rate=float(d.low_confidence_rate.mean()),
                  mean_clusters_covered=float(d.clusters_covered.mean()),
                  full_uni_calls_per_slide=float(meta.n_patches.mean()),
                  uni_calls_reduction_factor=float(meta.n_patches.mean() / d.uni_calls.mean()))
        (out_dir / "decider_summary.json").write_text(json.dumps(ds, indent=2))
        print("\n[decider]", json.dumps(ds, indent=1))
    print(f"[v2] done -> {out_dir}")


if __name__ == "__main__":
    main()
