"""V2.1 experiment (docs/WP/WP002.md): target-free adaptive STOP driven by information gain.

Methods: full_uni, cheap_fixed_{K} (cheap greedy routing, fixed budget), decider_fixed_{K} (Decider routing,
fixed budget, shuffled options), threshold_adaptive (cheap routing + deterministic saturation STOP, threshold
chosen per training fold by inner CV), decider_adaptive (Decider halt + routing, min 4 / max 32, no target).

  python scripts/run_adaptive_cv.py --tag decider_v3_adaptive
Outputs under results/<tag>/: oof_predictions.csv, summary.csv, main_table.md, uni_calls_histogram.png,
uni_calls_stats.json, information_gain_examples.png, routing_analysis.json, stop_calibration.csv,
stop_vs_gain.json, threshold_selection.csv, traces/<method>/<slide>.json
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
from src.acquisition.adaptive_agent import (STRATEGY_KEYS, AdaptiveRouterV21, SaturationThresholdRouter,  # noqa: E402
                                            run_adaptive_episode)
from src.acquisition.lazy_uni_store import LazyUNIFeatureStore  # noqa: E402
from src.acquisition.sequential_agent import acquire_indices, cheap_greedy  # noqa: E402
from src.common.config import load_config, resolve  # noqa: E402
from src.evaluation.cross_validation import fit_predict, summarise  # noqa: E402
from src.evaluation.metrics import slide_metrics  # noqa: E402
from src.evaluation.splits import make_folds  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--methods", default="full_uni,cheap_fixed,decider_fixed,threshold_adaptive,decider_adaptive")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ad = cfg["adaptive"]
    methods = [m for m in args.methods.split(",") if m]
    seeds = args.seeds if args.seeds is not None else list(cfg["evaluation"]["seeds"])
    out_dir = resolve(cfg["paths"]["results_dir"]) / args.tag
    (out_dir / "traces").mkdir(parents=True, exist_ok=True)
    (out_dir / "config_used.yaml").write_text(yaml.safe_dump({k: v for k, v in cfg.items() if not k.startswith("_")}))

    meta = pd.read_csv(resolve(cfg["paths"]["metadata_csv"]), dtype={"case_id": str, "patient_id": str}).sort_values("slide_id").reset_index(drop=True)
    sids, y = meta.slide_id.tolist(), meta.label.values.astype(int)
    art = resolve(cfg["paths"]["candidates_dir"]).parent
    pool_k = int(cfg["acquisition"]["candidate_pool_size"])
    pools = {s: {k: v for k, v in np.load(art / f"candidate_pool_K{pool_k}" / f"{s}.npz", allow_pickle=True).items()} for s in sids}
    store = LazyUNIFeatureStore(dict(zip(meta.slide_id, meta.features_npy)), l2_normalize=bool(cfg["embeddings"]["l2_normalize"]))
    runs: dict[str, dict] = {}
    acq_rows = []
    t0 = time.time()

    def register(name, sid, E, calls, extra=None):
        runs.setdefault(name, {})[sid] = {"X": E.mean(0).astype(np.float32), "uni_calls": calls}
        acq_rows.append({"run": name, "slide_id": sid, "uni_calls": calls, **(extra or {})})

    def run_router(name, router):
        (out_dir / "traces" / name).mkdir(parents=True, exist_ok=True)
        for s in sids:
            ep = run_adaptive_episode(s, pools[s], store, router, ad, shuffle_seed=int(ad["shuffle_seed"]) * 100003 + hash(s) % 10007,
                                      shortlist_size=int(ad["shortlist_size"]))
            st = ep["stats"]
            register(name, s, ep["embedding"], st["uni_calls"], {k: v for k, v in st.items() if not isinstance(v, dict)}
                     | {"position_counts": json.dumps(st["position_counts"]), "strategy_counts": json.dumps(st["strategy_counts"])})
            (out_dir / "traces" / name / f"{s}.json").write_text(json.dumps({k: v for k, v in ep.items() if k != "embedding"}, indent=1, default=float))
        calls = [runs[name][s]["uni_calls"] for s in sids]
        print(f"[acq] {name}: UNI calls mean {np.mean(calls):.1f} median {np.median(calls):.0f} min {min(calls)} max {max(calls)} ({time.time()-t0:.0f}s)", flush=True)

    if "full_uni" in methods:
        for s in sids:
            c = np.load(resolve(cfg["paths"]["candidates_dir"]) / f"{s}.npz", allow_pickle=True)
            register("full_uni", s, c["all_patch_mean"][None, :], int(c["n_patches"]))
    if "cheap_fixed" in methods:
        for K in ad["fixed_budgets"]:
            for s in sids:
                idx = cheap_greedy(pools[s], K)
                register(f"cheap_fixed_{K}", s, acquire_indices(s, store, idx), len(idx))
        print(f"[acq] cheap_fixed done ({time.time()-t0:.0f}s)", flush=True)
    if "threshold_adaptive" in methods:
        for thr in ad["saturation_thresholds"]:
            run_router(f"threshold_{thr}", AdaptiveRouterV21(SaturationThresholdRouter(thr), min_calls=int(ad["min_uni_calls"]),
                                                             max_calls=int(ad["max_uni_calls"]), low_confidence_threshold=0.0))
    backend = None
    if "decider_fixed" in methods or "decider_adaptive" in methods:
        from src.decider.client import StrandsDeciderBackend
        backend = StrandsDeciderBackend(cfg["decider"]["model_id"], args.device)
    if "decider_fixed" in methods:
        for K in ad["fixed_budgets"]:
            run_router(f"decider_fixed_{K}", AdaptiveRouterV21(backend, min_calls=K, max_calls=K, low_confidence_threshold=float(ad["low_confidence_threshold"])))
    if "decider_adaptive" in methods:
        run_router("decider_adaptive", AdaptiveRouterV21(backend, min_calls=int(ad["min_uni_calls"]), max_calls=int(ad["max_uni_calls"]),
                                                         low_confidence_threshold=float(ad["low_confidence_threshold"])))
    pd.DataFrame(acq_rows).to_csv(out_dir / "acquisition_stats.csv", index=False)

    # ---------------- classifier CV ----------------
    pred_rows, thr_rows = [], []
    for seed in seeds:
        folds = make_folds(meta, int(cfg["evaluation"]["folds"]), seed)
        for f, (tr, te) in enumerate(folds):
            for name, per in runs.items():
                if name.startswith("threshold_"):
                    continue
                X = np.stack([per[s]["X"] for s in sids])
                p = fit_predict(cfg["classifier"], X[tr], y[tr], X[te])
                pred_rows += [dict(method=name, seed=seed, fold=f, rep=0, slide_id=sids[i], true_label=int(y[i]), pred_probability=float(p[j]),
                                   pred_label=int(p[j] >= 0.5), num_selected=per[sids[i]]["uni_calls"], num_examined=per[sids[i]]["uni_calls"]) for j, i in enumerate(te)]
            if "threshold_adaptive" in methods:   # nested: choose the threshold on the training fold only
                inner_meta = meta.iloc[tr].reset_index(drop=True)
                best, best_auc = None, -1
                for thr in ad["saturation_thresholds"]:
                    per = runs[f"threshold_{thr}"]
                    Xtr = np.stack([per[s]["X"] for s in inner_meta.slide_id]); ytr = inner_meta.label.values.astype(int)
                    oof = np.zeros(len(ytr))
                    for itr, ite in make_folds(inner_meta, 4, seed):
                        oof[ite] = fit_predict(cfg["classifier"], Xtr[itr], ytr[itr], Xtr[ite])
                    auc = slide_metrics(ytr, oof)["auroc"]
                    if auc > best_auc:
                        best, best_auc = thr, auc
                thr_rows.append(dict(seed=seed, fold=f, threshold=best, inner_auroc=best_auc))
                per = runs[f"threshold_{best}"]
                X = np.stack([per[s]["X"] for s in sids])
                p = fit_predict(cfg["classifier"], X[tr], y[tr], X[te])
                pred_rows += [dict(method="threshold_adaptive", seed=seed, fold=f, rep=0, slide_id=sids[i], true_label=int(y[i]), pred_probability=float(p[j]),
                                   pred_label=int(p[j] >= 0.5), num_selected=per[sids[i]]["uni_calls"], num_examined=per[sids[i]]["uni_calls"]) for j, i in enumerate(te)]
    pred = pd.DataFrame(pred_rows); pred.to_csv(out_dir / "oof_predictions.csv", index=False)
    if thr_rows:
        pd.DataFrame(thr_rows).to_csv(out_dir / "threshold_selection.csv", index=False)
    summary, per_run = summarise(pred, slide_metrics)
    summary.to_csv(out_dir / "summary.csv", index=False); per_run.to_csv(out_dir / "per_run_metrics.csv", index=False)

    # ---------------- main table ----------------
    order = ["full_uni"] + [f"cheap_fixed_{K}" for K in ad["fixed_budgets"]] + ["threshold_adaptive"] + \
            [f"decider_fixed_{K}" for K in ad["fixed_budgets"]] + ["decider_adaptive"]
    label = {"full_uni": "Full UNI", "threshold_adaptive": "Threshold Adaptive (cheap routing + saturation rule)", "decider_adaptive": "**Decider Adaptive V3**"}
    lines = ["| Method | Mean UNI calls | AUROC | Balanced Acc | F1 |", "|---|---:|---:|---:|---:|"]
    for m in order:
        if m not in set(summary.method):
            continue
        r = summary[summary.method == m].iloc[0]
        name = label.get(m, m.replace("cheap_fixed_", "Cheap Fixed-").replace("decider_fixed_", "Decider Fixed-"))
        lines.append(f"| {name} | {r.mean_patches_used_mean:,.1f} | {r.auroc_mean:.3f} ± {r.auroc_std:.3f} | {r.balanced_accuracy_mean:.3f} | {r.f1_mean:.3f} |")
    (out_dir / "main_table.md").write_text("\n".join(lines) + "\n"); print("\n".join(lines))

    # ---------------- adaptive-budget analyses ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    st = pd.read_csv(out_dir / "acquisition_stats.csv")
    analysis = {}
    for name in [m for m in ("decider_adaptive", "threshold_adaptive") if m in methods]:
        if name == "threshold_adaptive":
            # calls under the threshold chosen most often
            chosen = pd.DataFrame(thr_rows).threshold.mode()[0]
            d = st[st.run == f"threshold_{chosen}"]
        else:
            d = st[st.run == name]
        c = d.uni_calls.values
        mn, mx = int(ad["min_uni_calls"]), int(ad["max_uni_calls"])
        analysis[name] = dict(mean=float(c.mean()), median=float(np.median(c)), std=float(c.std()), min=int(c.min()), max=int(c.max()),
                              p25=float(np.percentile(c, 25)), p75=float(np.percentile(c, 75)),
                              pct_stopped_at_min=float(np.mean(c == mn)), pct_5_8=float(np.mean((c > mn) & (c <= 8))),
                              pct_9_16=float(np.mean((c > 8) & (c <= 16))), pct_17_24=float(np.mean((c > 16) & (c <= 24))),
                              pct_25_31=float(np.mean((c > 24) & (c < mx))), pct_forced_max=float(np.mean(c >= mx)))
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist(c, bins=np.arange(mn - 0.5, mx + 1.5, 1), color="#4C72B0" if name == "decider_adaptive" else "#DD8452")
        ax.set_xlabel("UNI calls per slide"); ax.set_ylabel("slides"); ax.set_title(f"{name}: UNI budget distribution (n={len(c)})")
        fig.tight_layout(); fig.savefig(out_dir / (f"uni_calls_histogram.png" if name == "decider_adaptive" else f"uni_calls_histogram_{name}.png"), dpi=150); plt.close(fig)
    (out_dir / "uni_calls_stats.json").write_text(json.dumps(analysis, indent=2))

    if "decider_adaptive" in methods:
        traces = {s: json.loads((out_dir / "traces" / "decider_adaptive" / f"{s}.json").read_text()) for s in sids}
        # routing: position vs strategy
        pos = {k: 0 for k in "ABCD"}; strat = {k: 0 for k in STRATEGY_KEYS.values()}; first_pos = {k: 0 for k in "ABCD"}
        for ep in traces.values():
            for t in ep["trace"]:
                if t.get("selected_option"):
                    pos[t["selected_option"]] += 1; strat[t["selected_strategy"]] += 1
        n = sum(pos.values())
        routing = {"position_selection_pct": {k: round(100 * v / n, 1) for k, v in pos.items()},
                   "strategy_selection_pct": {k: round(100 * v / n, 1) for k, v in strat.items()}, "n_routing_decisions": n}
        # fixed-budget decider runs also have shuffled routing: pool them for a larger sample
        pos2 = {k: 0 for k in "ABCD"}; strat2 = {k: 0 for k in STRATEGY_KEYS.values()}
        for K in ad["fixed_budgets"]:
            for p in (out_dir / "traces" / f"decider_fixed_{K}").glob("*.json") if (out_dir / "traces" / f"decider_fixed_{K}").exists() else []:
                for t in json.loads(p.read_text())["trace"]:
                    if t.get("selected_option"):
                        pos2[t["selected_option"]] += 1; strat2[t["selected_strategy"]] += 1
        n2 = sum(pos2.values())
        if n2:
            routing["all_decider_runs"] = {"position_selection_pct": {k: round(100 * v / n2, 1) for k, v in pos2.items()},
                                           "strategy_selection_pct": {k: round(100 * v / n2, 1) for k, v in strat2.items()}, "n": n2}
        (out_dir / "routing_analysis.json").write_text(json.dumps(routing, indent=2))
        # STOP calibration + STOP vs gain
        rows = []
        for ep in traces.values():
            for t in ep["trace"]:
                if t.get("stop_allowed") and t.get("stop_probability") is not None:
                    rows.append(dict(slide_id=ep["slide_id"], uni_calls=t["uni_calls"], stop_probability=t["stop_probability"],
                                     stopped=t["decision"] == "STOP", recent_gain_mean=t["recent_gain_mean"], latest_gain=t["state"]["latest_information_gain"],
                                     slope=t["recent_gain_slope"], similarity=t["mean_pairwise_uni_similarity"]))
        sc = pd.DataFrame(rows)
        if len(sc):
            sc["bucket"] = pd.cut(sc.stop_probability, [0, .2, .4, .6, .8, 1.0], include_lowest=True)
            cal = sc.groupby("bucket", observed=False).agg(frequency=("stopped", "size"), actual_stop_rate=("stopped", "mean"),
                                                           mean_recent_gain=("recent_gain_mean", "mean"), mean_uni_calls=("uni_calls", "mean")).reset_index()
            cal.to_csv(out_dir / "stop_calibration.csv", index=False)
            s1, s0 = sc[sc.stopped], sc[~sc.stopped]
            svg = dict(n_stop=int(len(s1)), n_continue=int(len(s0)),
                       recent_gain_mean={"at_STOP": float(s1.recent_gain_mean.mean()) if len(s1) else None, "at_CONTINUE": float(s0.recent_gain_mean.mean())},
                       latest_gain={"at_STOP": float(s1.latest_gain.mean()) if len(s1) else None, "at_CONTINUE": float(s0.latest_gain.mean())},
                       slope={"at_STOP": float(s1.slope.mean()) if len(s1) else None, "at_CONTINUE": float(s0.slope.mean())},
                       similarity={"at_STOP": float(s1.similarity.mean()) if len(s1) else None, "at_CONTINUE": float(s0.similarity.mean())},
                       corr_stop_probability_vs_recent_gain=float(np.corrcoef(sc.stop_probability, sc.recent_gain_mean)[0, 1]),
                       corr_stop_probability_vs_uni_calls=float(np.corrcoef(sc.stop_probability, sc.uni_calls)[0, 1]),
                       corr_stop_probability_vs_similarity=float(np.corrcoef(sc.stop_probability, sc.similarity)[0, 1]))
            (out_dir / "stop_vs_gain.json").write_text(json.dumps(svg, indent=2))
            print("[stop]", json.dumps(svg, indent=1))
        # information-gain examples: early / medium / late stop
        calls = st[st.run == "decider_adaptive"].set_index("slide_id").uni_calls
        picks = [calls.idxmin(), calls.sort_values().index[len(calls) // 2], calls.idxmax()]
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
        for ax, s, lab in zip(axes, picks, ("early stop", "median stop", "late stop")):
            g = traces[s]["information_gains"]
            ax.plot(range(1, len(g) + 1), g, marker="o"); ax.axvline(len(g) + 0.5, color="red", linestyle="--", label="STOP")
            ax.set_title(f"{lab}: {s} ({len(g)} UNI calls)", fontsize=9); ax.set_xlabel("acquisition round"); ax.grid(alpha=.3)
        axes[0].set_ylabel("information gain (1 - max cos)"); axes[0].legend()
        fig.tight_layout(); fig.savefig(out_dir / "information_gain_examples.png", dpi=150); plt.close(fig)
        print("[routing]", json.dumps(routing, indent=1))
    print("[budget]", json.dumps(analysis, indent=1)); print(f"[v2.1] done -> {out_dir}")


if __name__ == "__main__":
    main()
