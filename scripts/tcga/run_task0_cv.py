"""WP005 Task 0 benchmark: frozen acquisition policies on COAD/KIRC/LUAD (3-class), cached simulation.

Methods (frozen, same as decider_v3_adaptive / decider_v3_active): full_uni, random_K (3 draws), uniform_K,
cheap_topk_K, cheap_greedy_K, decider_fixed_K, decider_adaptive  for K in budgets [4, 8, 16, 32].
Acquisition code is reused unchanged from src/acquisition; only the data source (TCGA cache) and the
multiclass evaluation differ. Candidate pools (K=128, cheap features) are written next to the coords layer.

  CUDA_VISIBLE_DEVICES=1 python scripts/tcga/run_task0_cv.py [--methods ...] [--seeds 0]
Outputs results/tcga_generalization/task0_3class/: metadata.json, config.yaml, oof_predictions.csv, metrics.csv,
per_run_metrics.csv, acquisition_stats.csv, budget_curve.csv/.png, summary.md, traces/<method>/<slide>.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from decision_patterns.adaptive_expensive_observation_router import HeuristicRouter  # noqa: E402
from src.acquisition.adaptive_agent import AdaptiveRouterV21, run_adaptive_episode  # noqa: E402
from src.acquisition.candidate_pool import build_cheap_pool  # noqa: E402
from src.acquisition.lazy_uni_store import LazyUNIFeatureStore  # noqa: E402
from src.acquisition.sequential_agent import acquire_indices, cheap_cluster_topk, cheap_greedy, random_patches, uniform_spatial_patches  # noqa: E402
from src.common.config import load_config, resolve  # noqa: E402
from src.evaluation.splits import make_folds  # noqa: E402
from src.tcga.evaluation import fit_predict_multi, multiclass_metrics, patient_bootstrap_ci, summarise_multi  # noqa: E402


def git_info() -> dict:
    def q(*a):
        try:
            return subprocess.check_output(["git", *a], text=True).strip()
        except Exception:
            return ""
    return {"git_commit": q("rev-parse", "HEAD"), "git_branch": q("branch", "--show-current"), "git_dirty": bool(q("status", "--porcelain"))}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/tcga_task0.yaml")
    ap.add_argument("--base-config", default="configs/default.yaml")
    ap.add_argument("--methods", default="full_uni,random,uniform,cheap_topk,cheap_greedy,decider_fixed,decider_adaptive")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--router", default="strands")
    ap.add_argument("--limit", type=int, default=0, help="slides per class (smoke)")
    ap.add_argument("--run-id", default="")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    tcfg = load_config(args.config)["tcga"]; cfg = load_config(args.base_config)
    acq, ad = cfg["acquisition"], cfg["adaptive"]
    budgets = list(ad["fixed_budgets"]); seeds = args.seeds if args.seeds is not None else list(cfg["evaluation"]["seeds"])
    methods = [m for m in args.methods.split(",") if m]
    out_dir = resolve(tcfg["results_dir"]) / (args.run_id or "")
    (out_dir / "traces").mkdir(parents=True, exist_ok=True)
    table = resolve(tcfg["slide_table"])
    meta = pd.read_csv(table).sort_values("slide_id").reset_index(drop=True)
    if args.limit:
        meta = meta.groupby("category").head(args.limit).reset_index(drop=True)
    sids, y = meta.slide_id.tolist(), meta.label.values.astype(int); n_classes = int(y.max() + 1)
    meta.to_csv(out_dir / "manifest_snapshot.csv", index=False)      # exactly the slides used, with GDC provenance
    (out_dir / "figures").mkdir(exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump({"tcga": tcfg, "acquisition": acq, "adaptive": ad, "classifier": cfg["classifier"], "evaluation": cfg["evaluation"]}))
    (out_dir / "metadata.json").write_text(json.dumps({**git_info(), "experiment": tcfg["task"], "timestamp": datetime.now().isoformat(), "seeds": seeds,
                                                        "config_sha256": sha(resolve(args.config)), "base_config_sha256": sha(resolve(args.base_config)),
                                                        "slide_table_sha256": sha(table), "n_slides": len(meta), "labels": meta.label.value_counts().sort_index().to_dict(),
                                                        "decider_model": cfg["decider"]["model_id"], "router": args.router}, indent=2))

    # ---- candidate pools from cheap features (stored in the coords layer, D001 convention) ----
    pools, coords = {}, {}
    for r in meta.itertuples():
        z = np.load(r.cheap_path); coords[r.slide_id] = z["coords"]
        pdir = Path(r.cheap_path).parent.parent / f"candidate_pool_K{int(acq['candidate_pool_size'])}"; pdir.mkdir(exist_ok=True)
        pf = pdir / f"{r.slide_id}.npz"
        if not pf.exists():
            pool = build_cheap_pool(z["features"], z["coords"], [str(n) for n in z["feature_names"]], int(acq["candidate_pool_size"]), seed=int(acq["seed"]))
            np.savez(pf, slide_id=r.slide_id, **pool)
        pools[r.slide_id] = {k: v for k, v in np.load(pf, allow_pickle=True).items()}
    store = LazyUNIFeatureStore(dict(zip(meta.slide_id, meta.features_npy)), l2_normalize=True)
    runs, acq_rows, t0 = {}, [], time.time()

    def register(name, sid, E, calls, extra=None):
        runs.setdefault(name, {})[sid] = {"X": E.mean(0).astype(np.float32), "uni_calls": calls}
        acq_rows.append({"run": name, "slide_id": sid, "uni_calls": calls, **(extra or {})})

    if "full_uni" in methods:
        for r in meta.itertuples():   # compute baseline: mean over ALL cached rows (outside the policy, by definition)
            E = np.load(r.features_npy, mmap_mode="r"); register("full_uni", r.slide_id, np.asarray(E), int(E.shape[0]))
    for K in budgets:
        for s in sids:
            n = store.n_patches(s)
            if "random" in methods:
                for a in (0, 1, 2):
                    idx = random_patches(n, K, seed=1000 * a + K); register(f"random_{K}_s{a}", s, acquire_indices(s, store, idx), len(idx))
            if "uniform" in methods:
                idx = uniform_spatial_patches(coords[s], K); register(f"uniform_{K}", s, acquire_indices(s, store, idx), len(idx))
            if "cheap_topk" in methods:
                idx = cheap_cluster_topk(pools[s], K); register(f"cheap_topk_{K}", s, acquire_indices(s, store, idx), len(idx))
            if "cheap_greedy" in methods:
                idx = cheap_greedy(pools[s], K); register(f"cheap_greedy_{K}", s, acquire_indices(s, store, idx), len(idx))
        print(f"[task0] fixed K={K} baselines done ({time.time()-t0:.0f}s)", flush=True)

    def run_router(name, router):
        (out_dir / "traces" / name).mkdir(parents=True, exist_ok=True)
        for s in sids:
            ep = run_adaptive_episode(s, pools[s], store, router, ad, shuffle_seed=int(ad["shuffle_seed"]) * 100003 + hash(s) % 10007, shortlist_size=int(ad["shortlist_size"]))
            st = ep["stats"]
            register(name, s, ep["embedding"], st["uni_calls"], {k: v for k, v in st.items() if not isinstance(v, dict)} | {"strategy_counts": json.dumps(st["strategy_counts"]), "position_counts": json.dumps(st["position_counts"])})
            (out_dir / "traces" / name / f"{s}.json").write_text(json.dumps({k: v for k, v in ep.items() if k != "embedding"}, indent=1, default=float))
        c = [runs[name][s]["uni_calls"] for s in sids]
        print(f"[task0] {name}: UNI calls mean {np.mean(c):.1f} min {min(c)} max {max(c)} ({time.time()-t0:.0f}s)", flush=True)

    if "decider_fixed" in methods or "decider_adaptive" in methods:
        if args.router == "strands":
            from src.decider.client import StrandsDeciderBackend
            backend = StrandsDeciderBackend(cfg["decider"]["model_id"], args.device)
        else:
            backend = HeuristicRouter()
        if "decider_fixed" in methods:
            for K in budgets:
                run_router(f"decider_fixed_{K}", AdaptiveRouterV21(backend, min_calls=K, max_calls=K, low_confidence_threshold=float(ad["low_confidence_threshold"])))
        if "decider_adaptive" in methods:
            run_router("decider_adaptive", AdaptiveRouterV21(backend, min_calls=int(ad["min_uni_calls"]), max_calls=int(ad["max_uni_calls"]), low_confidence_threshold=float(ad["low_confidence_threshold"])))
    pd.DataFrame(acq_rows).to_csv(out_dir / "acquisition_stats.csv", index=False)
    assert store.total_calls == sum(r["uni_calls"] for r in acq_rows if r["run"] != "full_uni"), "UNI call accounting mismatch"

    # ---- classifier CV: patient-level StratifiedGroupKFold, multinomial LR, per-seed OOF ----
    pred_rows = []
    for seed in seeds:
        for f, (tr, te) in enumerate(make_folds(meta, int(cfg["evaluation"]["folds"]), seed)):
            for name, per in runs.items():
                X = np.stack([per[s]["X"] for s in sids]); P = fit_predict_multi(cfg["classifier"], X[tr], y[tr], X[te])
                for j, i in enumerate(te):
                    pred_rows.append(dict(method=name, seed=seed, fold=f, slide_id=sids[i], true_label=int(y[i]), pred_label=int(P[j].argmax()),
                                          **{f"p{c}": float(P[j, c]) for c in range(n_classes)}, num_selected=per[sids[i]]["uni_calls"]))
    pred = pd.DataFrame(pred_rows); pred.to_csv(out_dir / "oof_predictions.csv", index=False)
    summary, per_run = summarise_multi(pred, n_classes)
    summary.to_csv(out_dir / "metrics.csv", index=False); per_run.to_csv(out_dir / "per_run_metrics.csv", index=False)
    ci = patient_bootstrap_ci(pred, n_classes)

    rows = []
    for _, r in summary.iterrows():
        base = re.sub(r"_s\d+$", "", r.method); fam = base.rsplit("_", 1)[0] if base[-1].isdigit() else base
        rows.append(dict(method=r.method, family=fam, uni_calls=r.mean_patches_used_mean, auroc=r.auroc_mean, auroc_std=r.auroc_std,
                         ci_lo=ci[r.method][0], ci_hi=ci[r.method][1], balanced_accuracy=r.balanced_accuracy_mean, f1=r.f1_mean, accuracy=r.accuracy_mean,
                         **{f"auroc_class{c}": r[f"auroc_class{c}_mean"] for c in range(n_classes) if f"auroc_class{c}_mean" in r},
                         **({"pr_auc": r["pr_auc_mean"], "recall_class0": r["recall_class0_mean"], "recall_class1": r["recall_class1_mean"]} if "pr_auc_mean" in r else {})))
    curve = pd.DataFrame(rows)
    rnd = per_run[per_run.method.str.startswith("random_")].copy()
    if len(rnd):
        rnd["K"] = rnd.method.str.extract(r"random_(\d+)_s")[0].astype(int)
        agg = rnd.groupby("K").agg(auroc=("auroc", "mean"), auroc_std=("auroc", "std"), balanced_accuracy=("balanced_accuracy", "mean"), f1=("f1", "mean"), accuracy=("accuracy", "mean")).reset_index()
        curve = curve[~curve.method.str.startswith("random_")]
        curve = pd.concat([curve, pd.DataFrame(dict(method=[f"random_{k}" for k in agg.K], family="random", uni_calls=agg.K.astype(float), auroc=agg.auroc, auroc_std=agg.auroc_std,
                                                    balanced_accuracy=agg.balanced_accuracy, f1=agg.f1, accuracy=agg.accuracy))], ignore_index=True)
    curve = curve.sort_values(["family", "uni_calls"]).reset_index(drop=True); curve.to_csv(out_dir / "budget_curve.csv", index=False)
    lab = {"full_uni": "Full UNI", "random": "Random", "uniform": "Spatial uniform", "cheap_topk": "Cheap Top-K", "cheap_greedy": "Cheap Greedy", "decider_fixed": "Decider routing (fixed K)", "decider_adaptive": "Decider adaptive STOP"}
    lines = [f"# {tcfg['task']}: {len(meta)} slides, labels {meta.label.value_counts().sort_index().to_dict()}, {len(seeds)} seeds x 5 folds\n",
             ("| Method | UNI calls | ROC-AUC | 95% CI (patient bootstrap) | Bal Acc | F1 | Acc |" if n_classes == 2 else "| Method | UNI calls | macro AUROC | 95% CI (patient bootstrap) | Bal Acc | macro F1 | Acc |"), "|---|---:|---:|---:|---:|---:|---:|"]
    for fam in ("full_uni", "random", "uniform", "cheap_topk", "cheap_greedy", "decider_fixed", "decider_adaptive"):
        for _, r in curve[curve.family == fam].sort_values("uni_calls").iterrows():
            sd = f" ± {r.auroc_std:.3f}" if pd.notna(r.auroc_std) else ""; cis = f"[{r.ci_lo:.3f}, {r.ci_hi:.3f}]" if pd.notna(r.get("ci_lo", np.nan)) else "—"
            name = lab[fam] + (f" = {int(round(r.uni_calls))}" if fam not in ("full_uni", "decider_adaptive") else "")
            lines.append(f"| {name} | {r.uni_calls:,.1f} | {r.auroc:.3f}{sd} | {cis} | {r.balanced_accuracy:.3f} | {r.f1:.3f} | {r.accuracy:.3f} |")
    st = pd.DataFrame(acq_rows)
    if "decider_adaptive" in set(st.run):
        d = st[st.run == "decider_adaptive"]; strat = {}
        for v in d.strategy_counts:
            for k, c in json.loads(v).items(): strat[k] = strat.get(k, 0) + c
        n = sum(strat.values())
        lines += ["", f"Decider adaptive: UNI calls mean {d.uni_calls.mean():.1f}, median {d.uni_calls.median():.0f}, std {d.uni_calls.std():.1f}, min {d.uni_calls.min()}, max {d.uni_calls.max()}; "
                      f"stopped by the model in {100*d.stopped_by_model.mean():.0f}% of slides; mean latency {d.mean_latency_ms.mean():.0f} ms/round; "
                      f"strategy mix " + ", ".join(f"{k} {100*v/n:.0f}%" for k, v in strat.items())]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n"); print("\n".join(lines))
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for fam, g in curve.groupby("family"):
            g = g.sort_values("uni_calls"); style = dict(marker="o") if fam != "full_uni" else dict(marker="*", markersize=14, linestyle="none")
            ax.errorbar(g.uni_calls, g.auroc, yerr=g.auroc_std.fillna(0), label=fam, capsize=3, **style)
        ax.set_xscale("log"); ax.set_xlabel("UNI calls per slide (log)"); ax.set_ylabel("macro AUROC (5x5-fold CV)"); ax.set_title(f"{tcfg['task']}: performance vs UNI budget")
        ax.grid(alpha=.3); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out_dir / "budget_curve.png", dpi=150); fig.savefig(out_dir / "figures" / "fig1_auroc_vs_uni_calls.png", dpi=150)
    except Exception as e:
        print("plot skipped:", e)
    print(f"[task0] done -> {out_dir}")


if __name__ == "__main__":
    main()
