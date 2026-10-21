"""WP006 analyses on a finished task run (evaluation only): paired comparisons at the pre-specified K, adaptive-STOP
budget analysis, site confounding + site-held-out sensitivity CV, confusion statistics, figures 2 and 3.
  python scripts/tcga/analyze_task.py --config configs/tcga_task1.yaml
Writes <results_dir>/analysis.md, analysis.json, figures/fig2_k16_comparison.png, figures/fig3_adaptive_budget.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, wilcoxon
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common.config import load_config, resolve  # noqa: E402
from src.tcga.evaluation import fit_predict_multi, multiclass_metrics  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/tcga_task1.yaml")
    ap.add_argument("--base-config", default="configs/default.yaml")
    args = ap.parse_args()
    tcfg = load_config(args.config)["tcga"]; cfg = load_config(args.base_config)
    R = resolve(tcfg["results_dir"]); K = int(tcfg.get("primary_budget", 16))
    oof = pd.read_csv(R / "oof_predictions.csv"); st = pd.read_csv(R / "acquisition_stats.csv"); meta = pd.read_csv(R / "manifest_snapshot.csv")
    pr = pd.read_csv(R / "per_run_metrics.csv"); n_classes = int(oof.true_label.max() + 1)
    out, lines = {}, [f"# Analysis — {tcfg['task']} (primary budget K={K})\n"]

    # ---- 1. paired comparisons at K (per-seed OOF AUROC differences + paired patient bootstrap of the AUROC difference) ----
    ref = f"decider_fixed_{K}"; comps = [f"random_{K}_s0", f"random_{K}_s1", f"random_{K}_s2", f"uniform_{K}", f"cheap_topk_{K}", f"cheap_greedy_{K}", "decider_adaptive", "full_uni"]
    rng = np.random.default_rng(0); sids = sorted(oof.slide_id.unique())
    def auc_of(df):
        return roc_auc_score(df.true_label, df.p1) if n_classes == 2 else multiclass_metrics(df.true_label.values, df[[f"p{c}" for c in range(n_classes)]].values)["auroc"]
    lines += [f"## 1. Paired comparisons vs {ref}\n", "| Comparator | ΔAUROC (mean over 5 seeds, Decider − comparator) | seeds Decider better | paired bootstrap 95% CI of Δ | Wilcoxon p (per-slide log-loss) |", "|---|---:|---:|---:|---:|"]
    for m in comps:
        if m not in set(oof.method):
            continue
        d_seed = []
        for s in sorted(oof.seed.unique()):
            a = oof[(oof.method == ref) & (oof.seed == s)]; b = oof[(oof.method == m) & (oof.seed == s)]
            d_seed.append(auc_of(a) - auc_of(b))
        # paired bootstrap over patients, AUROC averaged over seeds within each replicate
        A = {s: oof[(oof.method == ref) & (oof.seed == s)].set_index("slide_id") for s in oof.seed.unique()}
        B = {s: oof[(oof.method == m) & (oof.seed == s)].set_index("slide_id") for s in oof.seed.unique()}
        boots = []
        for _ in range(1000):
            samp = rng.choice(sids, len(sids), replace=True); da = []
            for s in A:
                ya = A[s].loc[samp]; yb = B[s].loc[samp]
                if ya.true_label.nunique() < 2: continue
                da.append(auc_of(ya) - auc_of(yb))
            boots.append(np.mean(da))
        # per-slide log-loss of the true class, averaged over seeds, paired Wilcoxon
        def ll(df):
            p = np.clip(df[[f"p{c}" for c in range(n_classes)]].values[np.arange(len(df)), df.true_label.values], 1e-6, 1); return pd.Series(-np.log(p), index=df.slide_id.values)
        la = pd.concat([ll(A[s].reset_index()) for s in A], axis=1).mean(1); lb = pd.concat([ll(B[s].reset_index()) for s in B], axis=1).mean(1)
        try:
            pw = wilcoxon(la.loc[sids], lb.loc[sids]).pvalue
        except ValueError:
            pw = float("nan")
        out[f"paired_{m}"] = {"delta_auroc_mean": float(np.mean(d_seed)), "seeds_decider_better": int(sum(d > 0 for d in d_seed)), "boot_ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))], "wilcoxon_p": float(pw)}
        lines.append(f"| {m} | {np.mean(d_seed):+.3f} | {sum(d > 0 for d in d_seed)}/5 | [{np.percentile(boots, 2.5):+.3f}, {np.percentile(boots, 97.5):+.3f}] | {pw:.3g} |")

    # ---- 2. confusion / recall at K (binary) ----
    if n_classes == 2:
        lines += ["", f"## 2. Binary detail at K={K} (means over seeds)\n", "| Method | recall class0 | recall class1 | PR-AUC | TN / FP / FN / TP |", "|---|---:|---:|---:|---|"]
        for m in ["full_uni", f"random_{K}_s0", f"uniform_{K}", f"cheap_topk_{K}", f"cheap_greedy_{K}", ref, "decider_adaptive"]:
            g = pr[pr.method == m]
            if len(g):
                lines.append(f"| {m} | {g.recall_class0.mean():.3f} | {g.recall_class1.mean():.3f} | {g.pr_auc.mean():.3f} | {g.tn.mean():.0f} / {g.fp.mean():.0f} / {g.fn.mean():.0f} / {g.tp.mean():.0f} |")

    # ---- 3. adaptive STOP ----
    d = st[st.run == "decider_adaptive"].merge(meta[["slide_id", "category", "n_patches"]], on="slide_id")
    traces = {s: json.loads((R / "traces" / "decider_adaptive" / f"{s}.json").read_text()) for s in d.slide_id}
    stop_rows = []
    for s, ep in traces.items():
        for t in ep["trace"]:
            if t.get("stop_allowed") and t.get("stop_probability") is not None:
                stop_rows.append(dict(slide_id=s, calls=t["uni_calls"], p_stop=t["stop_probability"], gain=t["recent_gain_mean"], pct=t.get("recent_gain_percent_of_initial"), stopped=t["decision"] == "STOP"))
    sr = pd.DataFrame(stop_rows)
    # STOP reasons per slide
    reason = d.apply(lambda r: "model STOP" if r.stopped_by_model else ("max budget" if r.uni_calls >= int(cfg["adaptive"]["max_uni_calls"]) else "candidates exhausted"), axis=1)
    out["adaptive"] = {"calls": d.uni_calls.describe().round(2).to_dict(), "by_class": d.groupby("category").uni_calls.agg(["mean", "median", "std"]).round(2).to_dict(),
                       "stop_reasons": reason.value_counts().to_dict(), "corr_calls_vs_tissue_patches": float(np.corrcoef(d.uni_calls, d.n_patches)[0, 1]),
                       "gain_at_STOP": float(sr[sr.stopped].gain.mean()) if sr.stopped.any() else None, "gain_at_CONTINUE": float(sr[~sr.stopped].gain.mean()),
                       "pct_initial_at_STOP": float(sr[sr.stopped].pct.mean()) if sr.stopped.any() else None, "pct_initial_at_CONTINUE": float(sr[~sr.stopped].pct.mean()),
                       "corr_pstop_vs_gain_pct": float(np.corrcoef(sr.p_stop, sr.pct.fillna(100))[0, 1])}
    # per-slide: does the budget relate to correctness? (adaptive OOF correctness vs calls)
    corr_ok = oof[oof.method == "decider_adaptive"].groupby("slide_id").apply(lambda g: (g.pred_label == g.true_label).mean()).rename("acc").reset_index().merge(d[["slide_id", "uni_calls"]])
    out["adaptive"]["mean_calls_correct_vs_incorrect_slides"] = [float(corr_ok[corr_ok.acc >= 0.5].uni_calls.mean()), float(corr_ok[corr_ok.acc < 0.5].uni_calls.mean())]
    a = out["adaptive"]
    lines += ["", "## 3. Adaptive STOP\n", f"* UNI calls: mean {a['calls']['mean']}, median {a['calls']['50%']}, std {a['calls']['std']}, min {a['calls']['min']}, max {a['calls']['max']}; by class {a['by_class']}",
              f"* STOP reasons: {a['stop_reasons']}; corr(calls, tissue patches) = {a['corr_calls_vs_tissue_patches']:+.2f}",
              f"* recent information gain at STOP {a['gain_at_STOP']:.3f} vs at CONTINUE {a['gain_at_CONTINUE']:.3f}; recent gain as % of initial: {a['pct_initial_at_STOP']:.0f}% vs {a['pct_initial_at_CONTINUE']:.0f}%; corr(P(STOP), %initial) = {a['corr_pstop_vs_gain_pct']:+.2f}",
              f"* mean calls on slides classified correctly (≥3/5 seeds) vs incorrectly: {a['mean_calls_correct_vs_incorrect_slides'][0]:.1f} vs {a['mean_calls_correct_vs_incorrect_slides'][1]:.1f}"]

    # ---- 4. site confounding + site-held-out sensitivity ----
    # TCGA barcode TSS codes are assigned per project, so the 2-character code is confounded with the label by
    # construction; group by the GDC tissue_source_site NAME (hospital) from the frozen manifest instead.
    man = pd.read_csv(resolve(tcfg["manifest"])).set_index("file_name") if tcfg.get("manifest") else None
    if man is not None:
        meta = meta.assign(tss=meta.slide_id.map(lambda s: man.loc[s + ".svs", "tissue_source_site_name"]))
    ct = pd.crosstab(meta.tss, meta.label); chi2, p, _, _ = chi2_contingency(ct); V = float(np.sqrt(chi2 / len(meta)))
    shared = int(((ct > 0).sum(1) == n_classes).sum()); excl = float(meta.tss.map(lambda s: (ct.loc[s] > 0).sum() < n_classes).mean())
    out["site"] = {"sites": int(len(ct)), "shared": shared, "cramers_v": V, "chi2_p": float(p), "slides_from_single_class_sites": excl}
    lines += ["", "## 4. Tissue source site\n", f"* {len(ct)} sites, {shared} shared by all classes, Cramér's V = {V:.2f} (χ² p = {p:.1e}); {100*excl:.0f}% of slides come from single-class sites.",
              "* Site-held-out sensitivity (StratifiedGroupKFold with groups = TSS, 5 folds × 5 seeds, same classifier) — a site never appears on both sides:", "", "| Method | patient-grouped AUROC | site-grouped AUROC |", "|---|---:|---:|"]
    # rebuild slide embeddings = we only have OOF predictions, not X; recompute X from acquisition traces is heavy -> use the saved per-run X? Not saved. Re-derive from the UNI cache via acquired indices.
    from src.acquisition.lazy_uni_store import LazyUNIFeatureStore
    store = LazyUNIFeatureStore(dict(zip(meta.slide_id, meta.features_npy)))
    def X_for(method):
        X = []
        for s in meta.slide_id:
            if method == "full_uni":
                X.append(np.load(store._paths[s], mmap_mode="r").mean(0)); continue
            tr = json.loads((R / "traces" / method / f"{s}.json").read_text()) if (R / "traces" / method / f"{s}.json").exists() else None
            if tr is None:
                return None
            X.append(np.stack([store.acquire(s, i) for i in tr["acquired_patch_index"]]).mean(0))
        return np.stack(X)
    y = meta.label.values.astype(int)
    for m in ["full_uni", f"cheap_greedy_{K}", ref, "decider_adaptive"]:
        X = X_for(m)
        if X is None:
            continue
        aucs = []
        for seed in range(5):
            P = np.zeros((len(y), n_classes))
            for tr_, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, meta.tss.values):
                P[te] = fit_predict_multi(cfg["classifier"], X[tr_], y[tr_], X[te])
            aucs.append(roc_auc_score(y, P[:, 1]) if n_classes == 2 else roc_auc_score(y, P, multi_class="ovr"))
        pg = pr[pr.method == m].auroc.mean()
        out[f"site_heldout_{m}"] = {"auroc_mean": float(np.mean(aucs)), "auroc_std": float(np.std(aucs, ddof=1))}
        lines.append(f"| {m} | {pg:.3f} | {np.mean(aucs):.3f} ± {np.std(aucs, ddof=1):.3f} |")
    lines.append("\nTop sites: " + "; ".join(f"{k} {dict(v)}" for k, v in ct.assign(n=ct.sum(1)).sort_values("n", ascending=False).head(6).drop(columns="n").iterrows()))

    # ---- figures ----
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    s = pd.read_csv(R / "metrics.csv"); rows = [("Random", f"random_{K}_s0"), ("Spatial", f"uniform_{K}"), ("Cheap Top-K", f"cheap_topk_{K}"), ("Cheap Greedy", f"cheap_greedy_{K}"), ("Decider Fixed", ref), ("Decider Adaptive", "decider_adaptive"), ("Full UNI", "full_uni")]
    fig, ax = plt.subplots(figsize=(8, 3.8)); x = np.arange(len(rows)); w = 0.38
    ba = [s[s.method == m].balanced_accuracy_mean.iloc[0] for _, m in rows]; f1 = [s[s.method == m].f1_mean.iloc[0] for _, m in rows]
    ax.bar(x - w/2, ba, w, label="balanced accuracy"); ax.bar(x + w/2, f1, w, label="F1"); ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows], fontsize=8)
    ax.set_ylim(min(ba + f1) - 0.05, 1.0); ax.set_title(f"{tcfg['task']}: K={K} comparison (5x5-fold CV)"); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(R / "figures" / f"fig2_k{K}_comparison.png", dpi=150)
    fig, ax = plt.subplots(figsize=(6, 3.5)); mn, mx = int(cfg["adaptive"]["min_uni_calls"]), int(cfg["adaptive"]["max_uni_calls"])
    for cat, g in d.groupby("category"):
        ax.hist(g.uni_calls, bins=np.arange(mn - .5, mx + 1.5), alpha=.6, label=cat)
    ax.set_xlabel("UNI calls per slide (adaptive STOP)"); ax.set_ylabel("slides"); ax.legend(); ax.set_title(f"{tcfg['task']}: adaptive budget distribution"); fig.tight_layout(); fig.savefig(R / "figures" / "fig3_adaptive_budget.png", dpi=150)
    (R / "analysis.md").write_text("\n".join(lines) + "\n"); (R / "analysis.json").write_text(json.dumps(out, indent=2, default=float))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
