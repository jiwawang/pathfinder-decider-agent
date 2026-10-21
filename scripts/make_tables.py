"""Collect results/<tag>/summary.csv + decider_summary.json from several runs into one Markdown table.

Run: python scripts/make_tables.py --runs decider_v2 threshold_v2 decider_v2_random_order --out results/ablation_table.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.config import load_config, resolve  # noqa: E402

LABELS = {"all_mean": "A. All-patch mean", "random_k": "B. Random-K", "vote_topk": "C. Vote Top-K",
          "kmeans_topk": "C'. KMeans Top-K (cluster size)", "candidates_mean": "C''. All 32 candidates mean",
          "decider": "Proposed: Decider-K"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default="results/ablation_table.md")
    args = ap.parse_args()
    cfg = load_config(args.config)
    rdir = resolve(cfg["paths"]["results_dir"])
    lines = []
    for tag in args.runs:
        d = rdir / tag
        if not (d / "summary.csv").exists():
            lines.append(f"\n## {tag}\n\n_missing summary.csv_\n")
            continue
        s = pd.read_csv(d / "summary.csv")
        seeds = sorted(pd.read_csv(d / "per_run_metrics.csv").seed.unique().tolist())
        lines.append(f"\n## Run `{tag}` ({len(seeds)} seed(s): {seeds}; 5-fold patient-level CV, L2 logistic regression)\n")
        lines.append("| Method | AUROC | Balanced Acc. | F1 | Accuracy | Mean patches used |")
        lines.append("|---|---|---|---|---|---|")
        order = [m for m in LABELS if m in set(s.method)] + [m for m in s.method if m not in LABELS]
        for m in order:
            r = s[s.method == m].iloc[0]
            sd = f" ± {r.auroc_std:.3f}" if pd.notna(r.auroc_std) else ""
            lines.append(f"| {LABELS.get(m, m)} | {r.auroc_mean:.3f}{sd} | {r.balanced_accuracy_mean:.3f} | {r.f1_mean:.3f} | "
                         f"{r.accuracy_mean:.3f} | {r.mean_patches_used_mean:,.1f} |")
        if (d / "decider_summary.json").exists():
            ds = json.loads((d / "decider_summary.json").read_text())
            ad = ds["action_distribution"]
            lines.append(f"\nDecider: {ds['decisions']:,} decisions in {ds['episodes']} episodes; latency mean "
                         f"{ds['latency_ms']['mean']:.1f} ms / median {ds['latency_ms']['median']:.1f} ms / p95 "
                         f"{ds['latency_ms']['p95']:.1f} ms; mean confidence {ds['mean_confidence']:.3f}; low-confidence rate "
                         f"{100*ds['low_confidence_rate']:.1f}%; SELECT/SKIP/STOP = {100*ad['select']:.1f}% / {100*ad['skip']:.1f}% / "
                         f"{100*ad['stop']:.1f}%; selected {ds['mean_selected']:.1f} of {ds['mean_candidates']:.0f} candidates "
                         f"({100*ds['selection_ratio_vs_all_patches']:.3f}% of all tissue patches); examined {ds['mean_examined']:.1f}; "
                         f"stopped by the model in {100*ds['stopped_by_model_rate']:.1f}% of episodes.\n")
    out = resolve(args.out)
    out.write_text("# PathFinder ablation tables\n" + "\n".join(lines))
    print("\n".join(lines))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
