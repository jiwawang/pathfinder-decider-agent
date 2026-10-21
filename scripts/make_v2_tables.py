"""Merge V2 runs into the WP001 section 20 main table + budget curve table -> results/v2_tables.md

Run: python scripts/make_v2_tables.py --decider-run decider_v3_active --heuristic-run heuristic_v3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.config import load_config, resolve  # noqa: E402

FAM = {"full_uni": "Full UNI (all-patch mean)", "random": "Random-K", "uniform": "Spatial uniform-K",
       "cheap_topk": "Cheap Top-K (largest cheap clusters)", "cheap_greedy": "Cheap greedy-K (unified exploration score)",
       "decider_fixed": "Decider, fixed budget K", "decider": "Decider, adaptive STOP"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--decider-run", default="decider_v3_active")
    ap.add_argument("--heuristic-run", default="heuristic_v3")
    ap.add_argument("--joint-run", default="decider_v3_joint_stop", help="failure-mode run: STOP inside the 5-way choice")
    ap.add_argument("--out", default="results/v2_tables.md")
    args = ap.parse_args()
    cfg = load_config(args.config)
    rdir = resolve(cfg["paths"]["results_dir"])
    d = pd.read_csv(rdir / args.decider_run / "budget_curve.csv")
    h = pd.read_csv(rdir / args.heuristic_run / "budget_curve.csv")
    h = h[h.family.str.startswith("decider")].assign(family=lambda x: "heuristic_router" + x.family.str.replace("decider", "", regex=False))
    FAM["heuristic_router"] = "Heuristic router, adaptive STOP (no model)"
    FAM["heuristic_router_fixed"] = "Heuristic router, fixed budget K (no model)"
    cur = pd.concat([d, h]).reset_index(drop=True)
    joint = None
    if (rdir / args.joint_run / "budget_curve.csv").exists():
        j = pd.read_csv(rdir / args.joint_run / "budget_curve.csv")
        joint = j[j.method == "decider"].iloc[0]
    # collapse random draws: family 'random' already has one method per K (draws are averaged in the CV summary)
    ds = json.loads((rdir / args.decider_run / "decider_summary.json").read_text())
    bench = json.loads((rdir / "uni_benchmark.json").read_text()) if (rdir / "uni_benchmark.json").exists() else None

    lines = ["# PathFinder V2 — sequential active patch acquisition\n",
             f"Patient-level 5-fold CV x 5 seeds, L2 logistic regression on the mean of the acquired UNI embeddings. "
             f"Run `{args.decider_run}` (Strands Decider) and `{args.heuristic_run}` (model-free router).\n",
             "## Main table (WP001 section 20)\n", "| Method | UNI calls / slide | AUROC | Balanced Acc. | F1 |", "|---|---|---|---|---|"]
    order = ["full_uni", "random", "uniform", "cheap_topk", "cheap_greedy", "heuristic_router_fixed", "heuristic_router", "decider_fixed", "decider"]
    for fam in order:
        g = cur[cur.family == fam].sort_values("uni_calls")
        for _, r in g.iterrows():
            sd = f" ± {r.auroc_std:.3f}" if pd.notna(r.auroc_std) else ""
            name = FAM[fam] + (f" = {int(round(r.uni_calls))}" if fam not in ("full_uni", "decider", "heuristic_router") else "")
            lines.append(f"| {name} | {r.uni_calls:,.1f} | {r.auroc:.3f}{sd} | {r.balanced_accuracy:.3f} | {r.f1:.3f} |")
    if joint is not None:
        lines += ["", f"Failure mode kept for the record (`{args.joint_run}`): with STOP offered as a fifth option inside the routing "
                  f"question, the Decider never stopped (P(STOP) <= 0.07 on every round) and always spent the maximum budget: "
                  f"{joint.uni_calls:.0f} UNI calls, AUROC {joint.auroc:.3f} ± {joint.auroc_std:.3f}. The separate CONTINUE/STOP "
                  f"question fixed this (see decision_patterns/ADAPTIVE_ROUTER.md)."]
    lines += ["", "## Decider acquisition statistics\n",
              f"* UNI calls per slide: mean {ds['mean_uni_calls']:.1f} (min {ds['min_uni_calls']}, max {ds['max_uni_calls']}); "
              f"full-slide pass = {ds['full_uni_calls_per_slide']:,.0f} calls, i.e. **{ds['uni_calls_reduction_factor']:,.0f}x fewer UNI patch evaluations**",
              f"* {ds['decisions']:,} decisions over {ds['slides']} slides; latency mean {ds['latency_ms']['mean']:.1f} ms / median "
              f"{ds['latency_ms']['median']:.1f} ms / p95 {ds['latency_ms']['p95']:.1f} ms; mean confidence {ds['mean_confidence']:.3f}; "
              f"low-confidence fallback rate {100*ds['low_confidence_rate']:.1f}%",
              f"* agent-initiated STOP in {100*ds['stopped_by_model_rate']:.1f}% of slides; {ds['mean_clusters_covered']:.1f} cheap clusters covered per slide"]
    if bench:
        full = ds["full_uni_calls_per_slide"]
        uni_b1 = bench["uni_end_to_end_ms_batch1"]
        uni_batched = 1000.0 / bench["uni_patches_per_second_by_batch"]["32"]
        dec = ds["latency_ms"]["mean"]
        n = ds["mean_uni_calls"]
        lines += ["", "## Measured compute (RTX 4090, `results/uni_benchmark.json`)\n",
                  f"* UNI per patch: {bench['uni_encode_ms_batch1']['median']:.1f} ms encode (batch 1, median), "
                  f"{uni_b1:.1f} ms end-to-end incl. patch read; batched throughput {bench['uni_patches_per_second_by_batch']['32']:.0f} patches/s "
                  f"({uni_batched:.2f} ms/patch); peak GPU memory {bench['peak_gpu_memory_gb']:.1f} GB; live vs cached embedding cosine {bench['live_vs_cached_cosine']:.3f}",
                  f"* Full-slide UNI pass: {full:,.0f} patches x {uni_batched:.2f} ms = **{full*uni_batched/1000:.0f} s GPU time per slide** (batched)",
                  f"* Decider agent: {n:.1f} UNI calls x {uni_b1:.1f} ms + {ds['mean_rounds']:.1f} decisions x {dec:.1f} ms = "
                  f"**{(n*uni_b1 + ds['mean_rounds']*dec)/1000:.2f} s per slide** (Decider overhead {ds['mean_rounds']*dec/1000:.2f} s, "
                  f"{100*ds['mean_rounds']*dec/(n*uni_b1 + ds['mean_rounds']*dec):.0f}% of the agent's time)",
                  f"* Net GPU reduction: {full*uni_batched/(n*uni_b1 + ds['mean_rounds']*dec):,.0f}x. Caveat: the cheap features still read every tissue "
                  f"patch once (~2-3 ms/patch of CPU/IO, ~50-80 s per slide here) because these SVS files have no low-magnification "
                  f"pyramid level; with a pyramid that cost drops by the downsample factor squared."]
    out = resolve(args.out)
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines)); print(f"\n-> {out}")


if __name__ == "__main__":
    main()
