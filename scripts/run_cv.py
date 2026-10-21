"""Main experiment: leakage-safe CV comparing All-Mean / Random-K / Vote Top-K / KMeans Top-K / Decider-K.

Examples
  python scripts/run_cv.py --tag threshold_smoke --decider-backend threshold --seeds 0
  python scripts/run_cv.py --tag decider_v1 --seeds 0 1 2 3 4

Outputs under results/<tag>/:
  oof_predictions.csv   one row per (method, seed, fold, rep, slide)
  summary.csv           mean ± std over seeds of AUROC / BA / F1 / Acc / patches used
  per_run_metrics.csv   metrics of every (method, seed, rep)
  decider_stats.csv     per-episode statistics (latency, confidence, action counts)
  decider_summary.json  aggregate latency / confidence / action distribution
  traces/seed*_fold*/   full decider trace per slide and fold (cache; reruns reuse them)
  agent_traces/         canonical trace per slide: the fold where it was a TEST slide (first seed)
  config_used.yaml
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from decision_patterns.evidence_selector import DEFAULT_QUESTION, EvidenceSelector  # noqa: E402
from src.common.config import load_config, resolve  # noqa: E402
from src.decider.client import build_backend  # noqa: E402
from src.decider.episode import STATE_LABELS, run_episode  # noqa: E402
from src.evaluation.cross_validation import load_candidates, run_fold, summarise  # noqa: E402
from src.evaluation.metrics import slide_metrics  # noqa: E402
from src.evaluation.splits import make_folds  # noqa: E402

ALL_METHODS = ["all_mean", "random_k", "vote_topk", "kmeans_topk", "candidates_mean", "decider"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--methods", default=",".join(ALL_METHODS))
    ap.add_argument("--decider-backend", default=None, help="strands | threshold (overrides config)")
    ap.add_argument("--candidates-dir", default=None, help="override (cluster-count ablations)")
    ap.add_argument("--order", default=None, help="priority | random (overrides config)")
    ap.add_argument("--fresh", action="store_true", help="ignore cached traces")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.decider_backend:
        cfg["decider"]["backend"] = args.decider_backend
    if args.order:
        cfg["decider"]["candidate_order"] = args.order
    seeds = args.seeds if args.seeds is not None else list(cfg["evaluation"]["seeds"])
    methods = [m for m in args.methods.split(",") if m]
    out_dir = resolve(cfg["paths"]["results_dir"]) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.fresh and (out_dir / "traces").exists():
        shutil.rmtree(out_dir / "traces")
    (out_dir / "config_used.yaml").write_text(yaml.safe_dump({k: v for k, v in cfg.items() if not k.startswith("_")}))

    meta = pd.read_csv(resolve(cfg["paths"]["metadata_csv"]), dtype={"case_id": str, "patient_id": str})
    meta = meta.sort_values("slide_id").reset_index(drop=True)
    sids = meta["slide_id"].tolist()
    cand_dir = resolve(args.candidates_dir) if args.candidates_dir else resolve(cfg["paths"]["candidates_dir"])
    cands = load_candidates(cand_dir, sids)
    pools = None
    if "random_k" in methods:
        pool_dir = resolve(cfg["paths"]["candidates_dir"]).parent / "random_pool"
        pools = load_candidates(pool_dir, sids)
    D = cands[sids[0]]["embedding"].shape[1]
    print(f"[cv] slides={len(sids)} D={D} candidates/slide={np.mean([len(c['patch_index']) for c in cands.values()]):.1f} "
          f"methods={methods} seeds={seeds}", flush=True)

    episode_fn = None
    if "decider" in methods:
        dc = cfg["decider"]
        backend = build_backend(dc)
        selector = EvidenceSelector(backend, min_selected=int(dc["min_selected"]),
                                    target_selected=int(dc["target_selected"]), max_selected=int(dc["max_selected"]),
                                    low_confidence_threshold=float(dc["low_confidence_threshold"]),
                                    low_confidence_action=dc.get("low_confidence_action", "SKIP"),
                                    question=dc.get("question") or DEFAULT_QUESTION,
                                    criteria=dc.get("criteria"), state_labels=STATE_LABELS)
        order_mode = dc.get("candidate_order", "priority")

        weak_thr = float(dc.get("weak_remaining_threshold", 0.3))

        def episode_fn(sid, cand, votes, is_train, _sel=selector, _order=order_mode, _w=weak_thr):
            return run_episode(_sel, cand, votes, order_mode=_order, seed=0, slide_id=sid, weak_remaining_threshold=_w)

    pred_rows, stat_rows = [], []
    t0 = time.time()
    for seed in seeds:
        folds = make_folds(meta, int(cfg["evaluation"]["folds"]), seed)
        for f, (tr, te) in enumerate(folds):
            trace_dir = out_dir / "traces" / f"seed{seed}_fold{f}"
            rows, stats = run_fold(meta, cands, pools, tr, te, cfg=cfg, seed=seed, fold=f,
                                   episode_fn=episode_fn, methods=methods, trace_dir=trace_dir)
            pred_rows += rows
            stat_rows += stats
            dec = [r for r in rows if r["method"] == "decider"]
            msg = f"[cv] seed={seed} fold={f} test={len(te)} elapsed={time.time()-t0:.0f}s"
            if dec:
                msg += f" decider_k_fold={dec[0]['k_fold']}"
            print(msg, flush=True)

    pred = pd.DataFrame(pred_rows)
    pred.to_csv(out_dir / "oof_predictions.csv", index=False)
    summary, per_run = summarise(pred, slide_metrics)
    summary.to_csv(out_dir / "summary.csv", index=False)
    per_run.to_csv(out_dir / "per_run_metrics.csv", index=False)

    cols = ["method", "auroc_mean", "auroc_std", "balanced_accuracy_mean", "f1_mean", "accuracy_mean", "mean_patches_used_mean"]
    print("\n" + summary[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    if stat_rows:
        st = pd.DataFrame(stat_rows)
        st.to_csv(out_dir / "decider_stats.csv", index=False)
        lat, n_dec = [], 0
        for p in (out_dir / "traces").rglob("*.json"):
            tr = json.loads(p.read_text())["trace"]
            n_dec += sum("confidence" in t for t in tr)
            lat += [t["latency_ms"] for t in tr if t.get("latency_ms", 0) > 0]
        lat = np.array(lat) if lat else np.zeros(1)
        test = st[st.is_test]
        ds = dict(
            episodes=int(len(st)), decisions=int(n_dec),
            latency_ms={"mean": float(lat.mean()), "median": float(np.median(lat)), "p95": float(np.percentile(lat, 95))},
            mean_confidence=float(st.mean_confidence.mean()),
            low_confidence_rate=float(st.low_confidence_rate.mean()),
            mean_selected=float(test.num_selected.mean()), mean_examined=float(test.num_examined.mean()),
            mean_candidates=float(test.num_candidates.mean()),
            selection_ratio_vs_all_patches=float((test.num_selected / meta.set_index("slide_id").loc[test.slide_id, "n_patches"].values).mean()),
            stopped_by_model_rate=float(test.stopped_by_model.mean()),
            action_distribution={a: float(st[f"n_{a}"].sum() / st[["n_select", "n_skip", "n_stop"]].values.sum()) for a in ("select", "skip", "stop")},
        )
        (out_dir / "decider_summary.json").write_text(json.dumps(ds, indent=2))
        print("\n[decider]", json.dumps(ds, indent=1))
        # canonical per-slide trace: the fold where the slide was a test slide, first seed
        canon = out_dir / "agent_traces"
        canon.mkdir(exist_ok=True)
        first = st[(st.seed == seeds[0]) & st.is_test]
        for _, r in first.iterrows():
            src = out_dir / "traces" / f"seed{r.seed}_fold{r.fold}" / f"{r.slide_id}.json"
            shutil.copy(src, canon / f"{r.slide_id}.json")
    print(f"[cv] done -> {out_dir}")


if __name__ == "__main__":
    main()
