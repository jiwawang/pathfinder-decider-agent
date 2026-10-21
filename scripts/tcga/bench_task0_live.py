"""Task 0 live benchmark: on-demand UNI through the MPP-matched reader, per-slide timing split (WP005 section 13).
  CUDA_VISIBLE_DEVICES=0 python scripts/tcga/bench_task0_live.py --n-per-class 1
Writes results/tcga_generalization/task0_3class/live_benchmark.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.acquisition.adaptive_agent import AdaptiveRouterV21, run_adaptive_episode  # noqa: E402
from src.common.config import load_config, resolve  # noqa: E402
from src.decider.client import StrandsDeciderBackend  # noqa: E402
from src.tcga.live_store import TCGALiveUNIFeatureStore  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/tcga_task0.yaml")
    ap.add_argument("--base-config", default="configs/default.yaml")
    ap.add_argument("--n-per-class", type=int, default=1)
    args = ap.parse_args()
    tcfg = load_config(args.config)["tcga"]; cfg = load_config(args.base_config); acq, ad = cfg["acquisition"], cfg["adaptive"]
    meta = pd.read_csv(resolve(tcfg["slide_table"])).groupby("category").head(args.n_per_class)
    coords = {r.slide_id: np.load(r.cheap_path)["coords"] for r in meta.itertuples()}
    live = TCGALiveUNIFeatureStore({r.slide_id: r.wsi_path for r in meta.itertuples()}, coords, {r.slide_id: int(r.window_px) for r in meta.itertuples()})
    backend = StrandsDeciderBackend(cfg["decider"]["model_id"], "cuda")
    out = []
    for r in meta.itertuples():
        pool = {k: v for k, v in np.load(Path(r.cheap_path).parent.parent / f"candidate_pool_K{int(acq['candidate_pool_size'])}" / f"{r.slide_id}.npz", allow_pickle=True).items()}
        cheap_s = float(np.load(r.cheap_path)["seconds"])
        for name, mn, mx in (("fixed16", 16, 16), ("adaptive", int(ad["min_uni_calls"]), int(ad["max_uni_calls"]))):
            n0 = len(live.latency_ms); t0 = time.perf_counter()
            ep = run_adaptive_episode(r.slide_id, pool, live, AdaptiveRouterV21(backend, min_calls=mn, max_calls=mx, low_confidence_threshold=float(ad["low_confidence_threshold"])), ad)
            total = time.perf_counter() - t0; uni_s = sum(live.latency_ms[n0:]) / 1000; dec_s = sum(t["latency_ms"] for t in ep["trace"] if t.get("model_queried")) / 1000
            out.append(dict(slide_id=r.slide_id, category=r.category, mode=name, n_tissue_patches=int(r.n_patches), window_px=int(r.window_px),
                            cheap_preprocessing_s=round(cheap_s, 1), uni_calls=ep["stats"]["uni_calls"], decider_calls=sum(1 for t in ep["trace"] if t.get("model_queried")),
                            decider_s=round(dec_s, 3), uni_s=round(uni_s, 3), loop_s=round(total, 3), end_to_end_s=round(total + cheap_s, 1),
                            full_uni_equivalent_s_batched=round(int(r.n_patches) / 290.0, 1)))
            print(json.dumps(out[-1]), flush=True)
    res = resolve(tcfg["results_dir"]); res.mkdir(parents=True, exist_ok=True)
    (res / "live_benchmark.json").write_text(json.dumps({"gpu": "RTX 4090", "runs": out}, indent=2)); print(f"-> {res/'live_benchmark.json'}")


if __name__ == "__main__":
    main()
