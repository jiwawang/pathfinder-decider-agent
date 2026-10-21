"""Real cost of patch -> UNI on this GPU (WP001 section 24), plus the Decider overhead per decision.

Run: python scripts/bench_uni.py [--slide-id ...] [--n 64]
Writes results/uni_benchmark.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.acquisition.lazy_uni_store import LiveUNIFeatureStore  # noqa: E402
from src.common.config import load_config, resolve  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--slide-id", default="<private-slide>")
    ap.add_argument("--n", type=int, default=64)
    args = ap.parse_args()
    cfg = load_config(args.config)
    meta = pd.read_csv(resolve(cfg["paths"]["metadata_csv"]), dtype={"case_id": str}).set_index("slide_id")
    art = resolve(cfg["paths"]["candidates_dir"]).parent
    coords = {args.slide_id: np.load(art / "cheap_features" / f"{args.slide_id}.npz")["coords"]}
    store = LiveUNIFeatureStore({args.slide_id: meta.loc[args.slide_id, "wsi_path"]}, coords)
    idx = np.random.default_rng(0).choice(len(coords[args.slide_id]), size=args.n, replace=False)
    # warm-up
    store.acquire(args.slide_id, int(idx[0]))
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    crop_ms, enc_ms = [], []
    for i in idx[1:]:
        t0 = time.perf_counter(); img = store.crop(args.slide_id, int(i)); t1 = time.perf_counter()
        x = store.tf(img).unsqueeze(0).cuda()
        with torch.inference_mode():
            store.model(x); torch.cuda.synchronize()
        t2 = time.perf_counter()
        crop_ms.append((t1 - t0) * 1000); enc_ms.append((t2 - t1) * 1000)
    # batched throughput (encode only)
    imgs = torch.stack([store.tf(store.crop(args.slide_id, int(i))) for i in idx]).cuda()
    thr = {}
    for bs in (1, 8, 32, 64):
        xb = imgs[:bs]
        with torch.inference_mode():
            store.model(xb); torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(5):
                store.model(xb)
            torch.cuda.synchronize()
        thr[bs] = float(5 * bs / (time.perf_counter() - t0))
    # cached-vs-live sanity: live embedding should match the cached one
    cached = np.load(meta.loc[args.slide_id, "features_npy"], mmap_mode="r")[int(idx[1])]
    live = store.acquire(args.slide_id, int(idx[1]))   # goes through the audited accessor
    cos = float(cached @ live / (np.linalg.norm(cached) * np.linalg.norm(live)))
    out = dict(gpu=torch.cuda.get_device_name(0), slide_id=args.slide_id, n=int(args.n - 1),
               patch_read_ms={"mean": float(np.mean(crop_ms)), "median": float(np.median(crop_ms))},
               uni_encode_ms_batch1={"mean": float(np.mean(enc_ms)), "median": float(np.median(enc_ms)), "p95": float(np.percentile(enc_ms, 95))},
               uni_end_to_end_ms_batch1=float(np.mean(crop_ms) + np.mean(enc_ms)),
               uni_patches_per_second_by_batch=thr, peak_gpu_memory_gb=float(torch.cuda.max_memory_allocated() / 1e9),
               live_vs_cached_cosine=cos)
    resolve("results").mkdir(exist_ok=True)
    (resolve("results") / "uni_benchmark.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
