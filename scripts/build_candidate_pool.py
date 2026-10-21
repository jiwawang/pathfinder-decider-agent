"""Cheap candidate pools: artifacts/candidate_pool_K{pool_size}/{slide_id}.npz (no UNI involved).

Run: python scripts/build_candidate_pool.py [--pool-size 128]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.acquisition.candidate_pool import build_cheap_pool  # noqa: E402
from src.common.config import load_config, resolve  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--pool-size", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    k = args.pool_size or int(cfg["acquisition"]["candidate_pool_size"])
    meta = pd.read_csv(resolve(cfg["paths"]["metadata_csv"]), dtype={"case_id": str})
    art = resolve(cfg["paths"]["candidates_dir"]).parent
    cheap_dir, out_dir = art / "cheap_features", art / f"candidate_pool_K{k}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for r in meta.itertuples():
        out = out_dir / f"{r.slide_id}.npz"
        if out.exists() and not args.overwrite:
            continue
        src = cheap_dir / f"{r.slide_id}.npz"
        if not src.exists():
            print(f"[pool] SKIP {r.slide_id}: no cheap features yet")
            continue
        t0 = time.time()
        z = np.load(src)
        pool = build_cheap_pool(z["features"], z["coords"], [str(n) for n in z["feature_names"]], k,
                                seed=int(cfg["acquisition"]["seed"]))
        np.savez(out, slide_id=r.slide_id, **pool)
        print(f"[pool] {r.slide_id}: N={int(pool['n_patches'])} K={int(pool['pool_size'])} "
              f"rep mean={pool['representativeness'].mean():.2f} {time.time()-t0:.1f}s", flush=True)
    print(f"[pool] done -> {out_dir}")


if __name__ == "__main__":
    main()
