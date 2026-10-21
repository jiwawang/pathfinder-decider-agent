"""Cluster every slide's embeddings and write artifacts/candidates/{slide_id}.npz.

Run: python scripts/build_candidates.py [--config configs/default.yaml] [--num-clusters 32]
Also caches the all-patch mean embedding (Baseline A) so run_cv.py never reloads 8 GB of features.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.candidates.clustering import build_candidates  # noqa: E402
from src.common.config import load_config, resolve  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--num-clusters", type=int, default=0, help="override config (ablation)")
    ap.add_argument("--out-dir", default="", help="override candidates_dir (ablation)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    cs = cfg["candidate_selection"]
    k = args.num_clusters or int(cs["num_clusters"])
    df = pd.read_csv(resolve(cfg["paths"]["metadata_csv"]), dtype={"case_id": str, "patient_id": str})
    emb_dir = resolve(cfg["paths"]["embeddings_dir"])
    out_dir = resolve(args.out_dir) if args.out_dir else resolve(cfg["paths"]["candidates_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    for _, row in df.iterrows():
        out = out_dir / f"{row.slide_id}.npz"
        if out.exists() and not args.overwrite:
            continue
        src = emb_dir / f"{row.slide_id}.npz"
        if not src.exists():
            print(f"[candidates] SKIP {row.slide_id}: no embeddings")
            continue
        t0 = time.time()
        z = np.load(src)
        cand = build_candidates(z["embeddings"], z["coords"], k, seed=int(cs["seed"]),
                                n_init=int(cs["kmeans_n_init"]))
        np.savez(out, slide_id=row.slide_id, label=np.int32(row.label), **cand)
        print(f"[candidates] {row.slide_id}: N={int(cand['n_patches'])} K={int(cand['num_clusters'])} "
              f"rep[min={cand['representativeness'].min():.3f} mean={cand['representativeness'].mean():.3f}] "
              f"{time.time()-t0:.1f}s", flush=True)
    print(f"[candidates] done -> {out_dir}")


if __name__ == "__main__":
    main()
