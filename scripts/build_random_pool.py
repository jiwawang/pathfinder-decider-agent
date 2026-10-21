"""Cache a fixed random subset of real patch embeddings per slide for the Random-K baseline.

Reading the full embeddings (8 GB) inside every CV repeat is too slow, so each slide gets a pool of
`pool_size` uniformly sampled patches (seeded). Random-K draws K of them per repeat.
Run: python scripts/build_random_pool.py [--pool-size 256]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.config import load_config, resolve  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--pool-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = load_config(args.config)
    df = pd.read_csv(resolve(cfg["paths"]["metadata_csv"]), dtype={"case_id": str})
    emb_dir = resolve(cfg["paths"]["embeddings_dir"])
    out_dir = resolve(cfg["paths"]["candidates_dir"]).parent / "random_pool"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    for _, row in df.iterrows():
        out = out_dir / f"{row.slide_id}.npz"
        if out.exists():
            continue
        z = np.load(emb_dir / f"{row.slide_id}.npz")
        n = z["embeddings"].shape[0]
        idx = np.sort(rng.choice(n, size=min(args.pool_size, n), replace=False))
        np.savez(out, slide_id=row.slide_id, patch_index=idx.astype(np.int32),
                 embedding=z["embeddings"][idx], coords=z["coords"][idx], n_patches=np.int32(n))
        print(f"[random_pool] {row.slide_id}: {len(idx)}/{n}", flush=True)
    print(f"[random_pool] done -> {out_dir}")


if __name__ == "__main__":
    main()
