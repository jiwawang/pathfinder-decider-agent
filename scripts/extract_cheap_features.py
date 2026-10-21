"""Compute cheap features for every tissue patch of every slide -> artifacts/cheap_features/{slide_id}.npz

Run: python scripts/extract_cheap_features.py [--workers 8] [--slides id ...]
Keys: features [N, F] float32, feature_names, coords [N, 2] int32, seconds (wall time for the slide)
"""
from __future__ import annotations

import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cheap_features.extract import FEATURE_NAMES, extract_slide  # noqa: E402
from src.common.config import load_config, resolve  # noqa: E402


def one(args: tuple) -> str:
    slide_id, wsi_path, h5_path, out, work_size = args
    out = Path(out)
    if out.exists():
        return f"skip {slide_id}"
    t0 = time.time()
    with h5py.File(h5_path, "r") as f:
        coords = f["coords"][:].astype(np.int32)
        ps = int(f.attrs.get("patch_size", 224))
    feats = extract_slide(wsi_path, coords, ps, work_size)
    tmp = out.with_suffix(".tmp.npz")
    np.savez(tmp, features=feats, feature_names=np.array(FEATURE_NAMES), coords=coords,
             slide_id=slide_id, seconds=np.float32(time.time() - t0))
    tmp.rename(out)
    return f"{slide_id}: {feats.shape} in {time.time()-t0:.0f}s"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--work-size", type=int, default=56)
    ap.add_argument("--slides", nargs="*", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    meta = pd.read_csv(resolve(cfg["paths"]["metadata_csv"]), dtype={"case_id": str})
    if args.slides:
        meta = meta[meta.slide_id.isin(args.slides)]
    out_dir = resolve(cfg["paths"]["candidates_dir"]).parent / "cheap_features"
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(r.slide_id, r.wsi_path, r.h5_path, str(out_dir / f"{r.slide_id}.npz"), args.work_size) for r in meta.itertuples()]
    with Pool(args.workers) as pool:
        for msg in pool.imap_unordered(one, jobs):
            print(f"[cheap] {msg}", flush=True)
    print(f"[cheap] done -> {out_dir}")


if __name__ == "__main__":
    main()
