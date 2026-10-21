"""Cheap features for TCGA Task 0, stored next to the TRIDENT coords (D001 convention):
<job_dir>/<coords_dir>/cheap_features/<slide_id>.npz  (features [N,19], feature_names, coords [N,2], mpp, window_px, seconds)

Same 19 features and the same patch_features() as the OBR run; the only difference is the reader
(MPP-matched level-0 window, LANCZOS -> 224). Rows are in h5 coordinate order.
  python scripts/tcga/extract_cheap_features_tcga.py [--workers 12]
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.cheap_features.extract import FEATURE_NAMES, add_spatial_features, patch_features  # noqa: E402
from src.common.config import load_config, resolve  # noqa: E402
from src.tcga.reader import TARGET_PX, read_patch  # noqa: E402


def one(args: tuple) -> str:
    sid, wsi, h5p, out, window, work_size = args
    import openslide
    out = Path(out)
    if out.exists():
        return f"skip {sid}"
    t0 = time.time()
    with h5py.File(h5p, "r") as f:
        coords = f["coords"][:].astype(np.int64)
    s = openslide.OpenSlide(wsi)
    try:
        w0, h0 = s.level_dimensions[0]
        rows = [patch_features(read_patch(s, x, y, window), work_size) for x, y in coords]
    finally:
        s.close()
    feats = add_spatial_features(np.stack(rows).astype(np.float32), coords, (w0, h0), window)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npz")
    np.savez(tmp, features=feats, feature_names=np.array(FEATURE_NAMES), coords=coords.astype(np.int32), slide_id=sid,
             mpp=np.float32(TARGET_PX * 0.5 / window), window_px=np.int32(window), target_px=np.int32(TARGET_PX), seconds=np.float32(time.time() - t0))
    tmp.rename(out)
    return f"{sid}: {feats.shape} window {window}px in {time.time()-t0:.0f}s"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/tcga_task0.yaml")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--work-size", type=int, default=56)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    cfg = load_config(args.config)
    df = pd.read_csv(resolve(cfg["tcga"]["slide_table"]))
    if args.limit:
        df = df.groupby("category").head(args.limit)
    jobs = [(r.slide_id, r.wsi_path, r.h5_path, r.cheap_path, int(r.window_px), args.work_size) for r in df.itertuples()]
    with Pool(args.workers) as pool:
        for msg in pool.imap_unordered(one, jobs):
            print(f"[cheap-tcga] {msg}", flush=True)
    print("[cheap-tcga] done")


if __name__ == "__main__":
    main()
