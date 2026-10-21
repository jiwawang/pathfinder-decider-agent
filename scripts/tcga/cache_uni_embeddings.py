"""UNI embedding cache for the cached simulation + Full-UNI baseline (WP005 plan section 2).
<job_dir>/<coords_dir>/uni_embeddings/<slide_id>.npy  float32 [N,1024], L2-normalised, rows in h5 coordinate order
<job_dir>/<coords_dir>/uni_embeddings/<slide_id>.json  (n, mpp, window_px, encoder, seconds)

Reads through the same reader as the cheap features (MPP window -> LANCZOS 224) with per-worker OpenSlide
handles (D001 style). Only this script and the Full-UNI baseline ever see all rows; the acquisition
experiments read rows through LazyUNIFeatureStore.acquire.
  CUDA_VISIBLE_DEVICES=0 python scripts/tcga/cache_uni_embeddings.py --shard 0 --num-shards 2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common.config import load_config, resolve  # noqa: E402
from src.tcga.reader import read_patch  # noqa: E402

TF = None  # set per encoder from the registry


class SlidePatches(Dataset):
    def __init__(self, wsi: str, coords: np.ndarray, window: int):
        self.wsi, self.coords, self.window = wsi, coords, window
        self._handles: dict[int, object] = {}

    def __len__(self):
        return len(self.coords)

    def _slide(self):
        import openslide
        pid = os.getpid()
        if pid not in self._handles:
            self._handles[pid] = openslide.OpenSlide(self.wsi)
        return self._handles[pid]

    def __getitem__(self, i):
        x, y = self.coords[i]
        return TF(read_patch(self._slide(), int(x), int(y), self.window)), i


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/tcga_task0.yaml")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--reverse", action="store_true", help="process the shard back to front (a second GPU can meet the first in the middle)")
    ap.add_argument("--encoder", default="uni", help="uni (historical cache, <layer>/uni_embeddings) | hoptimus0 (<layer>/hoptimus0_embeddings)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    df = pd.read_csv(resolve(cfg["tcga"]["slide_table"]))
    if args.limit:
        df = df.groupby("category").head(args.limit)
    df = df.iloc[args.shard::args.num_shards]
    if args.reverse:
        df = df.iloc[::-1]
    global TF
    from src.encoders.registry import build_encoder, encode
    model, TF, sp = build_encoder(args.encoder, "cuda")
    for r in df.itertuples():
        out = Path(r.features_npy) if args.encoder == "uni" else Path(r.features_npy).parent.parent / f"{args.encoder}_embeddings" / Path(r.features_npy).name
        if out.exists():
            continue
        t0 = time.time()
        with h5py.File(r.h5_path, "r") as f:
            coords = f["coords"][:].astype(np.int64)
        dl = DataLoader(SlidePatches(r.wsi_path, coords, int(r.window_px)), batch_size=args.batch_size, num_workers=args.workers, shuffle=False)
        E = np.zeros((len(coords), sp.dim), dtype=np.float32); seen = np.zeros(len(coords), dtype=bool)
        for x, idx in dl:
            E[idx.numpy()] = encode(model, sp, x, "cuda", True); seen[idx.numpy()] = True
        assert seen.all(), "missing rows"
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp.npy"); np.save(tmp, E); tmp.rename(out)
        out.with_suffix(".json").write_text(json.dumps({"slide_id": r.slide_id, "n": int(len(coords)), "mpp": float(r.mpp), "window_px": int(r.window_px),
                                                         "encoder": sp.hub, "encoder_name": args.encoder, "dim": sp.dim, "l2_normalized": True, "row_order": "h5 coords", "seconds": round(time.time() - t0, 1)}))
        print(f"[uni-cache] {r.slide_id}: {E.shape} in {time.time()-t0:.0f}s", flush=True)
    print("[uni-cache] done")


if __name__ == "__main__":
    main()
