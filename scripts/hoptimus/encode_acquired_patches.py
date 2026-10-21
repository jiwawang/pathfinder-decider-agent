"""WP011: encode the FROZEN Decider-Fixed-16 acquisitions of every Task slide with H-optimus-0 (or any registry
encoder) straight from the WSIs, so a classifier head can be trained on that encoder's embeddings without
touching the UNI cache. Writes, in the D001 layout next to the coords layer:
    <layer>/<encoder>_acquired_<method>/<slide_id>.npy   [n_acquired, dim]   (+ .json: patch indices, encoder, git)
This is NOT a benchmark re-run: the acquisitions (which patches) come from the recorded traces; only their
embeddings are recomputed with the new encoder.

  CUDA_VISIBLE_DEVICES=0 python scripts/hoptimus/encode_acquired_patches.py --config configs/tcga_task1.yaml --encoder hoptimus0
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import openslide
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common.config import load_config, resolve  # noqa: E402
from src.encoders.registry import build_encoder, encode  # noqa: E402
from src.tcga.reader import read_patch  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/tcga_task1.yaml")
    ap.add_argument("--run-dir", default=None, help="defaults to the task's results_dir")
    ap.add_argument("--method", default="decider_fixed_16")
    ap.add_argument("--encoder", default="hoptimus0")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    tcfg = load_config(args.config)["tcga"]; run = resolve(args.run_dir or tcfg["results_dir"])
    snap = pd.read_csv(run / "manifest_snapshot.csv")
    git = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    model, tf, sp = build_encoder(args.encoder, args.device)
    t_all = time.time(); n_patches = 0
    for r in snap.itertuples():
        layer = Path(r.cheap_path).parent.parent; out = layer / f"{args.encoder}_acquired_{args.method}" / f"{r.slide_id}.npy"
        if out.exists():
            continue
        tr = json.loads((run / "traces" / args.method / f"{r.slide_id}.json").read_text()); idx = tr["acquired_patch_index"]
        coords = np.load(r.cheap_path)["coords"]; s = openslide.OpenSlide(r.wsi_path)
        x = torch.stack([tf(read_patch(s, int(coords[i][0]), int(coords[i][1]), int(r.window_px))) for i in idx]); s.close()
        E = encode(model, sp, x, args.device, True); n_patches += len(idx)
        out.parent.mkdir(parents=True, exist_ok=True); np.save(out, E)
        out.with_suffix(".json").write_text(json.dumps({"slide_id": r.slide_id, "method": args.method, "acquired_patch_index": idx, "encoder": sp.hub, "encoder_name": args.encoder,
                                                        "dim": sp.dim, "license": sp.license, "source_run": str(run), "git_commit": git}))
    print(f"[encode] {args.encoder}: {n_patches} acquired patches over {len(snap)} slides in {time.time()-t_all:.0f}s -> <layer>/{args.encoder}_acquired_{args.method}/")


if __name__ == "__main__":
    main()
