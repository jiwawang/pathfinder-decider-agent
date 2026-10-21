"""Materialise artifacts/embeddings/{slide_id}.npz for every slide in metadata/slides.csv.

Fast path (default): convert the existing UNI features (features_root/<cls>/<case>_<slide>_features.npy
+ _metadata.csv, produced by <PRIVATE_COHORT_PIPELINE>/extract_features_from_reorg_h5.py) into the standard
format. Coordinates are checked against the h5 coords to guarantee alignment.

Slow path (--reextract): for slides without existing features, run UNI on the WSI patches. This needs
the `UNI` conda env (timm + openslide + cached MahmoodLab/UNI weights) and must be launched with that
interpreter. It reuses the exact transform / L2-normalisation of the original pipeline.

Output npz keys: embeddings [N,D] float32, coords [N,2] int32, patch_ids [N] int32, slide_id, D
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.config import load_config, resolve  # noqa: E402


def convert_existing(row: pd.Series, out: Path, l2: bool) -> tuple[int, int]:
    feats = np.load(row.features_npy, mmap_mode="r")
    meta = pd.read_csv(row.features_csv)
    with h5py.File(row.h5_path, "r") as f:
        coords = f["coords"][:].astype(np.int32)
    mc = meta[["x", "y"]].values.astype(np.int32)
    if not (len(coords) == len(meta) == feats.shape[0] and np.array_equal(coords, mc)):
        raise RuntimeError(f"coord/feature misalignment for {row.slide_id}: "
                           f"h5={len(coords)} csv={len(meta)} npy={feats.shape[0]}")
    emb = np.asarray(feats, dtype=np.float32)
    if l2:
        emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8)
    np.savez(out, embeddings=emb, coords=coords,
             patch_ids=np.arange(len(coords), dtype=np.int32),
             slide_id=row.slide_id, D=np.int32(emb.shape[1]))
    return emb.shape


def reextract_with_uni(row: pd.Series, out: Path, l2: bool, batch: int = 64) -> tuple[int, int]:
    """Run UNI on the slide's patches. Requires the UNI env; imported lazily."""
    import torch  # noqa
    from PIL import Image
    import openslide
    import timm
    from torchvision import transforms

    with h5py.File(row.h5_path, "r") as f:
        coords = f["coords"][:].astype(np.int32)
        ps = int(f.attrs.get("patch_size", 224))
    model = timm.create_model("hf-hub:MahmoodLab/UNI", pretrained=True, init_values=1e-5,
                              dynamic_img_size=True).cuda().eval()
    tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),
                             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    slide = openslide.OpenSlide(row.wsi_path)
    feats = []
    with torch.inference_mode():
        for i in range(0, len(coords), batch):
            imgs = [tf(slide.read_region((int(x), int(y)), 0, (ps, ps)).convert("RGB"))
                    for x, y in coords[i:i + batch]]
            e = model(torch.stack(imgs).cuda())
            if l2:
                e = torch.nn.functional.normalize(e, dim=1)
            feats.append(e.float().cpu().numpy())
    slide.close()
    emb = np.concatenate(feats, 0).astype(np.float32)
    np.savez(out, embeddings=emb, coords=coords, patch_ids=np.arange(len(coords), dtype=np.int32),
             slide_id=row.slide_id, D=np.int32(emb.shape[1]))
    return emb.shape


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--reextract", action="store_true", help="run UNI for slides lacking features")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    df = pd.read_csv(resolve(cfg["paths"]["metadata_csv"]), dtype={"case_id": str, "patient_id": str})
    out_dir = resolve(cfg["paths"]["embeddings_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    l2 = bool(cfg["embeddings"]["l2_normalize"])
    done = skipped = 0
    for _, row in df.iterrows():
        out = out_dir / f"{row.slide_id}.npz"
        if out.exists() and not args.overwrite:
            skipped += 1
            continue
        if isinstance(row.features_npy, str) and row.features_npy:
            shape = convert_existing(row, out, l2)
        elif args.reextract and row.wsi_path:
            shape = reextract_with_uni(row, out, l2)
        else:
            print(f"[embeddings] SKIP {row.slide_id}: no existing features (use --reextract in UNI env)")
            continue
        done += 1
        print(f"[embeddings] {row.slide_id} -> {shape}", flush=True)
    print(f"[embeddings] written={done} existing={skipped} dir={out_dir}")


if __name__ == "__main__":
    main()
