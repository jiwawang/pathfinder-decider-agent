"""Build the compact S3 demo layout for 1-3 TCGA-Lung slides (WP007 sections 10-11, 14), locally and without any
UNI embedding of the candidates:

  <out>/pathfinder-demo/slides/<slide_id>/
      thumbnail.jpg        slide thumbnail (long side 1024)
      cheap_features.npz   the slide's cheap features (UNI-free)
      candidates.npz       the frozen 128-candidate pool arrays (cheap stats only)
      candidates.json      index -> patch_index, x, y, cluster id/fraction (human-readable)
      patches/NNNN.png     the 128 candidates' RAW RGB 224-px patches (MPP-matched window, LANCZOS), no embeddings
      slide.json           provenance (gdc file id, md5, mpp, window, n_patches, classifier artifact, label kept out)
  <out>/pathfinder-demo/classifiers/<task>_seed<s>_fold<f>.joblib + .json
      the frozen classifier of the fold in which the demo slide is a VALIDATION slide, fit with the audited protocol
      on the training slides' Decider-Fixed-16 embeddings (re-read through the lazy store).
  <out>/pathfinder-demo/demo_manifest.json

  python scripts/aws/prepare_demo_artifacts.py --out <ARTIFACT_ROOT>/aws_demo
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import openslide
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.acquisition.lazy_uni_store import LazyUNIFeatureStore  # noqa: E402
from src.common.config import load_config, resolve  # noqa: E402
from src.evaluation.splits import make_folds  # noqa: E402
from src.models.slide_classifier import build_classifier  # noqa: E402
from src.tcga.reader import read_patch  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-config", default="configs/tcga_task1.yaml")
    ap.add_argument("--base-config", default="configs/default.yaml")
    ap.add_argument("--run-dir", default="results/tcga_generalization/task1_luad_lusc")
    ap.add_argument("--out", default="<ARTIFACT_ROOT>/aws_demo")
    ap.add_argument("--n-per-class", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--method", default="decider_fixed_16")
    ap.add_argument("--encoder", default="hoptimus0", help="classifier head encoder: hoptimus0 (hackathon) | uni (historical)")
    ap.add_argument("--no-patches", action="store_true", help="skip re-writing thumbnails/patches (artifacts already exist)")
    args = ap.parse_args()
    tcfg = load_config(args.task_config)["tcga"]; cfg = load_config(args.base_config)
    run = resolve(args.run_dir); out = Path(args.out) / "pathfinder-demo"
    snap = pd.read_csv(run / "manifest_snapshot.csv").sort_values("slide_id").reset_index(drop=True)
    meta_all = snap  # the slides used in the CV, in the runner's order
    demo = snap.sort_values("selection_rank").groupby("category").head(args.n_per_class)
    git = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    folds = make_folds(meta_all, int(cfg["evaluation"]["folds"]), args.seed)
    store = LazyUNIFeatureStore(dict(zip(meta_all.slide_id, meta_all.features_npy)))
    (out / "classifiers").mkdir(parents=True, exist_ok=True)
    from src.encoders.registry import spec as enc_spec
    sp = enc_spec(args.encoder)
    manifest = {"created": datetime.now().isoformat(), "git_commit": git, "task": tcfg["task"], "label_map": tcfg["label_map"], "encoder": args.encoder, "encoder_hub": sp.hub,
                "classifier_protocol": cfg["classifier"], "acquisition_method_for_training_embeddings": args.method, "slides": []}
    for r in demo.itertuples():
        sd = out / "slides" / r.slide_id; (sd / "patches").mkdir(parents=True, exist_ok=True)
        pool_path = Path(r.cheap_path).parent.parent / f"candidate_pool_K{int(cfg['acquisition']['candidate_pool_size'])}" / f"{r.slide_id}.npz"
        pool = np.load(pool_path, allow_pickle=True)
        cands = [{"index": int(i), "patch_index": int(pool["patch_index"][i]), "x": int(pool["coord_x"][i]), "y": int(pool["coord_y"][i]),
                  "cluster_id": int(pool["cluster_id"][i]), "cluster_fraction": float(pool["cluster_fraction"][i]), "png": f"patches/{i:04d}.png"} for i in range(int(pool["pool_size"]))]
        if not args.no_patches:
            s = openslide.OpenSlide(r.wsi_path)
            w0, h0 = s.level_dimensions[0]; sc = 1024 / max(w0, h0)
            s.get_thumbnail((int(w0 * sc), int(h0 * sc))).convert("RGB").save(sd / "thumbnail.jpg", quality=88)
            z = np.load(r.cheap_path); np.savez(sd / "cheap_features.npz", **{k: z[k] for k in z.files})
            np.savez(sd / "candidates.npz", **{k: pool[k] for k in pool.files})
            for c in cands:   # raw pixels only; no encoder anywhere here
                Image.fromarray(read_patch(s, c["x"], c["y"], int(r.window_px))).save(sd / c["png"])
            s.close()
            json.dump(cands, open(sd / "candidates.json", "w"), indent=1)
        # frozen classifier for the fold in which this slide is a validation slide
        idx = int(np.flatnonzero(meta_all.slide_id == r.slide_id)[0])
        f = next(k for k, (tr, te) in enumerate(folds) if idx in te); tr, te = folds[f]
        X = []
        for s_id, cp in zip(meta_all.slide_id.iloc[tr], meta_all.cheap_path.iloc[tr]):
            if args.encoder == "uni":
                trc = json.loads((run / "traces" / args.method / f"{s_id}.json").read_text())
                X.append(np.stack([store.acquire(s_id, i) for i in trc["acquired_patch_index"]]).mean(0))
            else:   # embeddings of the same frozen acquisitions, recomputed with the chosen encoder (scripts/hoptimus/encode_acquired_patches.py)
                X.append(np.load(Path(cp).parent.parent / f"{args.encoder}_acquired_{args.method}" / f"{s_id}.npy").mean(0))
        X = np.stack(X); y = meta_all.label.values[tr].astype(int)
        clf = build_classifier(cfg["classifier"]).fit(X, y)
        cname = f"{tcfg['task']}_{args.encoder}_seed{args.seed}_fold{f}"
        joblib.dump(clf, out / "classifiers" / f"{cname}.joblib")
        prov = {"artifact": cname, "dataset": tcfg["task"], "seed": args.seed, "fold": f, "n_train": int(len(tr)), "training_patient_ids": meta_all.patient_id.iloc[tr].tolist(),
                "validation_slides_of_this_fold": meta_all.slide_id.iloc[te].tolist(), "label_map": tcfg["label_map"], "feature_dim": int(X.shape[1]),
                "encoder": f"{sp.hub} (L2-normalised, 224 px @ 0.5 um/px)", "encoder_name": args.encoder, "encoder_license": sp.license,
                "training_embeddings": f"mean of the {args.method} acquisitions from {args.run_dir}, embedded with {args.encoder}",
                "classifier": cfg["classifier"], "git_commit": git}
        json.dump(prov, open(out / "classifiers" / f"{cname}.json", "w"), indent=1)
        sj = json.load(open(sd / "slide.json")) if (sd / "slide.json").exists() else {}
        sj.update({"slide_id": r.slide_id, "category": r.category, "gdc_file_id": r.gdc_file_id, "md5": r.md5, "mpp": float(r.mpp), "window_px": int(r.window_px),
                   "n_tissue_patches": int(r.n_patches), "n_candidates": int(pool["pool_size"]), "git_commit": git,
                   "note": "label is NOT stored here; see demo_manifest.json (evaluation only)"})
        sj.setdefault("classifier_artifacts", {})[args.encoder] = cname
        if args.encoder == "hoptimus0" or "classifier_artifact" not in sj:
            sj["classifier_artifact"] = cname      # default artifact = hackathon encoder
        json.dump(sj, open(sd / "slide.json", "w"), indent=1)
        manifest["slides"].append({"slide_id": r.slide_id, "category": r.category, "label": int(r.label), "classifier_artifact": cname, "encoder": args.encoder, "validation_fold": f})
        print(f"[demo] {r.slide_id}: {len(cands)} raw patches, classifier {cname} (train {len(tr)} slides)", flush=True)
    json.dump(manifest, open(out / f"demo_manifest_{args.encoder}.json", "w"), indent=1)
    json.dump(manifest, open(out / "demo_manifest.json", "w"), indent=1)
    print(f"[demo] -> {out}")


if __name__ == "__main__":
    main()
