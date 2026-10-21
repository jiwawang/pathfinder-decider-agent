"""Task 0 slide table from the TRIDENT settings_2 coords layers (D001 convention) + MPP audit + exclusions.

For each cancer type: <job_dir>/<coords_dir>/patches/<slide>_patches.h5 must exist and the slide must have a
valid MPP (reader.slide_mpp) and >= min_patches tissue patches. Writes, per coords layer, mpp_audit.csv and
exclusions.csv (D001 style, next to the data), and in the repo metadata/tcga_task0_slides.csv (pointers only).

  python scripts/tcga/build_task0_slide_table.py
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import h5py
import openslide
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common.config import PROJECT_ROOT, load_config, resolve  # noqa: E402
from src.tcga.reader import mpp_status, slide_mpp, window_px  # noqa: E402

TCGA_PATIENT = re.compile(r"^(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/tcga_task0.yaml")
    ap.add_argument("--min-patches", type=int, default=128)
    args = ap.parse_args()
    cfg = load_config(args.config)
    t = cfg["tcga"]
    manifest = pd.read_csv(resolve(t["manifest"])) if t.get("manifest") else None
    if manifest is not None:
        manifest = manifest.set_index("file_name")
    rows, excl_all = [], []
    for cat, label in t["label_map"].items():
        job = Path(t["job_dirs"][cat]); layer = job / t["coords_dir"]; svs_dir = Path(t["svs_dirs"][cat])
        audit, excl = [], []
        for svs in sorted(svs_dir.glob("*.svs")):
            sid = svs.stem
            if manifest is not None and svs.name not in manifest.index:
                continue                      # on disk but not a member of the frozen manifest
            h5 = layer / "patches" / f"{sid}_patches.h5"
            rec = {"slide_id": sid, "category": cat}
            try:
                s = openslide.OpenSlide(str(svs))
                mpp = slide_mpp(s); st = mpp_status(mpp)
                rec.update(mpp_x=s.properties.get("openslide.mpp-x"), mpp_y=s.properties.get("openslide.mpp-y"),
                           level0_w=s.level_dimensions[0][0], level0_h=s.level_dimensions[0][1], status=st)
                s.close()
            except Exception as e:  # unreadable file
                rec.update(status=f"skip: unreadable ({type(e).__name__})"); mpp = None
            audit.append(rec)
            if mpp is None:
                excl.append({"slide_id": sid, "category": cat, "reason": rec["status"]}); continue
            if not h5.exists():
                excl.append({"slide_id": sid, "category": cat, "reason": "skip: no TRIDENT coords (see _logs_coords.txt)"}); continue
            with h5py.File(h5, "r") as f:
                n = int(f["coords"].shape[0])
            if n < args.min_patches:
                excl.append({"slide_id": sid, "category": cat, "reason": f"skip: only {n} tissue patches (< {args.min_patches})"}); continue
            if rec["status"].startswith("warn"):
                print(f"[task0] WARNING {sid}: {rec['status']}")
            rows.append(dict(slide_id=sid, patient_id=TCGA_PATIENT.match(sid).group(1), label=int(label), category=cat,
                             wsi_path=str(svs), h5_path=str(h5), cheap_path=str(layer / "cheap_features" / f"{sid}.npz"),
                             features_npy=str(layer / "uni_embeddings" / f"{sid}.npy"), mpp=round(mpp, 5), window_px=window_px(mpp),
                             n_patches=n, tss=sid.split("-")[1],
                             **({"gdc_file_id": manifest.loc[svs.name, "gdc_file_id"], "md5": manifest.loc[svs.name, "md5"],
                                 "gdc_slide_id": manifest.loc[svs.name, "slide_id"], "selection_rank": int(manifest.loc[svs.name, "selection_rank"])}
                                if manifest is not None else {})))
        layer.mkdir(parents=True, exist_ok=True)   # may precede TRIDENT for a category still running
        pd.DataFrame(audit).to_csv(layer / "mpp_audit.csv", index=False)
        pd.DataFrame(excl, columns=["slide_id", "category", "reason"]).to_csv(layer / "exclusions.csv", index=False)
        excl_all += excl
        print(f"[task0] {cat}: {sum(r['category'] == cat for r in rows)} kept, {len(excl)} excluded -> {layer/'exclusions.csv'}")
    if manifest is not None:
        missing = [f for f in manifest.index if not any((Path(d) / f).exists() for d in t["svs_dirs"].values())]
        print(f"[task] manifest members not on disk: {len(missing)}")
    df = pd.DataFrame(rows).sort_values("slide_id").reset_index(drop=True)
    assert df.patient_id.is_unique, "one slide per patient violated"
    out = resolve(t["slide_table"])
    df.to_csv(out, index=False)
    print(f"[task0] slides={len(df)} patients={df.patient_id.nunique()} labels={df.label.value_counts().sort_index().to_dict()} "
          f"windows={df.window_px.value_counts().to_dict()} -> {out}")
    for e in excl_all:
        print("   excluded:", e["category"], e["slide_id"], "|", e["reason"])


if __name__ == "__main__":
    main()
