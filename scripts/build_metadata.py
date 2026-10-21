"""Build metadata/slides.csv from the existing reorg h5 tree (one representative slide per case).

Columns: slide_id, case_id, patient_id, slide_name, label, class_name, wsi_path, h5_path,
         features_npy, features_csv, n_patches
Run: python scripts/build_metadata.py [--config configs/default.yaml]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.config import load_config, resolve, slide_id_from  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    h5_root = Path(cfg["paths"]["h5_root"])
    feat_root = Path(cfg["paths"]["features_root"])
    wsi_root = Path(cfg["paths"]["wsi_root"])
    class_to_label = {cfg["labels"]["class_0"]: 0, cfg["labels"]["class_1"]: 1}

    rows = []
    for cls, label in class_to_label.items():
        for case_dir in sorted((h5_root / cls).iterdir()):
            if not case_dir.is_dir():
                continue
            h5s = sorted(case_dir.glob("*.h5"))
            if len(h5s) != 1:
                print(f"[metadata] WARNING {case_dir} has {len(h5s)} h5 files; expected 1")
            for h5p in h5s:
                slide_name = h5p.stem
                case_id = case_dir.name
                with h5py.File(h5p, "r") as f:
                    n = int(f["coords"].shape[0])
                npy = feat_root / cls / f"{case_id}_{slide_name}_features.npy"
                csv = feat_root / cls / f"{case_id}_{slide_name}_metadata.csv"
                wsi = wsi_root / cls / case_id / f"{slide_name}.svs"
                rows.append(dict(
                    slide_id=slide_id_from(case_id, slide_name),
                    case_id=case_id,
                    patient_id=case_id,          # one case == one patient in this cohort
                    slide_name=slide_name,
                    label=label,
                    class_name=cls,
                    wsi_path=str(wsi) if wsi.exists() else "",
                    h5_path=str(h5p),
                    features_npy=str(npy) if npy.exists() else "",
                    features_csv=str(csv) if csv.exists() else "",
                    n_patches=n,
                ))
    df = pd.DataFrame(rows).sort_values("slide_id").reset_index(drop=True)
    out = resolve(cfg["paths"]["metadata_csv"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[metadata] slides={len(df)} patients={df.patient_id.nunique()} "
          f"labels={df.label.value_counts().to_dict()} missing_features={(df.features_npy == '').sum()} "
          f"missing_wsi={(df.wsi_path == '').sum()} -> {out}")


if __name__ == "__main__":
    main()
