"""Write a D001-style README.md into each Patch_Trident_settings_2_hest job_dir (what was run, with the exact commands, counts).
  python scripts/tcga/write_layer_readme.py
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common.config import load_config, resolve  # noqa: E402


def main() -> None:
    t = load_config("configs/tcga_task0.yaml")["tcga"]
    table = pd.read_csv(resolve(t["slide_table"])) if resolve(t["slide_table"]).exists() else None
    for cat, job in t["job_dirs"].items():
        job = Path(job); layer = job / t["coords_dir"]
        n_h5 = len(list((layer / "patches").glob("*.h5"))) if (layer / "patches").exists() else 0
        n_cheap = len(list((layer / "cheap_features").glob("*.npz"))) if (layer / "cheap_features").exists() else 0
        n_uni = len(list((layer / "uni_embeddings").glob("*.npy"))) if (layer / "uni_embeddings").exists() else 0
        excl = pd.read_csv(layer / "exclusions.csv") if (layer / "exclusions.csv").exists() else pd.DataFrame(columns=["slide_id", "reason"])
        kept = int((table.category == cat).sum()) if table is not None else 0
        txt = f"""# {cat} — PathFinder patch layer (settings_2_hest), written {date.today()}

New TRIDENT job_dir for the PathFinder WP005 Task 0 experiment (settings_1_hest = the CCT 448 px layer, untouched).
Commands (repo ~/Project/Hackthon_AWS, scripts/tcga/run_trident_settings2.sh; TRIDENT 0.3.2, conda env "trident"):

    E=~/miniconda3/envs/trident; R="$E/bin/python $E/src/TRIDENT/run_batch_of_slides.py"
    W=<TCGA_ROOT>/{cat}/svs; J={job}
    $R --task seg    --wsi_dir $W --job_dir $J --gpus <id> --segmenter hest --skip_errors
    $R --task coords --wsi_dir $W --job_dir $J --mag 20 --patch_size 224 --overlap 0 \\
       --min_tissue_proportion 0.5 --coords_dir {t['coords_dir']} --skip_errors

seg: hest segmenter, no pen-mark / artifact / hole removal. coords: 224 px at nominal 20x, non-overlapping grid,
tissue proportion >= 0.5 (threshold in the dir name on purpose). Slides without MPP fail in seg and have no h5.

{t['coords_dir']}/
    patches/<slide>_patches.h5     TRIDENT coords (level-0 x,y; attrs patch_size_level0 etc.)           {n_h5} slides
    mpp_audit.csv                  per slide: openslide mpp-x/-y, level-0 size, status (ok / skip / warn)
    exclusions.csv                 slides left out and why                                           {len(excl)} slides
    cheap_features/<slide>.npz     PathFinder 19-D cheap features, rows in h5 order                   {n_cheap} slides
    uni_embeddings/<slide>.npy     UNI (hf-hub:MahmoodLab/UNI) L2-normalised, rows in h5 order        {n_uni} slides
    candidate_pool_K128/<slide>.npz  KMeans(128) on within-slide standardised cheap features (written by run_task0_cv.py)

Reading rule (src/tcga/reader.py, same as D001 / wsi_patch_dataset.slide_mpp): MPP only from openslide.mpp-x/-y,
skip if missing or x/y differ > 1 %; level-0 window = round(224 * 0.5 / mpp) px, PIL LANCZOS -> 224 px (112 um per
patch); magnification fields are never used. Cheap features and UNI read pixels through this one code path, in the
conda env "pathfinder" (scripts/tcga/extract_cheap_features_tcga.py, scripts/tcga/cache_uni_embeddings.py).

Slides used by Task 0 (metadata/tcga_task0_slides.csv in the repo): {kept}. Excluded:
""" + "".join(f"    {r.slide_id}: {r.reason}\n" for r in excl.itertuples())
        (job / "README.md").write_text(txt)
        print(f"[readme] {cat}: h5 {n_h5}, cheap {n_cheap}, uni {n_uni}, kept {kept}, excluded {len(excl)}")


if __name__ == "__main__":
    main()
