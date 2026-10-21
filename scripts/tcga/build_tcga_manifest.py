"""WP005 data adapter: frozen TCGA manifests from GDC metadata (no downloads, no classification).

Eligibility (GDC /files): project in task, data_type "Slide Image", experimental_strategy "Diagnostic Slide",
sample_type "Primary Tumor", access "open", format SVS. One slide per patient = lowest DX number by file-name
sort; patients shuffled with seed 0 and the first N per class kept (same rule as <TCGA_ROOT>/tcga_select_download.py,
so the 60 slides already on disk per project are the first 60 of the same ordering).

  python scripts/tcga/build_tcga_manifest.py --task lung   --n-per-class 100
  python scripts/tcga/build_tcga_manifest.py --task kidney --n-per-class 100
Writes metadata/tcga_<task>_manifest.csv (+ .sha256). Raw GDC responses are kept in metadata/tcga_gdc_raw/.
Re-running with the same arguments reproduces the file byte-for-byte; a frozen manifest must not be overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common.config import PROJECT_ROOT  # noqa: E402

TASKS = {"lung": {"TCGA-LUAD": 0, "TCGA-LUSC": 1}, "kidney": {"TCGA-KIRC": 0, "TCGA-KIRP": 1}}
LOCAL = {"TCGA-LUAD": "<TCGA_ROOT>/LUAD/svs", "TCGA-KIRC": "<TCGA_ROOT>/KIRC/svs", "TCGA-LUSC": "<TCGA_ROOT>/LUSC/svs", "TCGA-KIRP": "<TCGA_ROOT>/KIRP/svs"}
RAW = PROJECT_ROOT / "metadata" / "tcga_gdc_raw"


def rows_for(project: str, label: int, n: int, seed: int) -> list[dict]:
    hits = json.load(open(RAW / f"gdc_{project}.json"))["data"]["hits"]
    by_pat: dict[str, list] = {}
    for h in hits:
        c = h["cases"][0]
        if h.get("data_format") != "SVS" or h.get("access") != "open" or h.get("experimental_strategy") != "Diagnostic Slide":
            continue
        if c["samples"][0]["sample_type"] != "Primary Tumor":
            continue
        by_pat.setdefault(c["submitter_id"], []).append(h)
    chosen = {p: sorted(v, key=lambda h: h["file_name"])[0] for p, v in by_pat.items()}   # lowest DX
    pats = sorted(chosen)
    random.Random(seed).shuffle(pats)
    out = []
    for rank, p in enumerate(pats[:n]):
        h = chosen[p]; c = h["cases"][0]; s = c["samples"][0]; sl = s["portions"][0]["slides"][0]
        local = os.path.join(LOCAL[project], h["file_name"])
        out.append(dict(
            task_label=label, project_id=project, patient_id=c["submitter_id"], case_id=c["case_id"], sample_id=s["sample_id"],
            sample_barcode=s["submitter_id"], slide_barcode=sl["submitter_id"], slide_id=sl["slide_id"],
            gdc_file_id=h["file_id"], file_name=h["file_name"], file_size=h["file_size"], md5=h["md5sum"],
            experimental_strategy=h["experimental_strategy"], sample_type=s["sample_type"],
            slide_type=re.search(r"-(DX\d+)", h["file_name"]).group(1), section_location=sl.get("section_location", ""),
            tissue_source_site_code=c["submitter_id"].split("-")[1], tissue_source_site_name=c["tissue_source_site"]["name"],
            n_dx_slides_for_patient=len(by_pat[p]), selection_rank=rank, selection_seed=seed,
            local_path=local if os.path.exists(local) else "",
        ))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(TASKS), required=True)
    ap.add_argument("--n-per-class", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = PROJECT_ROOT / "metadata" / f"tcga_{args.task}_manifest.csv"
    if out.exists():
        raise SystemExit(f"{out} exists: frozen manifests are never overwritten (use a new file name for a new run)")
    rows = []
    for proj, lab in TASKS[args.task].items():
        rows += rows_for(proj, lab, args.n_per_class, args.seed)
    df = pd.DataFrame(rows).sort_values(["task_label", "patient_id"]).reset_index(drop=True)
    assert df.patient_id.is_unique and df.gdc_file_id.is_unique and df.md5.is_unique
    df.to_csv(out, index=False)
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    (out.with_suffix(".sha256")).write_text(f"{digest}  {out.name}\n")
    print(f"{out.name}: {len(df)} rows, labels {df.task_label.value_counts().to_dict()}, local {int((df.local_path != '').sum())}, "
          f"to download {df[df.local_path == ''].file_size.sum()/1e9:.1f} GB, sha256 {digest[:16]}")


if __name__ == "__main__":
    main()
