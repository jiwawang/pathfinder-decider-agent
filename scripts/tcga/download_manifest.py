"""Download the slides of a frozen manifest that are not on disk (D001 downloader rule: resume, size + md5 check).

  python scripts/tcga/download_manifest.py --manifest metadata/tcga_lung_manifest.csv --workers 4 [--first 3] [--dry-run]
Files go to <TCGA_ROOT>/<TYPE>/svs/<file_name> (TYPE = project suffix); a selection_n<N>.csv (patient, file_name, md5)
is written next to svs/ in the D001 format without touching the original selection.csv. Standard library only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

API = "https://api.gdc.cancer.gov/data"
CHUNK = 8 << 20


def md5_of(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(CHUNK), b""):
            h.update(b)
    return h.hexdigest()


def download_one(row: dict, retries: int = 10) -> str:
    out_dir = Path(f"<TCGA_ROOT>/{row['project_id'].split('-')[1]}/svs"); out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / row["file_name"]; size, md5 = int(row["file_size"]), row["md5"]
    if path.exists() and path.stat().st_size == size and md5_of(str(path)) == md5:
        return f"skip {row['file_name'][:28]}"
    part = str(path) + ".part"
    for attempt in range(1, retries + 1):
        try:
            have = os.path.getsize(part) if os.path.exists(part) else 0
            if have >= size:
                os.remove(part); have = 0
            req = urllib.request.Request(f"{API}/{row['gdc_file_id']}", headers={"Range": f"bytes={have}-"} if have else {})
            with urllib.request.urlopen(req, timeout=120) as r, open(part, "ab" if have else "wb") as f:
                for b in iter(lambda: r.read(CHUNK), b""):
                    f.write(b)
            if os.path.getsize(part) != size:
                raise IOError(f"size {os.path.getsize(part)} != {size}")
            if md5_of(part) != md5:
                os.remove(part); raise IOError("md5 mismatch")
            os.replace(part, path)
            return f"ok   {row['file_name'][:28]} {size/1e9:.2f} GB (attempt {attempt})"
        except Exception as e:  # resume on next attempt
            time.sleep(min(60, 5 * attempt)); err = e
    return f"FAIL {row['file_name'][:28]}: {err}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--first", type=int, default=0, help="only the first N missing rows (pipeline validation)")
    ap.add_argument("--project", default=None, help="restrict to one project id")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    rows = list(csv.DictReader(open(args.manifest)))
    if args.project:
        rows = [r for r in rows if r["project_id"] == args.project]
    for proj in sorted({r["project_id"] for r in rows}):   # D001-style selection record (never overwrite the original)
        d = Path(f"<TCGA_ROOT>/{proj.split('-')[1]}"); d.mkdir(parents=True, exist_ok=True)
        sel = d / f"selection_n{sum(r['project_id'] == proj for r in rows)}_wp005manifest.csv"
        if not sel.exists():
            with open(sel, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["patient", "file_name", "md5", "gdc_file_id"]); w.writeheader()
                for r in rows:
                    if r["project_id"] == proj:
                        w.writerow({"patient": r["patient_id"], "file_name": r["file_name"], "md5": r["md5"], "gdc_file_id": r["gdc_file_id"]})
    missing = [r for r in rows if not os.path.exists(f"<TCGA_ROOT>/{r['project_id'].split('-')[1]}/svs/{r['file_name']}")]
    missing.sort(key=lambda r: (r["project_id"], int(r["selection_rank"])))
    if args.first:
        missing = missing[:args.first]
    print(f"[download] {len(missing)} files, {sum(int(r['file_size']) for r in missing)/1e9:.1f} GB, {args.workers} workers", flush=True)
    if args.dry_run:
        return
    t0 = time.time(); done = 0
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(download_one, r): r for r in missing}
        for fu in as_completed(futs):
            done += 1; print(f"[download] {done}/{len(missing)} {fu.result()} ({time.time()-t0:.0f}s)", flush=True)
    print("[download] done")


if __name__ == "__main__":
    main()
