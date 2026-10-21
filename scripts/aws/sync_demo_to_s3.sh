#!/usr/bin/env bash
# Upload the compact demo artifacts (raw patches, cheap metadata, frozen classifiers; NO weights, NO WSIs, NO embeddings)
# to the private bucket. ~26 MB.   bash scripts/aws/sync_demo_to_s3.sh
set -euo pipefail
: "${S3_BUCKET:?set S3_BUCKET}"; P="${S3_PREFIX:-pathfinder-demo}"; SRC=<ARTIFACT_ROOT>/aws_demo/pathfinder-demo
python - "$SRC" "$S3_BUCKET" "$P" <<'PY'
import sys, os, boto3
src, b, p = sys.argv[1:4]; s3 = boto3.client("s3"); n = 0
for root, _, files in os.walk(src):
    for f in files:
        if f.endswith((".joblib", ".json", ".npz", ".png", ".jpg")):
            full = os.path.join(root, f); key = p + "/" + os.path.relpath(full, src); s3.upload_file(full, b, key); n += 1
print(f"uploaded {n} objects to s3://{b}/{p}/ (bucket must stay private)")
PY
