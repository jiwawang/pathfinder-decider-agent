#!/usr/bin/env bash
# WP007 bootstrap for ONE GPU EC2 instance (Deep Learning Base AMI, Ubuntu, g6.xlarge / g5.xlarge), run as the login user.
#   export AWS_REGION=... BEDROCK_MODEL_ID=... S3_BUCKET=... HF_TOKEN=<your gated-UNI token, never committed>
#   bash scripts/aws/ec2_setup.sh
# Mirrors the local `pathfinder` env exactly (requirements-aws.txt = pip freeze), pulls the demo artifacts from the
# private bucket with the instance role, downloads the gated UNI weights with the owner's HF token, runs the smoke tests.
set -euo pipefail
cd "$(dirname "$0")/../.."
if ! command -v conda >/dev/null; then source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true; fi
conda create -y -n pathfinder python=3.11 >/dev/null
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate pathfinder
pip install -q -r requirements-aws.txt
sudo apt-get install -y -q openslide-tools >/dev/null 2>&1 || true
export PATHFINDER_CACHE=/opt/pathfinder-demo; mkdir -p "$PATHFINDER_CACHE"
python - <<'PY'
import boto3, os
b, p = os.environ["S3_BUCKET"], os.environ.get("S3_PREFIX", "pathfinder-demo"); s3 = boto3.client("s3"); dest = os.environ["PATHFINDER_CACHE"]; n = 0
for page in s3.get_paginator("list_objects_v2").paginate(Bucket=b, Prefix=p + "/"):
    for o in page.get("Contents", []):
        t = os.path.join(dest, o["Key"][len(p) + 1:]); os.makedirs(os.path.dirname(t), exist_ok=True); s3.download_file(b, o["Key"], t); n += 1
print("demo objects fetched:", n)
PY
python -c "from huggingface_hub import login; import os; login(token=os.environ['HF_TOKEN'], add_to_git_credential=False)"
python scripts/aws/smoke_test.py --test all
echo "[setup] done; run: python scripts/aws/run_agent.py --slide-id <id> --budget 16 --mode live --source s3"
