#!/usr/bin/env bash
# Upload everything the online agent needs (no WSIs, no full-slide embeddings) to S3.
#   bash scripts/aws/sync_artifacts_to_s3.sh s3://<bucket>/pathfinder
# Needs: aws cli configured (aws configure) with s3:PutObject on the bucket.
set -euo pipefail
DEST="${1:?usage: $0 s3://bucket/prefix}"
cd "$(dirname "$0")/../.."
aws s3 sync metadata            "$DEST/metadata"
aws s3 sync artifacts/candidates "$DEST/artifacts/candidates"
aws s3 sync artifacts/random_pool "$DEST/artifacts/random_pool"
aws s3 sync artifacts/thumbnails "$DEST/artifacts/thumbnails"
# results of the offline CV (traces, predictions) so the demo can replay them without a GPU
aws s3 sync results "$DEST/results" --exclude "*/traces/*"
echo "synced -> $DEST"
