#!/usr/bin/env bash
# Stop (not terminate) the PathFinder GPU instance so no hourly compute is billed; EBS and S3 are kept.
#   bash scripts/aws/stop_instance.sh <instance-id> [region]
set -euo pipefail
ID="${1:?instance id}"; REGION="${2:-${AWS_REGION:-us-east-1}}"
python - "$ID" "$REGION" <<'EOF2'
import sys, boto3
iid, region = sys.argv[1:3]; ec2 = boto3.client("ec2", region_name=region)
r = ec2.stop_instances(InstanceIds=[iid]); print({i["InstanceId"]: i["CurrentState"]["Name"] for i in r["StoppingInstances"]})
EOF2
