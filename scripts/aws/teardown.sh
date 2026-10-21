#!/usr/bin/env bash
# Tear down WP007 temporary resources. Explicit, opt-in, never implicit:
#   bash scripts/aws/teardown.sh --terminate-instance <id> [--delete-demo-prefix] [--delete-stack <cfn-stack>]
# Keeps by default: the S3 bucket and its runs/ prefix (evidence), the instance (only --terminate-instance removes it).
# Never empties a whole bucket; --delete-demo-prefix removes only <S3_PREFIX>/slides and /classifiers.
set -euo pipefail
REGION="${AWS_REGION:-us-east-1}"; PY=python
while [[ $# -gt 0 ]]; do case "$1" in
  --terminate-instance) IID="$2"; shift 2;;
  --delete-demo-prefix) DELPREFIX=1; shift;;
  --delete-stack) STACK="$2"; shift 2;;
  *) echo "unknown arg $1"; exit 1;; esac; done
if [[ -n "${IID:-}" ]]; then $PY -c "import boto3; print(boto3.client('ec2', region_name='$REGION').terminate_instances(InstanceIds=['$IID'])['TerminatingInstances'][0]['CurrentState'])"; fi
if [[ -n "${DELPREFIX:-}" ]]; then $PY - <<'EOF2'
import boto3, os
b, p = os.environ["S3_BUCKET"], os.environ.get("S3_PREFIX", "pathfinder-demo"); s3 = boto3.resource("s3").Bucket(b); n = 0
for sub in ("slides/", "classifiers/"):
    for o in s3.objects.filter(Prefix=f"{p}/{sub}"): o.delete(); n += 1
print("deleted demo objects:", n, "(runs/ kept)")
EOF2
fi
if [[ -n "${STACK:-}" ]]; then $PY -c "import boto3; boto3.client('cloudformation', region_name='$REGION').delete_stack(StackName='$STACK'); print('delete requested for stack $STACK')"; fi
echo "[teardown] done (region $REGION)"
