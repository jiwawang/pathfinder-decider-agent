#!/usr/bin/env bash
# WP007 Phase 1 diagnostics. Prints identity/region/quota/Bedrock/S3 status WITHOUT printing any secret
# (no access keys, no session tokens; account id is masked). Safe to paste into a report.
#   bash scripts/aws/check_environment.sh [region]
REGION="${1:-${AWS_REGION:-${AWS_DEFAULT_REGION:-us-east-1}}}"
PY=${PY:-python}
echo "== region: $REGION"
echo "== aws cli: $(command -v aws >/dev/null && aws --version 2>&1 | head -1 || echo 'not installed (boto3 is used instead)')"
echo "== config files: $(ls ~/.aws 2>/dev/null | tr '\n' ' ' || true)"
echo "== credential env vars set: $(env | grep -E '^(AWS_ACCESS_KEY_ID|AWS_SECRET_ACCESS_KEY|AWS_SESSION_TOKEN|AWS_PROFILE|AWS_REGION|AWS_DEFAULT_REGION|BEDROCK_MODEL_ID|S3_BUCKET)=' | cut -d= -f1 | tr '\n' ' ')"
$PY - "$REGION" <<'PYEOF'
import sys, boto3, botocore
region = sys.argv[1]
def mask(s): return s[:4] + "…" + s[-2:] if s and len(s) > 8 else "…"
try:
    ident = boto3.client("sts", region_name=region).get_caller_identity()
    print(f"== identity: account {mask(ident['Account'])}, principal {ident['Arn'].split(':')[-1].replace(ident['Account'], mask(ident['Account']))}")
except Exception as e:
    print(f"== identity: NOT AUTHENTICATED ({type(e).__name__})"); sys.exit(0)
def q(code, name):
    try:
        v = boto3.client("service-quotas", region_name=region).get_service_quota(ServiceCode="ec2", QuotaCode=code)["Quota"]["Value"]
        print(f"== quota {name}: {v:.0f} vCPUs")
    except Exception as e:
        print(f"== quota {name}: unknown ({type(e).__name__})")
q("L-DB2E81BA", "Running On-Demand G and VT instances")
try:
    ec2 = boto3.client("ec2", region_name=region)
    for t in ("g6.xlarge", "g5.xlarge"):
        off = ec2.describe_instance_type_offerings(LocationType="availability-zone", Filters=[{"Name": "instance-type", "Values": [t]}])["InstanceTypeOfferings"]
        print(f"== {t} offered in AZs: {sorted(o['Location'] for o in off)}")
except Exception as e:
    print(f"== ec2 describe: {type(e).__name__}")
try:
    ms = boto3.client("bedrock", region_name=region).list_foundation_models(byOutputModality="TEXT")["modelSummaries"]
    small = [m["modelId"] for m in ms if any(k in m["modelId"] for k in ("haiku", "nova-micro", "nova-lite", "titan-text"))]
    print(f"== bedrock text models listed: {len(ms)}; small candidates: {small[:8]}")
except Exception as e:
    print(f"== bedrock list: {type(e).__name__}: {str(e)[:100]}")
try:
    bs = boto3.client("s3").list_buckets()["Buckets"]; print(f"== s3 buckets visible: {len(bs)}")
except Exception as e:
    print(f"== s3: {type(e).__name__}")
PYEOF
