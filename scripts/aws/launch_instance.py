"""Launch ONE GPU instance for WP007 with an instance role (no long-lived keys on the box). Costs money: run only after
the owner approved type/region/budget (WP007 section 7, 20).
  python scripts/aws/launch_instance.py --type g6.xlarge --key-name <ssh key> --role-name PathFinderDemoRole [--ami <id>]
Requires a pre-existing IAM instance profile with s3 read/write on the demo prefix and bedrock:InvokeModel; prints the
instance id, public DNS and the hourly on-demand price from the Pricing API when available.
"""
from __future__ import annotations

import argparse
import json
import os

import boto3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", default="g6.xlarge"); ap.add_argument("--key-name", required=True); ap.add_argument("--role-name", required=True)
    ap.add_argument("--ami", default=None, help="Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04); resolved via SSM when omitted")
    ap.add_argument("--volume-gb", type=int, default=150); ap.add_argument("--sg", default=None, help="security group id (SSH from your IP only)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(); region = os.environ.get("AWS_REGION", "us-east-1"); ec2 = boto3.client("ec2", region_name=region)
    ami = args.ami or boto3.client("ssm", region_name=region).get_parameter(Name="/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id")["Parameter"]["Value"]
    try:
        pr = boto3.client("pricing", region_name="us-east-1").get_products(ServiceCode="AmazonEC2", Filters=[
            {"Type": "TERM_MATCH", "Field": "instanceType", "Value": args.type}, {"Type": "TERM_MATCH", "Field": "regionCode", "Value": region},
            {"Type": "TERM_MATCH", "Field": "operatingSystem", "Value": "Linux"}, {"Type": "TERM_MATCH", "Field": "tenancy", "Value": "Shared"},
            {"Type": "TERM_MATCH", "Field": "preInstalledSw", "Value": "NA"}, {"Type": "TERM_MATCH", "Field": "capacitystatus", "Value": "Used"}], MaxResults=1)
        od = json.loads(pr["PriceList"][0])["terms"]["OnDemand"]; price = next(iter(next(iter(od.values()))["priceDimensions"].values()))["pricePerUnit"]["USD"]
    except Exception:
        price = "unknown"
    print(f"[launch] {args.type} in {region}, AMI {ami}, on-demand USD/h: {price}, EBS {args.volume_gb} GB gp3")
    if args.dry_run:
        return
    r = ec2.run_instances(ImageId=ami, InstanceType=args.type, MinCount=1, MaxCount=1, KeyName=args.key_name, IamInstanceProfile={"Name": args.role_name},
                          BlockDeviceMappings=[{"DeviceName": "/dev/sda1", "Ebs": {"VolumeSize": args.volume_gb, "VolumeType": "gp3", "DeleteOnTermination": True}}],
                          **({"SecurityGroupIds": [args.sg]} if args.sg else {}), TagSpecifications=[{"ResourceType": "instance", "Tags": [{"Key": "Name", "Value": "pathfinder-wp007"}, {"Key": "project", "Value": "pathfinder"}]}])
    iid = r["Instances"][0]["InstanceId"]; print("[launch] instance", iid); ec2.get_waiter("instance_running").wait(InstanceIds=[iid])
    d = ec2.describe_instances(InstanceIds=[iid])["Reservations"][0]["Instances"][0]; print("[launch] public dns:", d.get("PublicDnsName"), "| stop with scripts/aws/stop_instance.sh", iid)


if __name__ == "__main__":
    main()
