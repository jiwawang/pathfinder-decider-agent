# WP007-1 — Plan B (CPU EC2) feasibility, account status and blockers (2026-10-21)

Branch `integration/aws-bedrock`; no paid resource created; no core code touched. Measurements below are from
this workstation (`pathfinder` env) with thread limits chosen to approximate an 8-vCPU instance.

## 1. AWS account status (read-only probes, root sign-in via `aws login`)
| Check | Result |
|---|---|
| Authentication | works (`aws login` browser flow; boto3 needed `botocore[crt]` → `awscrt` installed in `pathfinder`, pinned in `requirements-aws.txt`) |
| Principal | **account root** (0 IAM users, no MFA on root, 0 access keys). Root should not be the runtime identity; an IAM user/role with least privilege is needed before any deployment. |
| EC2 (describe regions / instances / offerings) | **OptInRequired — "You are not subscribed to this service"** |
| S3 (list buckets) | **NotSignedUp — "Your account is not signed up for the S3 service"** |
| Cost Explorer | SubscriptionRequiredException |
| Service Quotas | readable: On-Demand Standard (A,C,D,H,I,M,R,T,Z) = **8 vCPUs**; On-Demand G/VT = **0 vCPUs** (appeal pending) |
| Bedrock control plane | readable: 95 text models listed, 49 on-demand; `amazon.nova-micro-v1:0` ACTIVE |
| Bedrock runtime (`Converse` and `InvokeModel`, nova-micro and claude-haiku-4-5) | **ValidationException: Operation not allowed** |

Interpretation: the account is not fully activated (new-account verification / payment validation pending) —
the same condition that blocks EC2 and S3 also blocks Bedrock inference. **This blocks Plan A and Plan B
equally; it is not a GPU-quota problem.** Required owner action: complete account activation (AWS usually sends a
"your account is now active" email; the Support case about the GPU quota may be the same workflow), then
enable Bedrock model access for one small model in the console, then create an IAM user/role (no root) and
sign in with it.

## 2. Plan B: can the agent run on a CPU instance? — Yes, slowly but usably
Measured (fp32, this host; EC2 vCPUs are hyperthreads, expect 1.5–2× slower):

| Component | 8 threads | 16 threads | Memory |
|---|---:|---:|---:|
| Strands Decider 2B load | 3.0 s | 2.2 s | **9.2 GB RSS** (torso upcast to fp32 on CPU by the package) |
| Decider, one routing question | 2.07 s | 1.72 s | |
| Decider, routing + halting (2 questions) | 2.58 s | 2.15 s | |
| UNI load | 3.1 s | 3.2 s | 2.3 GB RSS |
| UNI batch-1 per patch | 185 ms | 158 ms | |
| UNI 16 patches (sequential / batch-16) | 3.0 s / 2.8 s | 2.5 s / 2.0 s | |
| **Full agent loop, Decider Fixed-16 (`run_agent.py --device cpu`, 8 threads, both models in one process)** | **84 s** (5.1 s per decision, 184 ms per UNI call), end-to-end 94 s incl. model loads | | ~12 GB |

Instance: the Standard quota (8 vCPUs) allows exactly one **m7i.2xlarge (8 vCPU, 32 GB, ≈ \$0.40/h)**;
c7i.2xlarge (16 GB) is too tight for 9.2 + 2.3 GB plus the OS. Cost for 5–10 h of setup + smoke tests ≈ \$2–4,
EBS 100 GB gp3 ≈ \$0.01/h, S3 < \$0.10, Bedrock cents. No code change is needed: `run_agent.py --device cpu`
and `smoke_test.py --device cpu` already run; the halting question adds ~0.5 s per round in adaptive mode.
Decider confidence/choices on CPU are identical to GPU (same weights, fp32 vs bf16 differences only in the 3rd
decimal).

## 3. Hybrid fallback (only if CPU inference were unacceptable — it is not needed on these numbers)
AWS side: coordinator + Decider + classifier + Bedrock on the CPU instance. Local side: UNI on the RTX 4090
behind a localhost-only HTTP endpoint (bearer token, batch-1 `POST /embed` with the raw 224-px PNG), reached
from the instance through an SSH reverse tunnel (`ssh -R 8765:localhost:8765 ec2-user@…`) or Tailscale; the
coordinator's `PNGPatchUNIStore` gets a `RemoteUNIStore` subclass with the same `acquire()` contract, so the
accounting assertion (unselected candidates never encoded) still holds. Compute split would be stated as:
AWS = all decisions, evidence state, classifier, report; local = the 16 UNI forward passes. We do not recommend
it: it moves the expensive observation off AWS, which weakens the "agent on AWS" claim that WP007 must prove,
and CPU UNI is only 3 s per slide.

## 4. Recommended execution order once the account is active
1. `bash scripts/aws/check_environment.sh` → expect EC2/S3 OK, quota 8, Bedrock model access granted.
2. `AWS_REGION=us-east-1 BEDROCK_MODEL_ID=amazon.nova-micro-v1:0 python scripts/aws/smoke_test.py --test bedrock`
   from this host (real Converse call, request id + latency recorded) — Phase 2 evidence, no instance needed.
3. Create a private bucket, `bash scripts/aws/sync_demo_to_s3.sh` (26 MB).
4. IAM role for the instance (s3 read/write on the prefix, `bedrock:InvokeModel`/`Converse` on the model);
   `python scripts/aws/launch_instance.py --type m7i.2xlarge …` after cost approval (CPU AMI: Ubuntu 22.04,
   no NVIDIA driver needed); `bash scripts/aws/ec2_setup.sh`; `python scripts/aws/run_agent.py --slide-id … --budget 16
   --mode live --source s3 --device cpu`; `bash scripts/aws/stop_instance.sh <id>`.
5. If the G quota appeal is granted later, the same steps with `--type g6.xlarge --device cuda` (Plan A).

## 5. Parallel track (does not wait for AWS)
Repository scan for publication: 583 tracked files, largest 3.7 MB (OOF predictions), no secret patterns, no
weights, no WSIs. Items for WP008: `metadata/slides.csv` holds the private bevacizumab cohort's local paths and
case ids and must be excluded or anonymised before the repo goes public; the two decision-pattern documents
(`decision_patterns/README.md`, `decision_patterns/ADAPTIVE_ROUTER.md`) are current; the Builder article outline
(`docs/BUILDER_ARTICLE.md`) predates V2 and needs the TCGA-Lung and adaptive-STOP results.
