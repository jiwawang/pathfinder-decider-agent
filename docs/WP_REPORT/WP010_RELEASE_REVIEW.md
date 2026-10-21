# WP010 — v1.0-hackathon release review (2026-10-21; nothing published)

Candidate: `~/Project/Hackthon_AWS_public` regenerated from private commit `35ef948` on
`release/public-repository`. 175 files, 13 MB, one commit, no remote. Private development repository and its
full history untouched (`main`, two experiment branches, `integration/aws-bedrock`, `release/public-repository`, tags).

## 1. What v1.0 contains (and what was removed vs the WP008 export)
Kept: the complete agent (`src/`), the decision pattern (`decision_patterns/`), tests (23), the synthetic quick
start, the research runners (`scripts/*.py`, `scripts/tcga/`), the AWS entrypoint/smoke/launch/stop/teardown
scripts, configs, README/LICENSE/requirements/.env.example, eight WP reports (WP002, WP003, WP004, WP005 survey,
WP005 TCGA, WP006, WP007 Plan B, WP008 audit, this review), the main result aggregates (V1 `decider_v2`,
V2 `decider_v3_active`, V2.1 `decider_v3_adaptive`, `heuristic_v3`, `threshold_v2`, TCGA Task 0 and Task 1 in
full with public identifiers), sanitised AWS smoke traces, the frame renderer.
Removed for v1.0 (still in the private repo): superseded/failed/smoke runs (`decider_v1`, `decider_v2_random_order`,
`decider_v3_joint_stop`, `decider_v3_adaptive_v0_failed`, `_v1_sharedstate`, `heuristic_v3_smoke`, Task 0 smoke),
the V1 Streamlit UI and OBR-only demo/thumbnail scripts, internal planning docs (`WP005_PLAN`, outdated
`WP007_AWS_READINESS`, all `docs/WP/*` briefs, `docs/Doc`, `docs/SUBMISSION/*` drafts, `docs/RESEARCH/*`).
No key implementation was removed: `run_agent.py`, `adaptive_agent.py`, `lazy_uni_store.py`, the router
pattern, the Bedrock reporter and the TCGA pipeline are all present and tested.

## 2. Final security check (export content and its single-commit history)
Private cohort ids 0 · private/home paths 0 · host names 0 · credential patterns 0 · WSI/weights/embeddings 0 ·
largest file 3.7 MB (TCGA OOF predictions, public identifiers). Tests pass in the export; quick start runs; every
README reference resolves. `PUBLIC_EXPORT_MANIFEST.json` lists the 175 files and the exclusion categories.

## 3. Licence recommendation: Apache-2.0 (not GPL-3.0)
| | Apache-2.0 | GPL-3.0 |
|---|---|---|
| Closed-source commercial reuse of the code | allowed (attribution + notice) | not allowed: derivatives must be GPL and ship source |
| Our future paper / research | unaffected either way; Apache keeps collaborations with industry partners frictionless | GPL can deter industrial co-authors and some lab policies |
| Hackathon / AWS redistribution of the entry | explicitly fine | fine, but copyleft obligations propagate to anyone embedding the code |
| Patent grant | explicit | explicit (§11) |
| Compatibility with dependencies | strands-decider Apache-2.0, timm Apache-2.0, PyTorch BSD, scikit-learn BSD, boto3 Apache-2.0, OpenSlide-python LGPL-2.1 (dynamic use) — all compatible | compatible as well (GPL may wrap permissive code) |
| Third-party *model* terms | neither licence touches them: UNI stays CC-BY-NC-ND-4.0, Decider/Qwen Apache-2.0 | same |
Recommendation: **Apache-2.0** — it is what the repo has carried since the first commit, it matches the Strands
ecosystem, and it maximises reuse of the pattern, which is the hackathon's stated goal. GPL-3.0 would only be
preferable if the owner wants to prevent closed-source commercial forks. **Owner approval pending.**

## 4. Code ownership — to be confirmed by the owner
All code in the export was written in this project (with AI assistance) from October 15 2026 on; no code was
copied from the CCT/UNI-distillation repository (the D001 reading rule was re-implemented; TRIDENT is invoked
as an external tool; the earlier UNI feature files are data, not code). Whether the University of Oklahoma or
the lab holds rights over work produced by a student/staff member on institutional resources depends on the
institution's IP policy — **not determinable here; marked for the owner to confirm before publication.**

## 5. UNI licence and a compliant demo alternative
UNI (CC-BY-NC-ND-4.0, gated): the repository does not distribute weights; README §12 states the restrictions and
that competition/public-deployment coverage is not asserted. If the owner cannot confirm that a prize-bearing
hackathon demo is covered, the compliant alternative is to run the public demo with an **Apache-2.0 pathology
encoder** via the existing config switch (`demo.uni_model` / the encoder name in the stores): candidates
`bioptimus/H-optimus-0` (Apache-2.0; measured on the private cohort in the earlier 24-encoder benchmark at
AUROC 0.706 vs UNI 0.672 under the same protocol) or `prov-gigapath` (Apache-2.0). Switching the demo encoder
does not alter any frozen result; the benchmarks would remain reported with UNI and labelled as such. This is a
proposal only; nothing was re-run.

## 6. GitHub metadata (prepared, not applied)
* Description: "PathFinder — a Strands Decider 2B agent that decides where expensive pathology foundation-model
  inference should run on gigapixel slides, with Amazon Bedrock execution reports. Research prototype."
* Topics: `strands-decider-2b`, `context-memory`, `aws-bedrock`, `strands-agents`, `computational-pathology`,
  `whole-slide-imaging`, `active-acquisition`, `decision-model`, `uni`, `agent-experience-hackathon`.
* Release: tag `v1.0-hackathon` on the export's initial commit after approval
  (`cd ~/Project/Hackthon_AWS_public && git tag -a v1.0-hackathon -m "PathFinder hackathon release"`), then
  create the repository (private first, review, then public) and push `main` + tag. Replace
  `<GITHUB_URL_PLACEHOLDER>` in README §14 and in the article/video before publishing those.
* Default branch `main`; no secrets scanning exceptions needed; enable Dependabot alerts.

## 7. Pre-publication acceptance checklist
- [ ] Owner approves the licence (Apache-2.0 recommended)
- [ ] Owner approves `PUBLIC_EXPORT_MANIFEST.json` (175 files)
- [ ] Owner confirms institutional IP position
- [ ] Owner confirms the UNI position for the public demo (or chooses the Apache-2.0 encoder alternative)
- [ ] Regenerate the export from the approved private commit, re-run `pytest`, re-run the security scan
- [ ] Create repository, push, tag `v1.0-hackathon`, fill the URL placeholders
- [x] No credentials, private ids, private paths, weights or WSIs in the export (verified today)
- [x] Tests and quick start pass in the export

## A. Final licence and scope conclusion
Apache-2.0 for the code, unchanged; scope = the 175-file v1.0 export above (agent, pattern, quick start, AWS
scripts, TCGA pipeline, aggregate results, reports). Private cohort data, per-slide private results, drafts and
internal briefs stay private.

## B. What is still missing for a valid submission
1. **AWS execution evidence** — EC2 and S3 are active; the blocker is now **Bedrock model authorization**
   ("Operation not allowed" for nova-micro and claude-haiku-4-5). Once granted: Bedrock smoke from this host,
   then one m7i.2xlarge (GPU quota is still 0) run of `run_agent.py --source s3 --device cpu` for the trace.
   Also: stop using the account root (create an IAM user/role) before any deployment.
2. **UNI compliance decision** for the public demo (§5).
3. **Public repository** creation/push after approvals; URL placeholders in README, article, video.
4. **Article publication** on Builder Center (draft complete) and the **demo video** (script/shot list/frames
   complete; recording and upload outstanding).
5. Final submission form with tags `#strands-decider-2b` and `#context-memory` before Oct 23, 11:59 PM PDT.

## C. Boundary between the hackathon version and future paper work
Hackathon v1.0 (frozen, public): the label-agnostic acquisition agent, the two typed Decider questions, the
information-gain state, the lazy accounting store, the baselines, the audit, the TCGA cross-task benchmarks and
the honest conclusion that decision-guided routing matches but does not beat cheap heuristics while providing
adaptive, traceable budget control. Paper (private roadmap `docs/RESEARCH/PATHFINDER_PAPER_ROADMAP.md`):
uncertainty-aware and cost-aware controllers, learned routers, hospital-grouped primary validation on new
cohorts, pre-registered budgets, encoder swaps and the statistical machinery to make a selection-bias-free claim.
None of the paper work is started, and none of it is implied by the public release.
