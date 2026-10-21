# WP008 — Public repository, reproducibility & submission compliance audit (2026-10-21)

## A. Git
* Source branch `integration/aws-bedrock` (contains WP004–WP007-1), source commit `bce12c2`.
* WP008 branch `release/public-repository`; new commits: quick start + example metadata + `.env.example`;
  allowlist export tool; README restructure + checklist; export redaction fix; this report.
* Working tree clean. No remote configured on the private repository; no GitHub repository created; nothing pushed.
* Preserved untouched: `main` (= `v2.1-audited`), `experiment/parallel-regional-search` (negative result),
  `experiment/tcga-generalization`, all tags, all local results.

## B. Security
* Tracked files audited: 583 (HEAD at audit time; 464 under `results/`). History audited over all 37 commits and
  all branches (content grep of every revision for AWS/HF/SSH key patterns: none; largest blobs 3.8 MB CSVs).
* Private material found: `metadata/slides.csv` (bevacizumab cohort local paths + case ids) — present in history
  since the baseline commit `e6556f7`; per-slide results of that cohort (`results/decider_*`, `threshold_v2`,
  `heuristic_v3*`, `live_demo`: OOF predictions, traces, acquisition/decider stats, reports — 352 files) carry
  the cohort's slide ids; `docs/WP_REPORT/WP004_CODE_AUDIT.md` lists fold-0 patient ids; absolute local paths,
  interpreter paths and the host name appear in configs, scripts, result configs and smoke traces.
* Secrets: none in any commit. AWS account id: never written to the repository (it appeared once in a tool
  output during WP007-1; it is not in any file).
* Consequence: **the private repository's history is not public-safe and must not be pushed.** A separate
  allowlist export with a fresh history was built instead (section F).

## C. Licensing
* Recommended OSI licence for the code: **Apache-2.0** — it is already in the repo since the first commit, it
  matches Strands Decider / strands-decider (Apache-2.0), it carries an explicit patent grant, and it is
  compatible with every runtime dependency (PyTorch BSD, timm Apache-2.0, scikit-learn BSD, boto3 Apache-2.0,
  OpenSlide-python LGPL-2.1 as a dynamically used library). MIT would also qualify; Apache-2.0 is the lower-risk
  choice for a submission that AWS may redistribute. **LICENSE APPROVAL: PENDING** (owner decision).
* Third-party code copied into the repo: none (TRIDENT is called as an external tool; the D001 reading rule was
  re-implemented). All contributions are the owner's with AI assistance.
* UNI: third-party gated model, CC-BY-NC-ND-4.0, non-commercial research only; weights never distributed; README
  §12 states the limits and that competition/public-deployment coverage is not asserted. Whether the hackathon
  demo is "non-commercial" is an **open release item for the owner** (also raised in WP007).
* Strands Decider 2B + Qwen3.5-2B base: Apache-2.0. TCGA: public (GDC data use policies apply).

## D. Reproducibility
* README rewritten to the 14-section structure; explains that the Decider acts *before* the encoder, the typed
  questions/options, the three reproduction levels, licences and limitations; states "AWS integration pending
  account authorization" and does not present local runs as AWS results.
* Quick start (Level 1): `scripts/quickstart_decision_pattern.py` — synthetic cheap features, mock expensive
  observation, real shortlist → Decider (heuristic or Strands) → lazy acquisition → evidence/STOP wiring; labelled
  as an interface demo. Runs in ~1 s on CPU.
* Decision pattern: `decision_patterns/adaptive_expensive_observation_router.py` + `ADAPTIVE_ROUTER.md`; code
  and docs cross-checked (routing question, halting question, options, guard-rails). The wiring
  Decider → option → candidate index → `acquire()` → UNI → evidence update is in `src/acquisition/adaptive_agent.py`
  and exercised by `tests/test_v21_adaptive.py`, the quick start and `scripts/aws/run_agent.py`.
* Tests in the export: 23 passed (private repo: same). README script/config/doc references: all resolve.
  `configs/default.yaml` private-cohort paths replaced by documented placeholders. `.env.example` has names only.
* Environments: README documents the single `pathfinder` env (both models verified to co-exist) and notes the
  legacy `UNI` env; `requirements-aws.txt` is the exact freeze. No environment was recreated or upgraded.

## E. Competition compliance (official requirements)
| Requirement | Status |
|---|---|
| Working agent using Strands Decider 2B paired with Amazon Bedrock, hosted on AWS | code complete; **AWS execution pending Bedrock model authorization** (update 2026-10-21: EC2 and S3 are now active; Bedrock Converse/InvokeModel still return "Operation not allowed"; GPU quota 0, CPU plan ready) |
| Reusable decision pattern with typed question, options, wiring, usage | done (`decision_patterns/ADAPTIVE_ROUTER.md`, quick start) |
| Public repository with OSI licence | public-safe export ready; repo creation and LICENSE approval pending owner |
| Builder Center article | not started (outline excluded from export as outdated) |
| Demo video < 3 min | not started |
| Tags `#strands-decider-2b`, `#context-memory` | in README and pattern docs; `#context-memory` justified (controls what enters the shared evidence context) |

## F. Public export
* Path: `~/Project/Hackthon_AWS_public` (`scripts/release/export_public.py`, allowlist-based, fresh `git init`,
  one commit "Initial public release (exported from private commit <sha>)", no remote).
* Included: 224 files (+ `PUBLIC_EXPORT_MANIFEST.json`): `src/`, `decision_patterns/`, `tests/`, `scripts/`,
  `configs/`, `app/`, README, LICENSE, requirements, `.gitignore`, `.env.example`, `docs/WP_REPORT/*`,
  `docs/SUBMISSION_CHECKLIST.md`, `metadata/slides.example.csv`, TCGA manifests/tables (path columns dropped),
  GDC raw responses, result aggregates of the private cohort (summaries, curves, decider summaries, config_used
  with redacted paths), the full TCGA result directories (public identifiers), sanitised AWS smoke traces.
* Excluded: 366 files — `metadata/slides.csv`; 352 private-cohort per-slide result files; 12 internal docs
  (`docs/WP/*`, `docs/Doc/D001.md`, `docs/TASK_BRIEF.md`, `docs/BUILDER_ARTICLE.md`); the export tool itself.
* Redactions: local absolute/interpreter paths → placeholders; private pipeline/root names; host names; private
  slide/case identifiers; CSV path columns dropped.
* History safety: export history is a single commit from redacted content; re-scan of the export (content and
  history): private ids 0, private roots/home paths 0, host names 0, secrets 0, data/weight files 0.
* Validation in the export: 23 tests pass, quick start runs, imports resolve, README references resolve.
* Ready to publish? **Ready for user review** — not published.

## G. Outstanding issues
| Level | Issue |
|---|---|
| CRITICAL | none (no credentials anywhere; private history will not be pushed) |
| HIGH | Public GitHub repository not yet created/approved; LICENSE approval pending; AWS execution evidence missing (Bedrock model authorization; EC2/S3 active since 2026-10-21) — the "AWS-hosted agent" requirement is unmet until then |
| MEDIUM | UNI licence position for a public, prize-bearing demo unresolved; Builder article and video not started; the `app/streamlit_app.py` V1 UI needs private-cohort thumbnails and is documented as legacy |
| LOW | `<TCGA_ROOT>/...` default paths remain in TCGA configs (generic data root, not private); result `config_used.yaml` files keep placeholder-redacted paths |

## H. Final verdict
**READY FOR USER REVIEW** (not published). Blockers for publishing: owner approval of the licence and of the
export contents; owner decision on the UNI licence statement; creation of the public repository by the owner.

## Answers to the seven questions
1. Can the current repository be made public safely? The private development repository — **no** (private cohort
   metadata and per-slide results are in its history). The export in `~/Project/Hackthon_AWS_public` — yes, after your review.
2. Remaining blockers: your approval of the licence and export; the UNI licence statement; AWS evidence is a
   compliance gap, not a publication blocker (EC2/S3 active; Bedrock authorization pending).
3. Separate public repository without private history? **Yes, required**; it is built (one commit, no remote).
4. Licence: **Apache-2.0** — already present, matches Strands, patent grant, dependency-compatible; MIT acceptable too.
5. Can others run the reusable pattern? Yes: `python scripts/quickstart_decision_pattern.py` (no data, no GPU);
   with `--backend strands` it uses the real Decider; tests pass in the export.
6. Features depending on the still-unavailable Bedrock/EC2 permissions: the Bedrock-generated report (template
   fallback otherwise), the S3-sourced cloud run, and all AWS timing evidence.
7. Needs your approval before publishing: licence choice, the export file list (`PUBLIC_EXPORT_MANIFEST.json`),
   the UNI licence wording in README §12, and the act of creating/pushing the public repository.
