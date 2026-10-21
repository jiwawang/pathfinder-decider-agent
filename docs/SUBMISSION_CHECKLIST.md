# Submission checklist (hackathon qualification is pass/fail)

Deadline: submissions due **October 23, 2026, 11:59 PM PDT**. Rubric: creativity 30% · effective Decider 2B use 30% ·
communication 25% · originality 15%. Only items that are really done are checked.

- [ ] AWS-hosted working agent — code ready (`scripts/aws/run_agent.py`, local dry runs in `results/aws_smoke/`); **AWS integration pending account authorization** (EC2/S3 not activated, Bedrock "Operation not allowed"; GPU quota appeal pending; CPU plan ready)
- [x] Encoder for all hackathon demos switched to H-optimus-0 (Apache-2.0); UNI results labelled historical (WP011)
- [x] Real Strands Decider 2B decision — routing (A/B/C/D) and halting (CONTINUE/STOP) per round, inside every CV fold; ~51k decisions on the private cohort, ~2.5k per TCGA run
- [ ] Real Amazon Bedrock integration — adapter ready (`src/bedrock/reporter.py`, Converse API); real call blocked by account state (smoke evidence: `results/aws_smoke/smoke_bedrock.json`)
- [ ] Public GitHub repository — owner approved (WP012); target https://github.com/jiwawang/pathfinder-decider-agent, tag `v1.0-hackathon`; push status in `docs/WP_REPORT/WP012_PUBLIC_RELEASE.md`
- [x] OSI-approved LICENSE — Apache-2.0, owner approved (WP012)
- [x] Reusable decision pattern — `decision_patterns/ADAPTIVE_ROUTER.md` + runnable wiring + synthetic quick start
- [ ] Builder Center article — **full English draft ready for publication review** (`docs/SUBMISSION/BUILDER_ARTICLE.md`); not published; AWS evidence placeholder to fill
- [ ] Public demo video < 3 minutes — narration script with timings and shot list ready (`docs/SUBMISSION/VIDEO_SCRIPT.md`, `VIDEO_SHOTLIST.md`); real-run frames rendered; recording/editing/upload not done
- [x] `#strands-decider-2b` — in README and pattern docs
- [x] `#context-memory` — in README and pattern docs (why: controls what enters the agent's evidence context)
- [ ] Final submission before the deadline
