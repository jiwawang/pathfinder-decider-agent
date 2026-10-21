# WP012 — PathFinder v1.0 public GitHub release (status 2026-10-21, IN PROGRESS)

Owner approval received (WP012): Apache-2.0, H-optimus-0 encoder, the reviewed public export, repository creation,
tag `v1.0-hackathon`. Nothing private, no data, no weights, no paid AWS, no article/video publication, no final form.

## 1. GitHub public repository URL
Planned: https://github.com/jiwawang/pathfinder-decider-agent (name verified free via the public API; fallback
`pathfinder-wsi-agent` also free). **Not created yet** — see §8.

## 2. Public commit SHA
Current export at `~/Project/Hackthon_AWS_public`: single commit `1578818`, exported from private commit
`cb450e4` (the WP011-reviewed 185-file version). A refreshed export from private commit `0fedb02` is pending (§8).

## 3. Release tag URL
Planned: https://github.com/jiwawang/pathfinder-decider-agent/releases/tag/v1.0-hackathon — not created yet.

## 4. LICENSE status
Apache-2.0 (`LICENSE`, full text, 2004 version), owner approved. Model weights are not covered and not distributed
(H-optimus-0 Apache-2.0 gated on Hugging Face; UNI CC-BY-NC-ND-4.0, historical runs only).

## 5. Public file count
185 files + `PUBLIC_EXPORT_MANIFEST.json` (186 tracked). Manifest ↔ tree: every tracked file is in the manifest,
every manifest entry exists, all 185 SHA-256 digests match. 18 MB; largest files are two public-TCGA OOF prediction
CSVs (3.8 MB, 3.7 MB). No `.bin/.safetensors/.pth/.pt/.svs/.h5/.npy/.joblib`, no `.env`, no `.pem`.

## 6. Security scan (export from `cb450e4`)
* AWS/HF/GitHub tokens, private keys: none.
* Private cohort identifiers / pipeline name / host name: none (only the generic phrase "private bevacizumab cohort"
  in licence/limitation text, accepted in the WP008 audit).
* 12-digit numbers: only UUID fragments in public TCGA manifests and float digits in traces; no account id.
* **Finding fixed before release:** two copies of experiment configs (`results/tcga_generalization/task*/config.yaml`)
  still carried the local TCGA data root `<TCGA_ROOT>/...` (public data, but a local absolute path). The export
  script now redacts it to `<TCGA_ROOT>` (private commit `0fedb02`); the export must be regenerated before push.
* README: `<GITHUB_URL_PLACEHOLDER>` and "pending owner approval" replaced by the real URL / approval (`0fedb02`).
  Encoder default `hoptimus0` in `configs/aws_demo.yaml` and `src/encoders/registry.py`; §11 states AWS integration
  is pending Bedrock model authorization; no local run is presented as a cloud run; UNI tables labelled historical.

## 7. Test results (inside the export)
`pytest tests`: 24 passed (1.1 s). `scripts/quickstart_decision_pattern.py`: OK (synthetic, no data/GPU/AWS).

## 8. Is the GitHub release complete? **No.** Two blockers, both outside what the agent may do alone
1. **No GitHub CLI and no API credential on this machine.** `gh` is not installed (apt candidate 2.45.0 exists, needs
   sudo); there is no `~/.config/gh`, no git credential helper. SSH works (`ssh -T git@github.com` authenticates as
   `jiwawang`), so the *push* can be done over SSH — but creating the repository, setting the description and the
   topics needs either `gh auth login` (owner runs it) or the web UI. No token was requested or stored.
2. **Regenerating the export was denied by the session's permission mode** (the export script deletes and re-creates
   the destination directory). The owner can run it directly:
   `python scripts/release/export_public.py --dest ~/Project/Hackthon_AWS_public`

Once (1) and (2) are done the remaining steps are: re-run the scan/tests in the export, `git remote add origin
git@github.com:jiwawang/pathfinder-decider-agent.git`, `git push -u origin main` (no force), `git tag -a v1.0-hackathon`,
`git push origin v1.0-hackathon`, verify via the public API (default branch, licence detection, README, tag → commit).

## 9. Hackathon tasks still open
* GitHub release (this WP), then update this report with the real SHA/URLs.
* Bedrock model authorization (`Operation not allowed`) → real AWS run evidence; GPU quota 0 (CPU plan m7i.2xlarge).
* Builder article publication, demo video recording/upload, final submission form (all owner actions, not approved
  for the agent).
