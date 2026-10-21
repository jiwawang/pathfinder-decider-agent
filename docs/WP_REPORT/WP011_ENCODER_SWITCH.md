# WP011 — Hackathon encoder switched to H-optimus-0 (2026-10-21)

Branch `release/public-repository`. Decider algorithm, prompts, shortlist, cheap features, classifier protocol and
CV untouched. UNI remains available as `encoder: uni` for the historical research configuration only.

## 1. Environment, cache and access
* `bioptimus/H-optimus-0` was already in the Hugging Face cache (4.3 GB `pytorch_model.bin`, config `license: Apache 2.0`),
  pulled in June for the 24-encoder TRIDENT benchmark; it loads fully **offline** (`HF_HUB_OFFLINE=1`) in the existing
  `pathfinder` env (timm 1.0.30, torch 2.14). No environment created, no package installed.
* Gated access: the model page requires accepting Bioptimus' terms; the cached snapshot proves the owner's account
  already has access. On a fresh machine (EC2) the owner must log in once with their own HF account; no token is
  stored in the repo or requested here.
* Reused code: the loading recipe (`timm.create_model("hf-hub:bioptimus/H-optimus-0", init_values=1e-5,
  dynamic_img_size=False)`) and the official normalisation (mean 0.707223/0.578729/0.703617, std 0.211883/0.230117/
  0.177517, 224 px, fp16 autocast on GPU) from the earlier benchmark implementation.

## 2. Code changes (data layer / configuration only)
* `src/encoders/registry.py`: `ENCODERS` (hoptimus0 default for demos, uni historical), `build_encoder`, `encode`
  (fp16 autocast only on CUDA; fp32 on CPU; L2-normalised float32 output).
* `LiveUNIFeatureStore(encoder=…)`, `TCGALiveUNIFeatureStore`, the cloud runner's `PNGPatchUNIStore`, `smoke_test.py`,
  `cache_uni_embeddings.py --encoder` all go through the registry. `configs/aws_demo.yaml: encoder: hoptimus0`;
  `configs/default.yaml: encoder.name: uni` documents the historical runs.
* Evidence memory, information gain, shortlist, STOP logic are dimension-agnostic (cosine on L2-normalised vectors);
  1536-d verified end to end. The runner asserts classifier/encoder binding (`encoder_name` in the artifact
  provenance and `n_features_in_ == 1536`).

## 3. Classifier heads (no UNI embeddings reused)
`scripts/hoptimus/encode_acquired_patches.py` re-encoded the **frozen** Decider-Fixed-16 acquisitions of all 198
TCGA-Lung slides with H-optimus-0 straight from the WSIs (3,168 patches, 64 s on one RTX 4090) into
`<layer>/hoptimus0_acquired_decider_fixed_16/`. `prepare_demo_artifacts.py --encoder hoptimus0` fit the fold
heads (seed 0, fold 1 for the LUAD demo slide, fold 2 for the LUSC one; 158 training slides each) with the audited
protocol. Sanity CV of the head on those embeddings (patient-level 5 × 5, `sanity_cv_hoptimus0_decider_fixed_16.json`):
**ROC-AUC 0.951 ± 0.007, balanced accuracy 0.897, F1 0.899** (UNI-era head on the same acquisitions: 0.964). This is
a head-generalisation check, not a benchmark re-run: the acquisitions themselves were made under the UNI run.

## 4. Real end-to-end smoke tests (local, GPU, `results/aws_smoke/`)
| run | slide | encoder calls | Decider calls | Decider median | encoder median | loop | prediction |
|---|---|---:|---:|---:|---:|---:|---|
| `20261021T161120Z-cca370` | LUSC TCGA-85-A4CL | 16 | 16 | 105 ms | 18.0 ms | 2.46 s | LUSC 0.9999 |
| `20261021T161141Z-91c855` | LUAD TCGA-95-7043 | 16 | 16 | 100 ms | 17.1 ms | 2.38 s | LUAD 0.996 |
Unselected candidates never encoded (asserted), 128 raw candidates per slide, encoder load 11.6 s, GPU memory 4.6 GB.
Encoder smoke (`smoke_uni.json`, name kept): PASS, shape 1536, 16 ms batch-1, reference embedding written for the
AWS cosine check. Video frames re-rendered from the LUAD run.

## 5. CPU EC2 risk (measured, not assumed)
H-optimus-0 on CPU, 8 threads, fp32: load 11 s, 5.6 GB RSS, **0.87 s per patch** (UNI was 0.18 s). Full loop with
Decider in one process: **91 s per slide**, end-to-end 109 s, **peak RSS 18.3 GB**. Consequence: the CPU plan needs
**m7i.2xlarge (32 GB)**; c7i.2xlarge (16 GB) is not viable. On g6.xlarge (24 GB GPU) Decider 4 GB + encoder 4.6 GB fit.

## 6. Documentation and export
README (§4b encoder section, §5, §10 compute, §11, §12 licences, §13), Builder article (encoder note, status),
video script/shot list (new run, H-optimus-0 timings, correct predictions), checklist updated. Historical UNI
results are labelled as such everywhere; no H-optimus-0 benchmark numbers are claimed. Public export regenerated,
scanned (no private ids/paths/secrets/weights) and tested (24 tests pass). Not pushed, not published.

## 7. Open items
* Bedrock model authorization (`Operation not allowed`) — AWS evidence still pending; EC2/S3 active; GPU quota 0.
* Owner approvals from WP010 (licence, export list, institutional IP) still pending.
* The benchmark tables remain UNI-era; a full H-optimus-0 re-run of Task 1 would need ~3 M encoder forwards
  (≈ 5 h on one GPU) and is explicitly out of scope for the hackathon.
