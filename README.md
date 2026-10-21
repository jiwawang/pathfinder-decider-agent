# PathFinder — an agent that decides where expensive pathology inference should run

**AWS Agent Experience Hackathon** · `#strands-decider-2b` · `#context-memory`

PathFinder is a research prototype that searches a gigapixel whole-slide image (WSI) the way a budget-aware
analyst would: it looks at cheap descriptions of every tissue region, asks **Strands Decider 2B** which region
deserves the next *expensive* observation (a pathology foundation-model forward pass — H-optimus-0 in the hackathon
demo, UNI in the historical research runs), runs that one inference, adds
the result to a shared evidence memory, and asks the Decider whether the evidence is saturating. Only the
acquired patches form the slide representation that a small classifier reads. **Amazon Bedrock** turns the
agent's trace into a plain-language execution report.

> Research prototype on public TCGA lung slides and a private treatment-response cohort. Not a diagnostic or
> clinical tool. The decision model is label-agnostic: it never sees a diagnosis.

## 1. Project overview
* Problem: a WSI holds 10 000–45 000 tissue patches; encoding all of them with a pathology foundation model costs
  ~1–2 GPU-minutes per slide, yet slide-level tasks are often decided by a handful of regions.
* Idea: put the decision model **before** the encoder. Cheap features (colour, texture, position; ~20 ms per
  patch on CPU) describe every region; the Decider routes the expensive encoder to one region per round and
  decides when to stop.
* Result: on TCGA LUAD-vs-LUSC the agent reaches the full-slide-encoding accuracy (ROC-AUC 0.964 vs 0.963)
  with **16 instead of ~15 000 encoder calls** per slide; end-to-end wall-clock is ~2× lower (the cheap pass still
  reads every patch once). Decision-guided routing beats random sampling clearly and matches strong cheap
  heuristics — see §10 for the honest comparison.

## 2. The problem it solves for agents
Many agents face "many cheap observations, one expensive sensor/model, limited budget". The reusable pattern here
(the *Adaptive Expensive-Observation Router*, §4) answers two typed questions per round — *where next?* and
*enough?* — with a small, fast, calibrated decision model instead of an LLM.

## 3. Architecture (the Decider acts before the encoder)
```
WSI ─► tissue segmentation (TRIDENT or saturation mask) ─► 224-px patch grid at 0.5 µm/px (112 µm per patch)
    ─► CHEAP features per patch: RGB/HSV statistics, entropy, edges, position, local tissue density      (no encoder)
    ─► KMeans on cheap features (K=128) ─► candidate pool of real patches
    ─► round r: shortlist of 4 candidates by 4 exploration strategies (largest unexplored cluster, cheap novelty,
       spatially farthest, under-covered neighbourhood); option letters shuffled
    ─► Strands Decider 2B  ┬─ "Is the evidence saturating? Compare recent and initial information gains"  → CONTINUE / STOP
                           └─ "Which candidate should receive the next expensive UNI inference?"          → A / B / C / D
    ─► the ENCODER runs on THAT ONE patch (demo: H-optimus-0, 1536-d; historical runs: UNI, 1024-d) — LazyUNIFeatureStore.acquire:
       unacquired rows are unreadable, every read is counted
    ─► shared evidence memory: embeddings, information gain = 1 − max cosine to prior evidence, coverage
    ─► … until STOP or the hard maximum (4 ≤ calls ≤ 32) ─► mean pooling ─► logistic regression ─► slide prediction
    ─► Amazon Bedrock (Converse API) ─► execution report grounded in the trace (never a diagnosis)
```
It is **not** "encode everything, then let the Decider filter embeddings": candidate generation and shortlisting
use only cheap features; the store raises if any code touches an embedding before the Decider chose it
(`tests/test_v2_acquisition.py`).

## 4. Strands Decider 2B decision pattern — Adaptive Expensive-Observation Router
`decision_patterns/adaptive_expensive_observation_router.py` + `decision_patterns/ADAPTIVE_ROUTER.md`
(generic), wired for pathology in `src/acquisition/adaptive_agent.py`.

| | Routing question | Halting question |
|---|---|---|
| state | budget used / max, remaining candidates, evidence redundancy, per-candidate cheap stats (novelty, distance, coverage, tissue, entropy, representativeness, why shortlisted) | information gain of every acquisition, last-4 mean/min/max/slope, mean of first vs last acquisitions and their ratio, coverage |
| typed question | *Which candidate should receive the next expensive UNI inference?* | *Is the evidence from expensive UNI observations saturating? Compare the recent information gains with the initial ones.* |
| options | `A` `B` `C` `D` (each with its strategy + numbers) | `CONTINUE` / `STOP` |
| guard-rails outside the model | low-confidence answer → heuristic's best option | STOP offered only between `min_uni_calls` and `max_uni_calls`; forced at max |

Why `#context-memory`: the pattern decides which expensive observations enter the agent's shared evidence
context and when that context is complete. Other uses: video frames → large VLM, retrieved chunks → deep
reranker, low-res tiles → high-resolution scan. Earlier V1 pattern (post-encoding SELECT/SKIP/STOP) is kept in
`decision_patterns/evidence_selector.py` / `README.md`.

```python
from src.decider.client import StrandsDeciderBackend
from src.acquisition.adaptive_agent import AdaptiveRouterV21, run_adaptive_episode
router = AdaptiveRouterV21(StrandsDeciderBackend("StrandsAgents/strands-decider-2B-qwen3.5-v1-2610"), min_calls=4, max_calls=32)
episode = run_adaptive_episode(slide_id, candidate_pool, lazy_store, router, cfg)   # trace, acquisitions, embeddings
```

## 4b. Encoder: H-optimus-0 for the hackathon, UNI for the historical results
`src/encoders/registry.py` makes the expensive encoder configurable. **The hackathon demo, the cloud entrypoint and
the smoke tests use `bioptimus/H-optimus-0` (Apache-2.0, 1.1 B parameters, 1536-d, official vendor preprocessing,
fp16 autocast on GPU).** All benchmark numbers in §10 were produced with UNI before this switch and are kept as
**historical UNI experiments**; they are not H-optimus-0 results and not AWS results. The demo classifier heads
were retrained on H-optimus-0 embeddings of the same frozen Decider-Fixed-16 acquisitions (patient-level 5×5 CV
sanity check on TCGA lung: ROC-AUC 0.951 ± 0.007, `results/tcga_generalization/task1_luad_lusc/sanity_cv_hoptimus0_decider_fixed_16.json`).
A classifier is bound to its encoder: the runner refuses a head trained on another encoder's embeddings.

## 5. Installation
```bash
conda create -n pathfinder python=3.11 && conda activate pathfinder
pip install -r requirements.txt            # strands-decider, torch, timm, scikit-learn, openslide-python, boto3 …
# exact versions used for every reported run: requirements-aws.txt (pip freeze)
```
Both models (Decider 2B and the encoder) run in this one environment (torch ≥ 2.10, transformers ≥ 5.15, timm ≥ 1.0.30).
A second `UNI` environment (timm + transformers 4.x) was used only for the earliest offline feature extraction and
is not required. System packages: `openslide-tools`; a GPU is optional (CPU, 8 threads: ~2 s per Decider decision,
~0.9 s per H-optimus-0 patch, ~18 GB RAM for the full loop). H-optimus-0 and UNI are both gated on Hugging Face:
accept the model terms with your own account and log in; no weights are redistributed here (§12).

## 6. Configuration
`configs/default.yaml` (all protocol values: pool size, shortlist, budgets, prompts, classifier, CV),
`configs/tcga_frozen.yaml` (the frozen protocol used for TCGA), `configs/tcga_task*.yaml` (data paths and labels),
`configs/aws_demo.yaml`. AWS settings come **only** from the environment — copy `.env.example`, never commit
`.env` or credentials.

## 7. Quick start (Level 1 — no data, no GPU, no AWS)
```bash
python scripts/quickstart_decision_pattern.py                    # model-free router, synthetic data, ~1 s
python scripts/quickstart_decision_pattern.py --backend strands  # real Strands Decider 2B (downloads ~4.5 GB)
python -m pytest tests -q                                        # 23 tests: leakage guards, lazy store, pattern wiring
```
The quick start uses synthetic cheap features and a **mock** expensive observation (random vectors). It exercises
the real wiring — shortlist → Decider → acquisition → evidence update → STOP — and prints the trace. It is an
interface demo, not a pathology prediction.

## 8. Running the agent on real slides
```bash
# Level 2 — research pipeline on your own slides (needs the WSIs, TRIDENT or the saturation mask, UNI access)
python scripts/tcga/extract_cheap_features_tcga.py --config configs/tcga_task1.yaml
python scripts/tcga/cache_uni_embeddings.py        --config configs/tcga_task1.yaml   # for the cached simulation + Full-UNI baseline only
python scripts/tcga/run_task0_cv.py                --config configs/tcga_task1.yaml   # all baselines + Decider, 5×5 CV
python scripts/tcga/bench_task0_live.py            --config configs/tcga_task1.yaml   # on-demand UNI, live timings
# Level 3 — cloud entrypoint (see §11)
python scripts/aws/run_agent.py --slide-id <id> --budget 16 --mode live --source s3
```

## 9. Reproducing the benchmarks
Every result directory holds `config.yaml`/`config_used.yaml`, `metrics.csv`/`summary.csv`, `budget_curve.csv`,
`summary.md` and, for public TCGA data, the full out-of-fold predictions, acquisition statistics and
`manifest_snapshot.csv` with GDC file ids and md5s. TCGA data is selected by the frozen manifests
(`metadata/tcga_*_manifest.csv`, lowest-DX diagnostic primary-tumour slide per patient, seed-0 shuffle) and can be
re-downloaded with `scripts/tcga/download_manifest.py`. The private bevacizumab cohort is not distributed; its
per-slide files are excluded from this repository and only aggregates are reported.

## 10. Experimental results
**TCGA LUAD vs LUSC** (198 patients, 98/100; patient-level StratifiedGroupKFold, 5 folds × 5 seeds; mean ± std over
seeds; `results/tcga_generalization/task1_luad_lusc`, `docs/WP_REPORT/WP006_LUNG_REPORT.md`):

| Method | UNI calls / slide | ROC-AUC | Bal. acc. | F1 |
|---|---:|---:|---:|---:|
| Full UNI (all patches) | 15,342 | 0.963 ± 0.009 | 0.891 | 0.895 |
| Random-16 | 16 | 0.935 ± 0.013 | 0.869 | 0.874 |
| Spatial uniform-16 | 16 | 0.945 ± 0.009 | 0.861 | 0.868 |
| Cheap Top-16 | 16 | 0.966 ± 0.008 | 0.903 | 0.905 |
| Cheap Greedy-16 | 16 | 0.967 ± 0.008 | 0.903 | 0.906 |
| Decider Fixed-16 | 16 | 0.964 ± 0.007 | 0.895 | 0.897 |
| Decider Adaptive STOP | 19.6 (7–32) | 0.962 ± 0.008 | 0.893 | 0.896 |

The Decider beats random/spatial sampling (paired Δ +0.014…+0.040, CIs exclude 0 for two of three draws) and is
indistinguishable from the cheap-feature heuristics (Δ −0.002/−0.003). K = 16 was pre-specified. The two TCGA
projects come from partly different hospitals (tissue-source-site Cramér's V = 0.62); site-grouped CV gives
≈0.02 lower AUROC with the same ranking. This is a cross-task test of the *acquisition strategy*, not a transfer
of any classifier.

**TCGA organ (COAD/KIRC/LUAD, 173 patients)**: every policy saturates at 4–8 calls (macro AUROC ≥ 0.998);
the label-agnostic adaptive STOP still spends ~19 calls — evidence need is task-dependent and the controller
tracks embedding redundancy, not task difficulty (`docs/WP_REPORT/WP005_TCGA_GENERALIZATION.md`).

**Private bevacizumab-response cohort (79 slides, 44/35; development set)**: Full UNI 0.679 ± 0.025; Decider
Fixed-16 0.728 ± 0.019; Cheap Top-16 0.723; Random-16 0.680. This cohort was used to develop the prompts and
K = 16 was highlighted after seeing the budget grid, so 0.728 carries development-set selection bias; an
independent audit found no train/validation leakage (`docs/WP_REPORT/WP004_CODE_AUDIT.md`). It is not an
untouched external test result.

**Compute** (RTX 4090, live; historical UNI runs): UNI 7–13 ms per on-demand patch; Decider ~100 ms per routing
decision (~150 ms with the halting question); one slide loop ≈ 2 s. With H-optimus-0 (hackathon demo, real runs in
`results/aws_smoke/`): 17–18 ms per on-demand patch, loop ≈ 2.4 s per slide on GPU; on CPU (8 threads) 0.87 s per
patch, loop ≈ 91 s, peak RSS 18 GB. Encoder calls drop ~1,000×; end-to-end time drops ≈ 2× because the cheap
features still read every patch (~17 ms each, parallelisable). Never read the call reduction as an end-to-end speed-up.

## 11. AWS deployment
Minimal architecture: one EC2 instance (g6.xlarge GPU, or m7i.2xlarge CPU with 32 GB — the 16 GB c7i is too small
for Decider + H-optimus-0) runs the Decider, H-optimus-0 and the coordinator; a private S3 bucket holds compact demo artifacts (thumbnail, cheap features, 128 candidate
coordinates and their raw patches — **no embeddings**); Amazon Bedrock generates the report. Scripts:
`scripts/aws/check_environment.sh`, `launch_instance.py`, `ec2_setup.sh`, `sync_demo_to_s3.sh`, `run_agent.py`,
`smoke_test.py`, `stop_instance.sh`, `teardown.sh`. Status: **AWS integration pending Bedrock model authorization** —
EC2 and S3 are active on the submission account, the GPU quota is still 0 (CPU plan: m7i.2xlarge, ~90 s per
slide), and Bedrock inference returns "Operation not allowed" until model access is granted; local dry runs of
the identical entrypoint are in `results/aws_smoke/` (`docs/WP_REPORT/WP007_*`). Do not treat the local runs as
AWS results.

## 12. Dataset & model licences
* **Code**: Apache-2.0 (`LICENSE`), approved by the owner for the v1.0-hackathon public release.
* **H-optimus-0** (bioptimus/H-optimus-0, Apache-2.0) is the hackathon encoder; access is gated on Hugging Face
  (accept the terms with your own account); weights are not redistributed here.
* **UNI** (MahmoodLab/UNI) is a third-party, gated model under CC-BY-NC-ND-4.0 (non-commercial academic research,
  no derivatives, no redistribution). It was used only for the historical research benchmarks in §10 and is not
  used for any hackathon demo or cloud run. This repository does **not** redistribute UNI weights; the code licence
  does not cover any model weights.
* **Strands Decider 2B** (StrandsAgents) and its Qwen3.5-2B base: Apache-2.0.
* **TCGA** slides are public (NCI GDC); the bevacizumab cohort is private and not distributed.
* Third-party dependencies keep their own licences (see `requirements-aws.txt`).

## 13. Limitations
Small cohorts (79–198 patients); site confounding in TCGA; cheap features read every patch once (no pyramid on
the private cohort's scanner files); the halting controller measures embedding redundancy, not task information;
the Decider shows a position preference among shuffled options (C > B ≈ D > A); AWS execution evidence is still
pending; the benchmark tables are UNI-era results and were not re-run with H-optimus-0 (only the demo classifier heads were retrained). Development-set tuning of prompts on the private cohort is documented in the audit.

## 14. Repository, citation, acknowledgments
Public repository: https://github.com/jiwawang/pathfinder-decider-agent · release tag `v1.0-hackathon`
(https://github.com/jiwawang/pathfinder-decider-agent/releases/tag/v1.0-hackathon).

UNI: Chen et al., *Towards a general-purpose foundation model for computational pathology*, Nat. Med. 2024.
TRIDENT (Mahmood Lab) for tissue segmentation on TCGA. Strands Agents / AWS for Strands Decider 2B.
TCGA data from the NCI Genomic Data Commons.
