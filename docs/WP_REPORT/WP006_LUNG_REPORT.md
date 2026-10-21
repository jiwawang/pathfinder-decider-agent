# WP006 — Task 1: TCGA-LUAD vs TCGA-LUSC with the frozen acquisition policy

Run `results/tcga_generalization/task1_luad_lusc/` (metadata.json, manifest_snapshot.csv, metrics.csv,
oof_predictions.csv, acquisition_stats.csv, summary.md, analysis.md/json, live_benchmark.json, figures/).
Data: `<TCGA_ROOT>/{LUAD,LUSC}/Patch_Trident_settings_2_hest/20x_224px_0px_overlap_tissue0.5/` (D001 convention,
README per job_dir). Logs of every run kept under `results/_logs/` (untracked).

## A. Git
Branch `experiment/tcga-generalization` (no merge, no push, no history rewrite); frozen baseline tag
`v2.1-audited` (`4530fe3`). Commits for this WP: WP006 brief; manifest-driven GDC downloader + Task 1/2 configs;
runner extensions (manifest-restricted slide table with GDC provenance, binary metrics, manifest snapshot,
figures); manifest-consistency test; pipeline script; analysis script; results + this report. **Core code diff vs
`v2.1-audited` in `src/acquisition/`, `decision_patterns/`, `src/evaluation/`, `src/models/`, `src/decider/`,
`configs/default.yaml`: none.** All changes are under `src/tcga/`, `scripts/tcga/`, `configs/tcga_*`, `metadata/`,
`tests/`, `docs/`, `results/`.

## B. Dataset
* Frozen manifest `metadata/tcga_lung_manifest.csv` (sha256 ed3669d4…): 100 LUAD + 100 LUSC patients, one
  lowest-DX diagnostic primary-tumour slide each, seed-0 selection. 60 LUAD were on disk; 40 LUAD + 100 LUSC
  (107.3 GB, 137 files) were downloaded with md5 verification (0 failures) in ~55 min at ~27 MB/s. Every used
  slide carries gdc_file_id, md5, GDC slide_id and patient barcode in `manifest_snapshot.csv`.
* Exclusions (frozen rule, `exclusions.csv`): 2 LUAD without MPP metadata (TCGA-05-4410, TCGA-05-5420). **Used:
  98 LUAD (label 0) + 100 LUSC (label 1) = 198 patients = 198 slides.** No duplicate patient / file id / md5.
* MPP: 0.246–0.253 µm/px (40x) on 183 slides, 0.5 µm/px (20x) on 15; windows 443–494 px or 222–224 px, all
  resampled to 224 px = 112 µm. No MPP outside 0.2–0.6. Tissue patches per slide: median ≈ 11 k (mean 15.3 k).
* Site confounding (GDC tissue_source_site name): 35 sites, 16 shared by both classes, Cramér's V = 0.62
  (χ² p = 6e-5); 26 % of slides come from single-class sites (e.g. IGC 18/8, Christiana 9/2, Prince Charles 5/10,
  Indivumed 4/10). Partial confounding cannot be excluded; see the site-held-out sensitivity below.

## C. Main results (pre-specified K = 16; patient-level StratifiedGroupKFold 5 × 5 seeds; mean ± std over seeds = split variability, not an independent-test CI)
| Method | UNI calls | ROC-AUC | 95 % CI (patient bootstrap) | Bal Acc | F1 | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Full UNI | 15,342 | 0.963 ± 0.009 | [0.938, 0.982] | 0.891 | 0.895 | 0.954 |
| Random-16 (3 draws) | 16 | 0.935 ± 0.013 | — | 0.869 | 0.874 | 0.941 |
| Spatial uniform-16 | 16 | 0.945 ± 0.009 | [0.915, 0.970] | 0.861 | 0.868 | 0.941 |
| Cheap Top-16 | 16 | 0.966 ± 0.008 | [0.944, 0.985] | 0.903 | 0.905 | 0.960 |
| Cheap Greedy-16 | 16 | 0.967 ± 0.008 | [0.947, 0.985] | 0.903 | 0.906 | 0.965 |
| Decider Fixed-16 | 16 | 0.964 ± 0.007 | [0.941, 0.983] | 0.895 | 0.897 | 0.962 |
| Decider Adaptive | 19.6 (7–32) | 0.962 ± 0.008 | [0.937, 0.982] | 0.893 | 0.896 | 0.958 |

Paired analysis vs Decider Fixed-16 (`analysis.md`): Decider − Random = +0.014 / +0.040 / +0.032 over the three
random draws (5/5 seeds, bootstrap CIs [+0.019, +0.066] and [+0.008, +0.057] exclude 0 for two draws, Wilcoxon on
per-slide log-loss p < 1e-4); Decider − Spatial = +0.019 (5/5 seeds, CI [−0.002, +0.042]); Decider − Cheap Top-K =
−0.002 (1/5 seeds, CI [−0.013, +0.010]); Decider − Cheap Greedy = −0.003 (1/5, CI [−0.012, +0.006]);
Decider − Full UNI = +0.002 (CI [−0.015, +0.019]). Confusion at K=16 (means): Decider 87 TN / 11 FP / 10 FN / 90 TP;
recalls LUAD 0.888, LUSC 0.902.

## D. Budget curves (`figures/fig1_auroc_vs_uni_calls.png`)
| K | Random | Spatial | Cheap Top-K | Cheap Greedy | Decider Fixed |
|---:|---:|---:|---:|---:|---:|
| 4 | 0.869 | 0.898 | 0.921 | 0.944 | 0.939 |
| 8 | 0.923 | 0.923 | 0.963 | 0.962 | 0.945 |
| 16 | 0.935 | 0.945 | 0.966 | 0.967 | 0.964 |
| 32 | 0.955 | 0.954 | 0.961 | 0.971 | 0.967 |

Full UNI = 0.963. Cheap-feature-guided policies (Cheap Greedy, Cheap Top-K, Decider) reach Full-UNI AUROC at 8–16
calls; random and spatial sampling need 32. Decider Fixed-8 (0.945) lags the two cheap heuristics at 8.

## E. Runtime (`live_benchmark.json`, live on-demand UNI, RTX 4090, 2 slides per class)
| per slide | LUAD (6.3 k / 4.3 k patches) | LUSC (27.4 k / 26.9 k patches) |
|---|---:|---:|
| cheap features (12 workers, 17.4 ms/patch) | 9.1 s / 6.3 s | 39.7 s / 39.1 s |
| Decider, fixed-16 (16 decisions) | 1.99 s / 1.61 s | 1.61 s / 1.62 s |
| UNI, 16 on-demand calls | 0.59 s / 0.50 s | 0.37 s / 0.36 s |
| agent loop total | 2.6 s / 2.1 s | 2.0 s / 2.0 s |
| full-slide UNI, batched (290 patches/s) | 21.6 s / 14.9 s | 94.4 s / 92.9 s |

UNI patch evaluations: 15,342 → 16 (≈ 960×). End-to-end: ≈ 12 s vs 22 s (LUAD) and ≈ 42 s vs 94 s (LUSC), i.e. ≈ 2×,
bounded by the cheap pre-pass that reads every tissue patch once. Inside the loop the Decider is 75–80 % of the time
(≈ 100 ms per routing decision, ≈ 150 ms per round when the halting question is also asked). Adaptive STOP used
10–16 calls on these four slides. Cached simulation counted exactly the logical UNI calls (assert in the runner).

## F. Cross-task interpretation
| Task | Full UNI | Random-16 | Cheap Greedy-16 | Decider-16 | Adaptive (calls) | calls to reach Full UNI |
|---|---:|---:|---:|---:|---:|---|
| OBR bevacizumab response (binary, n=79) | 0.679 | 0.680 | 0.702 | 0.728 | 0.704 (20.3) | ~8 (curve flat) |
| TCGA organ COAD/KIRC/LUAD (3-class, n=173) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 (18.6) | 4 |
| **TCGA LUAD vs LUSC (binary, n=198)** | 0.963 | 0.935 | 0.967 | 0.964 | 0.962 (19.6) | 8–16 guided, 32 random |

Task 1 is the first task where the sampling policy visibly matters: random-16 is 0.03 AUROC below the guided
policies and needs 32 calls to catch up. Guided selection by cheap features (large, diverse, tissue-rich clusters)
carries almost all of that benefit; the Decider's routing adds nothing measurable over the deterministic cheap
heuristics (differences −0.002 / −0.003, CIs include 0). Adaptive STOP budgets are slide-specific (7–32, median 17,
81 % model-initiated) and track embedding redundancy (recent gain 61 % of initial at STOP vs 88 % at CONTINUE,
corr(P(STOP), ratio) = −0.82), but are unrelated to tissue amount (corr +0.05) and to whether the slide ends up
correctly classified (19.8 vs 17.9 calls) — information-gain saturation is not classification-information saturation.
Site-held-out sensitivity (folds grouped by hospital): Full UNI 0.939 ± 0.008, Decider-16 0.946 ± 0.011, Adaptive
0.946 ± 0.010 — ≈ 0.02 lower than patient-grouped, so part of the signal is site-correlated but the ranking holds.

## G. Failure cases
* Decider Fixed-8 (0.945) < Cheap Top-8 (0.963) and Cheap Greedy-8 (0.962); Decider Fixed-4 (0.939) < Cheap Greedy-4
  (0.944). At small budgets the model's routing is slightly worse than the deterministic exploration score.
* Decider ≤ cheap heuristics at every K in balanced accuracy (0.895 vs 0.903 at 16).
* Routing position bias persists (shuffled options): strategy mix largest cluster 58 %, spatial far 19 %, under-covered
  18 %, cheap novelty 5 % — identical to Task 0 and OBR.
* 38 of 198 adaptive episodes hit the 32-call ceiling without the model stopping.
* Two LUAD slides lost to missing MPP metadata; no replacement (rule).

## H. Final conclusion
> Does Decider provide a measurable benefit over simpler acquisition strategies on within-organ pathology classification?

**NO over cheap-feature heuristics; YES over random and spatial sampling.** Decider-16 beats random-16 by
0.014–0.040 AUROC (paired CIs exclude zero in 2 of 3 draws) and matches Full UNI with ~960× fewer UNI evaluations,
but it is statistically indistinguishable from Cheap Top-16 / Cheap Greedy-16 (Δ ≈ −0.002, CIs within ±0.013),
which use the same cheap features and no model. Decision-guided acquisition remains computationally efficient and
predictively useful when the task becomes non-trivial; its measurable contribution is adaptive, traceable budget
control, not better patch selection than a well-designed cheap heuristic.

Task 2 (KIRC vs KIRP) was not started in this round: Task 1 is complete and the hackathon deliverables (AWS +
Bedrock integration, article, public repo, video; due 2026-10-23) take precedence per WP006 §22. Frozen kidney
manifest, configs and pipeline are ready (`configs/tcga_task2.yaml`, 135 GB download at today's throughput ≈ 1.5 h,
processing ≈ 3 h).
