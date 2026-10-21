# WP005 — Cross-task generalization of the frozen acquisition policy: Task 0 (TCGA COAD / KIRC / LUAD)

Run `results/tcga_generalization/task0_3class/` (summary, metrics, OOF predictions, acquisition stats, live
benchmark, figures); data layers under `<TCGA_ROOT>/<type>/Patch_Trident_settings_2_hest/` (D001 convention, README
in each job_dir). Lung (LUAD vs LUSC) and kidney (KIRC vs KIRP) remain frozen-manifest tasks for a later round;
nothing was downloaded in this round.

## A. Git & reproducibility
Branch `experiment/tcga-generalization` from tag `v2.1-audited` (`4530fe3`). Commits: WP005 brief, manifest generator +
frozen lung/kidney manifests, data survey, plan (D001-based, settings_2_hest, MPP rule), TCGA reader + slide-table
builder, cheap features / UNI cache / live store / multiclass evaluation / Task 0 runner, tests (22 passing), live
benchmark, job_dir READMEs, `--reverse` for the cache writer, results + this report. `metadata.json` in the result dir
records git commit, config and slide-table sha256, seeds, labels and the Decider model id. Core algorithm files
(`src/acquisition/*`, `decision_patterns/*`, prompts, classifier, CV splits) are untouched — the diff against
`v2.1-audited` touches only `src/tcga/`, `scripts/tcga/`, `configs/tcga_*.yaml`, `metadata/tcga_*`, tests and docs.

## B. Dataset
| | COAD (label 0) | KIRC (1) | LUAD (2) |
|---|---:|---:|---:|
| local diagnostic slides (one per patient, seed-0 selection) | 60 | 60 | 60 |
| excluded | 6 (no MPP in metadata: TCGA-AA-3489/3496/3509/3660/3663/3697) | 0 | 1 (TCGA-05-5420, no MPP) |
| used | **54** | **60** | **59** |
| tissue patches per slide (median, 224 px @ 0.5 µm/px, tissue ≥ 0.5) | 9.3 k | 20.4 k | 18.1 k |

173 slides = 173 patients. Slide type: Diagnostic Slide (DX1) only, Primary Tumor. Tissue: TRIDENT hest segmenter,
`--mag 20 --patch_size 224 --overlap 0 --min_tissue_proportion 0.5`. Reading: MPP only (skip if missing or x/y > 1 %
apart), level-0 window `round(224·0.5/mpp)` (443–494 px on these 40x scans; two 20x slides read 224), LANCZOS → 224,
so every patch covers 112 µm as on the OBR cohort. No MPP value fell outside 0.2–0.6 µm/px. Class is fully
confounded with tissue source site (one project = its own hospitals) — unavoidable for an organ task and not analysed.

## C. Main results (patient-level StratifiedGroupKFold 5 × 5 seeds, multinomial L2 logistic regression, mean of acquired UNI embeddings)
| Method | UNI calls / slide | macro AUROC | Bal Acc | macro F1 |
|---|---:|---:|---:|---:|
| Full UNI (all patches) | 15,824 | 1.000 ± 0.000 | 0.998 | 0.998 |
| Random-4 / 8 / 16 / 32 | 4–32 | 0.999 / 1.000 / 1.000 / 1.000 | 0.971 / 0.988 / 0.992 / 0.994 | 0.971 / 0.988 / 0.991 / 0.994 |
| Spatial uniform-4 / 8 / 16 / 32 | 4–32 | 1.000 (all) | 0.991 / 0.987 / 0.995 / 0.998 | 0.991 / 0.986 / 0.995 / 0.998 |
| Cheap Top-K 4 / 8 / 16 / 32 | 4–32 | 0.998 / 0.999 / 1.000 / 1.000 | 0.964 / 0.974 / 0.987 / 0.998 | 0.963 / 0.973 / 0.986 / 0.998 |
| Cheap Greedy 4 / 8 / 16 / 32 | 4–32 | 1.000 (all) | 0.994 / 0.998 / 0.997 / 0.997 | 0.994 / 0.998 / 0.996 / 0.996 |
| Decider Fixed-4 / 8 / 16 / 32 | 4–32 | 1.000 (all) | 0.994 / 0.997 / 0.992 / 0.997 | 0.994 / 0.996 / 0.992 / 0.996 |
| **Decider Adaptive STOP** | **18.6** (6–32) | **1.000 ± 0.000** | **0.994** | **0.994** |

Per-class one-vs-rest AUROCs are ≥ 0.997 for every method at K ≥ 4; patient-bootstrap 95 % CIs are [1.000, 1.000] for
almost every row (Cheap Top-4: [0.996, 1.000]). The task saturates at 4–8 UNI calls for every policy.

## D. Compute efficiency (RTX 4090; live benchmark on one slide per type, `live_benchmark.json`, `figures/task0_compute_comparison.png`)
| | LUAD (6.3 k patches) | COAD (40.2 k) | KIRC (6.7 k) |
|---|---:|---:|---:|
| cheap features, 12 workers (17.4 ms/patch single-worker, measured uncontended) | 9.1 s | 58.3 s | 9.7 s |
| Decider, fixed-16: 16 decisions | 2.14 s | 1.74 s | 1.77 s |
| on-demand UNI, 16 calls (live crop + encode) | 0.71 s | 0.26 s | 0.48 s |
| **PathFinder end-to-end** | **11.9 s** | **60.3 s** | **11.9 s** |
| full-slide UNI, batched 290 patches/s | 21.6 s | 138.7 s | 23.0 s |

UNI patch evaluations drop ~1,000× (15.8 k → 16); end-to-end wall-clock drops only ~2× because the cheap features
still read every tissue patch once (76–97 % of the loop). Inside the agent loop the Decider is 70–85 % of the time
(130–150 ms per round with the halting question). Adaptive STOP: 12 / 32 / 11 calls on these three slides.

## E. Cross-task generalization
* The frozen policy runs unchanged on a different organ set, scanner family and tissue rule; nothing broke.
* **Routing transfers but does not matter here**: Decider-K ≈ Cheap Greedy-K ≈ Spatial uniform-K ≈ Random-K at every K.
  The only visible differences are at K = 4, where the two heuristic-guided policies (Cheap Greedy, Decider) reach
  balanced accuracy 0.994 vs 0.971 (Random) and 0.964 (Cheap Top-4) — i.e. picking large, diverse clusters avoids the
  few slides where 4 random or 4 largest-cluster patches miss tumour. Strategy mix is identical across cancer types
  (largest cluster 57–59 %, spatial far 17–19 %, under-covered 18–19 %, cheap novelty 5–6 %).
* **The adaptive STOP does not adapt to task difficulty.** It spends 18.6 calls on average (COAD 18.7, KIRC 17.6, LUAD
  19.4; 86 % self-stopped) on a task that is solved with 4. Being label-agnostic by design, it measures evidence
  redundancy in UNI space, not how much evidence the classifier needs, so it cannot know the task is easy.
* Which tasks benefit: none of the three organ classes needs more than 4–8 patches. Which tasks fail: none.

## F. Statistical reliability
Seed std ≤ 0.001 for AUROC; patient-bootstrap CIs are degenerate at 1.000 because the classes are separable at the
slide level. Paired per-slide comparisons are uninformative at this ceiling. The only reliable ordering is at K = 4
(balanced accuracy: Cheap Greedy / Decider 0.994 > Spatial uniform 0.991 > Random 0.971 > Cheap Top-K 0.964).

## G. Failure cases
* Cheap Top-4 misclassifies the same few slides repeatedly (TCGA-CA-6719, TCGA-B0-4819, TCGA-DV-5576, TCGA-55-7284,
  TCGA-BP-5008 in ≥ 4 of 5 seeds): the four largest cheap clusters of those slides are non-diagnostic tissue.
* Position bias of the Decider routing persists after shuffling: C 31.5 %, B 27.0 %, D 25.5 %, A 16.1 %.
* Over-spending by the adaptive STOP on an easy task (above).
* Data: 7 of 180 slides lack MPP metadata and are excluded by rule.

## H. Conclusions
1. Cross-task transfer of the acquisition *mechanism* holds: the same cheap features, pools, shortlist, Decider prompts
   and classifier protocol run on TCGA without any tuning.
2. The *amount* of expensive evidence a task needs is task-dependent and, for organ-of-origin, tiny: 4 UNI patches
   reach macro AUROC ≥ 0.998. The adaptive controller, which watches redundancy rather than task difficulty, does not
   exploit this and spends ~19 calls. This is the expected counterpart of the OBR finding and the concrete evidence for
   the brief's closing hypothesis: adaptive acquisition must balance information coverage against task need, not just
   against embedding redundancy.
3. End-to-end runtime reduction is bounded by the cheap pre-pass (reading every tissue patch once): ~2× here, not the
   ~1,000× reduction in UNI calls. Reading cheap features from a pyramid level (these slides have one) is the obvious next lever.
4. Task 0 says nothing about the within-organ questions (LUAD vs LUSC, KIRC vs KIRP); those need the frozen lung/kidney
   manifests and the download the survey quantified.

## I. Deviations from the frozen protocol
* Three-class task instead of binary (registered in `WP005_PLAN.md` §7): same pipeline, multinomial LR, macro OvR AUROC,
  balanced accuracy, macro F1. Label map COAD=0, KIRC=1, LUAD=2.
* Tissue rule: TRIDENT hest + min tissue 0.5 (OBR used a saturation mask at ≥ 0.05); physical scale 112 µm via the
  MPP-matched window (OBR's 0.5 µm/px is an external record, not a measurement).
* Random-K draws: 3 per K; their bootstrap CI column is left blank (per-draw CIs not pooled).
* Cheap-feature timing in the live benchmark is reported from an uncontended single-worker measurement (17.4 ms/patch)
  scaled to 12 workers; the in-pipeline timings were inflated 3–5× by running TRIDENT, 12 extraction workers and two
  UNI loaders at once.
