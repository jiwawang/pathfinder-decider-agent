# WP003 report — V2.2 parallel regional search, Phase 1 (fixed budget 16)

Branch `experiment/parallel-regional-search`; results in `results/v2_2_parallel/` (`fixed16` = Strands Decider
routing, `smoke_heuristic` = model-free routing, seed 0 only; `wallclock.json`; `partition_report.csv`).
Every result directory carries `metadata.json` with the git commit and branch.

## A. Git
* baseline commit `e6556f7` on `main`, tag `v2.1-adaptive-baseline` (16 tests passing)
* experiment branch `experiment/parallel-regional-search`; commits: four-region partition + regional pools,
  batched UNI acquisition + global-evidence-aware shortlists, parallel fixed-budget loop, tests, experiment
  scripts + summaries, this report. `git status` clean at the end. No `v2.2-parallel-fixed-budget` tag: the
  Phase 1 failure condition (section 34) applies.

## B. Partition (79 slides, median-x then median-y)
Region shares 25.3 / 25.0 / 25.0 / 24.6 %, max/min ratio within a slide 1.03 on average, smallest region
2,256 patches. No empty region; the KMeans fallback was never needed. Regional pools: 4 x 32 = 128
candidates per slide, standardised with slide-level statistics.

## C. Main results (patient-level 5-fold CV x 5 seeds, same classifier, same Decider, 16 UNI calls each)

| Method | UNI Calls | UNI Batches | AUROC | Bal Acc | F1 | Spatial Coverage | Semantic Diversity | Loop time (live) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Serial Decider Fixed-16 | 16 | 16 | 0.728 ± 0.019 | 0.676 | 0.716 | 24.4% | 0.659 | 1.79 s |
| Parallel 4 regions x 4 rounds | 16 | 4 | 0.616 ± 0.031 | 0.579 | 0.629 | 26.3% | 0.680 | 1.87 s |

Per-seed AUROC: serial 0.699–0.746, parallel 0.571–0.651 (every seed worse). Paired per slide: spatial
coverage +0.019 (parallel higher on 59% of slides), semantic diversity +0.021 (62%). Regional balance:
serial 4.2 / 3.8 / 4.2 / 4.2 ± 1.5 (unconstrained, already nearly even); parallel exactly 4 / 4 / 4 / 4.
Model-free routing shows the same gap (seed 0: serial 0.746 vs parallel 0.662), so it is the topology, not
the Decider. Correlation of per-slide OOF probabilities between the two: 0.80.

## D. Runtime (live UNI, RTX 4090, 2 slides x 2 repeats)
| | Decider calls | UNI calls | UNI batches | Decider time | UNI time | total loop |
|---|---:|---:|---:|---:|---:|---:|
| Serial | 16 | 16 | 16 (batch 1) | 1.61 s | 0.156 s | 1.79 s |
| Parallel | 16 | 16 | 4 (batch 4, 21 ms each) | 1.47 s | 0.102 s | 1.87 s |

UNI batching alone is 1.53x faster, but UNI is 9% of the loop; the Decider (one state per call,
sequential for the four regions — Strands Decider has no multi-state batch API) is 85–90% of it.
Total wall-clock 0.96x, i.e. no improvement.

## E. Observations
* Spatial coverage and semantic diversity improve only marginally (+2 points, +0.02).
* AUROC drops clearly (−0.11) at the same budget; balanced accuracy and F1 drop too.
* Wall-clock does not improve; the sequential Decider dominates and UNI batching has nothing to save.
* Recommendation: **do not continue to Phase 2 (regional adaptive STOP)**. Section 34's failure condition
  is met. Branch and results are kept; nothing is merged to `main`.

## F. Problems / notes
* Likely cause of the AUROC drop: forcing 4 acquisitions per region replaces the serial policy's freedom
  to spend calls on the globally largest tissue clusters (largest-cluster strategy 48% in parallel vs 62%
  in serial) with 32-cluster regional pools that are coarser than the 128-cluster global pool; on this
  cohort the globally representative patches carry the signal. Not investigated further (section 42).
* Position preference after shuffling: C 30.8%, B 26.3%, A 21.4%, D 21.5% — milder than in V2.1 but present.
* 80 of 1,264 routing decisions fell back to the heuristic's best option on low confidence.
* Implementation deviation: the serial baseline is reconstructed from the V2.1 `decider_fixed_16` traces
  rather than re-run (identical Decider, pools and classifier; saves GPU time and keeps the comparison on
  the exact same acquisitions). The serial live loop in the benchmark is a fresh run.
