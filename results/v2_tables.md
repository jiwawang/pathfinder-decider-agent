# PathFinder V2 — sequential active patch acquisition

Patient-level 5-fold CV x 5 seeds, L2 logistic regression on the mean of the acquired UNI embeddings. Run `decider_v3_active` (Strands Decider) and `heuristic_v3` (model-free router).

## Main table (WP001 section 20)

| Method | UNI calls / slide | AUROC | Balanced Acc. | F1 |
|---|---|---|---|---|
| Full UNI (all-patch mean) | 25,033.3 | 0.679 ± 0.025 | 0.626 | 0.659 |
| Random-K = 4 | 4.0 | 0.607 ± 0.056 | 0.571 | 0.614 |
| Random-K = 8 | 8.0 | 0.625 ± 0.060 | 0.591 | 0.642 |
| Random-K = 16 | 16.0 | 0.680 ± 0.050 | 0.637 | 0.685 |
| Random-K = 32 | 32.0 | 0.642 ± 0.015 | 0.590 | 0.635 |
| Random-K = 64 | 64.0 | 0.670 ± 0.029 | 0.616 | 0.661 |
| Spatial uniform-K = 4 | 4.0 | 0.738 ± 0.042 | 0.663 | 0.704 |
| Spatial uniform-K = 8 | 8.0 | 0.673 ± 0.022 | 0.637 | 0.681 |
| Spatial uniform-K = 16 | 16.0 | 0.675 ± 0.029 | 0.627 | 0.665 |
| Spatial uniform-K = 32 | 32.0 | 0.689 ± 0.024 | 0.624 | 0.653 |
| Spatial uniform-K = 64 | 64.0 | 0.669 ± 0.019 | 0.619 | 0.647 |
| Cheap Top-K (largest cheap clusters) = 4 | 4.0 | 0.635 ± 0.010 | 0.618 | 0.662 |
| Cheap Top-K (largest cheap clusters) = 8 | 8.0 | 0.662 ± 0.035 | 0.612 | 0.659 |
| Cheap Top-K (largest cheap clusters) = 16 | 16.0 | 0.723 ± 0.017 | 0.669 | 0.714 |
| Cheap Top-K (largest cheap clusters) = 32 | 32.0 | 0.693 ± 0.021 | 0.659 | 0.706 |
| Cheap Top-K (largest cheap clusters) = 64 | 64.0 | 0.676 ± 0.019 | 0.643 | 0.677 |
| Cheap greedy-K (unified exploration score) = 4 | 4.0 | 0.722 ± 0.033 | 0.621 | 0.671 |
| Cheap greedy-K (unified exploration score) = 8 | 8.0 | 0.692 ± 0.019 | 0.622 | 0.668 |
| Cheap greedy-K (unified exploration score) = 16 | 16.0 | 0.702 ± 0.022 | 0.635 | 0.678 |
| Cheap greedy-K (unified exploration score) = 32 | 32.0 | 0.695 ± 0.024 | 0.624 | 0.673 |
| Cheap greedy-K (unified exploration score) = 64 | 64.0 | 0.675 ± 0.030 | 0.629 | 0.668 |
| Heuristic router, fixed budget K (no model) = 4 | 4.0 | 0.703 ± 0.036 | 0.636 | 0.685 |
| Heuristic router, fixed budget K (no model) = 8 | 8.0 | 0.738 ± 0.031 | 0.664 | 0.694 |
| Heuristic router, fixed budget K (no model) = 16 | 16.0 | 0.712 ± 0.021 | 0.640 | 0.683 |
| Heuristic router, fixed budget K (no model) = 32 | 32.0 | 0.721 ± 0.021 | 0.654 | 0.701 |
| Heuristic router, fixed budget K (no model) = 64 | 64.0 | 0.687 ± 0.017 | 0.622 | 0.660 |
| Heuristic router, adaptive STOP (no model) | 16.4 | 0.713 ± 0.020 | 0.643 | 0.686 |
| Decider, fixed budget K = 4 | 4.0 | 0.653 ± 0.032 | 0.608 | 0.660 |
| Decider, fixed budget K = 8 | 8.0 | 0.705 ± 0.025 | 0.632 | 0.678 |
| Decider, fixed budget K = 16 | 16.0 | 0.709 ± 0.014 | 0.635 | 0.679 |
| Decider, fixed budget K = 32 | 32.0 | 0.701 ± 0.025 | 0.644 | 0.683 |
| Decider, fixed budget K = 64 | 64.0 | 0.694 ± 0.020 | 0.631 | 0.680 |
| Decider, adaptive STOP | 16.4 | 0.723 ± 0.017 | 0.647 | 0.683 |

Failure mode kept for the record (`decider_v3_joint_stop`): with STOP offered as a fifth option inside the routing question, the Decider never stopped (P(STOP) <= 0.07 on every round) and always spent the maximum budget: 32 UNI calls, AUROC 0.698 ± 0.021. The separate CONTINUE/STOP question fixed this (see decision_patterns/ADAPTIVE_ROUTER.md).

## Decider acquisition statistics

* UNI calls per slide: mean 16.4 (min 16, max 32); full-slide pass = 25,033 calls, i.e. **1,524x fewer UNI patch evaluations**
* 1,375 decisions over 79 slides; latency mean 117.5 ms / median 139.5 ms / p95 141.9 ms; mean confidence 0.280; low-confidence fallback rate 13.0%
* agent-initiated STOP in 97.5% of slides; 16.4 cheap clusters covered per slide

## Measured compute (RTX 4090, `results/uni_benchmark.json`)

* UNI per patch: 7.7 ms encode (batch 1, median), 13.4 ms end-to-end incl. patch read; batched throughput 293 patches/s (3.41 ms/patch); peak GPU memory 1.9 GB; live vs cached embedding cosine 1.000
* Full-slide UNI pass: 25,033 patches x 3.41 ms = **85 s GPU time per slide** (batched)
* Decider agent: 16.4 UNI calls x 13.4 ms + 17.4 decisions x 117.5 ms = **2.27 s per slide** (Decider overhead 2.04 s, 90% of the agent's time)
* Net GPU reduction: 38x. Caveat: the cheap features still read every tissue patch once (~2-3 ms/patch of CPU/IO, ~50-80 s per slide here) because these SVS files have no low-magnification pyramid level; with a pyramid that cost drops by the downsample factor squared.
