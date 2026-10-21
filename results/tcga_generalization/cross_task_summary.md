# Cross-task summary (WP005)

| Task | slides | Full UNI | best 4-call policy | Decider Fixed-16 | Decider Adaptive (calls) | needs ≥ 16 calls? |
|---|---:|---:|---:|---:|---:|---|
| OBR bevacizumab response (binary, V2.1) | 79 | 0.679 ± 0.025 | — (K=4: Decider 0.725, Cheap greedy 0.722) | 0.728 ± 0.019 | 0.704 ± 0.032 (20.3) | no: flat from K≈8 |
| TCGA COAD / KIRC / LUAD organ (3-class, Task 0) | 173 | 1.000 | 1.000 (Cheap greedy / Decider, Bal Acc 0.994) | 1.000 | 1.000 (18.6) | no: saturates at 4 |
| TCGA LUAD vs LUSC (binary, Task 1) | 198 | 0.963 ± 0.009 | Cheap greedy-4 0.944 (Decider-4 0.939) | 0.964 ± 0.007 (Cheap greedy-16 0.967, Random-16 0.935) | 0.962 ± 0.008 (19.6) | guided policies 8–16; random 32 |
| TCGA KIRC vs KIRP (binary) | frozen manifest, 200 | not run (download) | | | | |

Reading: the acquisition mechanism transfers unchanged; on the within-organ lung task guided selection beats random sampling by 0.03 AUROC but the Decider does not beat the cheap heuristics (docs/WP_REPORT/WP006_LUNG_REPORT.md); the required evidence budget is task-dependent and the
label-agnostic adaptive STOP does not track it. Details: docs/WP_REPORT/WP005_TCGA_GENERALIZATION.md.
