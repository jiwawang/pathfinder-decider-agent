# PathFinder ablation tables

## Run `decider_v2` (5 seed(s): [0, 1, 2, 3, 4]; 5-fold patient-level CV, L2 logistic regression)

| Method | AUROC | Balanced Acc. | F1 | Accuracy | Mean patches used |
|---|---|---|---|---|---|
| A. All-patch mean | 0.679 ± 0.025 | 0.626 | 0.659 | 0.628 | 25,033.3 |
| B. Random-K | 0.649 ± 0.051 | 0.601 | 0.653 | 0.608 | 9.2 |
| C. Vote Top-K | 0.651 ± 0.043 | 0.572 | 0.630 | 0.580 | 9.2 |
| C'. KMeans Top-K (cluster size) | 0.691 ± 0.021 | 0.642 | 0.669 | 0.643 | 9.2 |
| C''. All 32 candidates mean | 0.700 ± 0.029 | 0.638 | 0.662 | 0.638 | 32.0 |
| Proposed: Decider-K | 0.684 ± 0.047 | 0.644 | 0.674 | 0.646 | 9.1 |

Decider: 51,219 decisions in 1975 episodes; latency mean 65.6 ms / median 65.3 ms / p95 66.8 ms; mean confidence 0.578; low-confidence rate 6.8%; SELECT/SKIP/STOP = 35.1% / 63.2% / 1.6%; selected 9.1 of 32 candidates (0.040% of all tissue patches); examined 25.9; stopped by the model in 42.8% of episodes.


## Run `threshold_v2` (5 seed(s): [0, 1, 2, 3, 4]; 5-fold patient-level CV, L2 logistic regression)

| Method | AUROC | Balanced Acc. | F1 | Accuracy | Mean patches used |
|---|---|---|---|---|---|
| A. All-patch mean | 0.679 ± 0.025 | 0.626 | 0.659 | 0.628 | 25,033.3 |
| B. Random-K | 0.666 ± 0.038 | 0.607 | 0.658 | 0.614 | 16.0 |
| C. Vote Top-K | 0.699 ± 0.019 | 0.607 | 0.643 | 0.610 | 16.0 |
| C'. KMeans Top-K (cluster size) | 0.702 ± 0.019 | 0.651 | 0.691 | 0.656 | 16.0 |
| C''. All 32 candidates mean | 0.700 ± 0.029 | 0.638 | 0.662 | 0.638 | 32.0 |
| Proposed: Decider-K | 0.702 ± 0.020 | 0.627 | 0.663 | 0.630 | 15.9 |

Decider: 34,260 decisions in 1975 episodes; latency mean 0.1 ms / median 0.0 ms / p95 0.0 ms; mean confidence 0.859; low-confidence rate 0.0%; SELECT/SKIP/STOP = 91.5% / 2.9% / 5.7%; selected 15.9 of 32 candidates (0.069% of all tissue patches); examined 17.4; stopped by the model in 97.7% of episodes.


## Run `decider_v2_random_order` (1 seed(s): [0]; 5-fold patient-level CV, L2 logistic regression)

| Method | AUROC | Balanced Acc. | F1 | Accuracy | Mean patches used |
|---|---|---|---|---|---|
| Proposed: Decider-K | 0.619 | 0.561 | 0.622 | 0.570 | 9.0 |

Decider: 11,758 decisions in 395 episodes; latency mean 66.8 ms / median 65.3 ms / p95 79.5 ms; mean confidence 0.545; low-confidence rate 8.0%; SELECT/SKIP/STOP = 30.3% / 68.3% / 1.3%; selected 9.0 of 32 candidates (0.039% of all tissue patches); examined 30.1; stopped by the model in 39.2% of episodes.
