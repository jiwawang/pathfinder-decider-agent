# Analysis — task1_luad_lusc (primary budget K=16)

## 1. Paired comparisons vs decider_fixed_16

| Comparator | ΔAUROC (mean over 5 seeds, Decider − comparator) | seeds Decider better | paired bootstrap 95% CI of Δ | Wilcoxon p (per-slide log-loss) |
|---|---:|---:|---:|---:|
| random_16_s0 | +0.014 | 5/5 | [-0.010, +0.039] | 0.0553 |
| random_16_s1 | +0.040 | 5/5 | [+0.019, +0.066] | 2.46e-05 |
| random_16_s2 | +0.032 | 5/5 | [+0.008, +0.057] | 4.28e-05 |
| uniform_16 | +0.019 | 5/5 | [-0.002, +0.042] | 0.00707 |
| cheap_topk_16 | -0.002 | 1/5 | [-0.013, +0.010] | 0.376 |
| cheap_greedy_16 | -0.003 | 1/5 | [-0.012, +0.006] | 0.44 |
| decider_adaptive | +0.002 | 5/5 | [-0.005, +0.009] | 0.0425 |
| full_uni | +0.002 | 4/5 | [-0.015, +0.019] | 0.934 |

## 2. Binary detail at K=16 (means over seeds)

| Method | recall class0 | recall class1 | PR-AUC | TN / FP / FN / TP |
|---|---:|---:|---:|---|
| full_uni | 0.857 | 0.924 | 0.954 | 84 / 14 / 8 / 92 |
| random_16_s0 | 0.867 | 0.916 | 0.941 | 85 / 13 / 8 / 92 |
| uniform_16 | 0.822 | 0.900 | 0.941 | 81 / 17 / 10 / 90 |
| cheap_topk_16 | 0.888 | 0.918 | 0.960 | 87 / 11 / 8 / 92 |
| cheap_greedy_16 | 0.882 | 0.924 | 0.965 | 86 / 12 / 8 / 92 |
| decider_fixed_16 | 0.888 | 0.902 | 0.962 | 87 / 11 / 10 / 90 |
| decider_adaptive | 0.876 | 0.910 | 0.958 | 86 / 12 / 9 / 91 |

## 3. Adaptive STOP

* UNI calls: mean 19.56, median 17.0, std 8.51, min 7.0, max 32.0; by class {'mean': {'LUAD': 19.35, 'LUSC': 19.77}, 'median': {'LUAD': 17.5, 'LUSC': 17.0}, 'std': {'LUAD': 8.4, 'LUSC': 8.65}}
* STOP reasons: {'model STOP': 160, 'max budget': 38}; corr(calls, tissue patches) = +0.05
* recent information gain at STOP 0.368 vs at CONTINUE 0.504; recent gain as % of initial: 61% vs 88%; corr(P(STOP), %initial) = -0.82
* mean calls on slides classified correctly (≥3/5 seeds) vs incorrectly: 19.8 vs 17.9

## 4. Tissue source site

* 35 sites, 16 shared by all classes, Cramér's V = 0.62 (χ² p = 6.3e-05); 26% of slides come from single-class sites.
* Site-held-out sensitivity (StratifiedGroupKFold with groups = TSS, 5 folds × 5 seeds, same classifier) — a site never appears on both sides:

| Method | patient-grouped AUROC | site-grouped AUROC |
|---|---:|---:|
| full_uni | 0.963 | 0.939 ± 0.008 |
| decider_fixed_16 | 0.964 | 0.946 ± 0.011 |
| decider_adaptive | 0.962 | 0.946 ± 0.010 |

Top sites: International Genomics Consortium {0: np.int64(18), 1: np.int64(8)}; Asterand {0: np.int64(9), 1: np.int64(8)}; Prince Charles Hospital {0: np.int64(5), 1: np.int64(10)}; Indivumed {0: np.int64(4), 1: np.int64(10)}; Christiana Healthcare {0: np.int64(9), 1: np.int64(2)}; Johns Hopkins {0: np.int64(7), 1: np.int64(2)}
