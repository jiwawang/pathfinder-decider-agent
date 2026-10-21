# WP004 — Code leakage & training-protocol audit (read-only)

Audited on 2026-10-17 against `main` @ `e1d7eb9` (code identical to tag `v2.1-adaptive-baseline`, `e6556f7`,
except docs). No code was modified. Evidence = file:line in this commit plus the result files named below.

# Executive Verdict

**PASS WITH CAVEATS.** The split → fit → predict → metric chain is patient-isolated and every learned
component is fit on the training fold only; the acquisition policy never sees a label or an unacquired UNI
embedding. The caveats are model-selection / reporting issues (development-set tuning and provenance), not
train/validation leakage.

## 1. Dataset split

`src/evaluation/splits.py:14-21` `make_folds(meta, n_folds, seed)`: `StratifiedGroupKFold(n_splits=5,
shuffle=True, random_state=seed)` with `y = meta.label`, `groups = meta.patient_id`; an assertion fails if a
patient appears on both sides. Called from `scripts/run_adaptive_cv.py:106` (outer) and `:122` (inner, on the
training rows only). Split unit = patient (== slide here). Split happens after acquisition, but acquisition is
label-free and slide-local (section 4), so nothing fitted across slides precedes the split.

Actual data (`metadata/slides.csv`, read-only check): 79 rows, 79 unique slide_id / case_id / patient_id /
wsi_path / features_npy / h5_path; `patient_id == case_id` for every row; max 1 slide per patient; no duplicate
WSI byte size or inode; no duplicate feature-file content (md5 of first 64 rows); `n_patches` equals the
feature-file row count for all 79. Labels: Effective→1 (44), Invalid→0 (35), crosstab exact. The CV reads
exactly these 79 slides (`oof_predictions.csv`: 79 rows per method per seed, one prediction per slide).

## 2. Patient isolation

Seed 0 folds from `results/decider_v3_adaptive/oof_predictions.csv` (not regenerated): sizes 16/16/16/16/15,
positives 9/9/9/9/8, union = all 79 exactly once, train∩validation patients = ∅ in every fold; fold
assignment identical across all 11 methods within a seed; seeds 0 and 1 agree on only 22% of assignments
(the seed really changes the split). Fold-0 validation patients: <private-slide>, <private-slide>, <private-slide>, <private-slide>, <private-slide>,
<private-slide>, <private-slide>, <private-slide>, <private-slide>, <private-slide>, <private-slide>, <private-slide>, <private-slide>, <private-slide>, <private-slide>, <private-slide>.

## 3. Fit boundary

| Component | Where | Fit scope | Validation involved? | Status |
|---|---|---|---|---|
| Cheap-feature standardisation | `src/acquisition/candidate_pool.py:12-15,23` | within one slide (its own patches) | no cross-slide statistic | PASS |
| Cheap KMeans (K=128 pool) | `candidate_pool.py:24-26`, run per slide by `scripts/build_candidate_pool.py:43` | one slide, no label | no | PASS |
| Information gain / coverage stats | `src/acquisition/adaptive_agent.py` `saturation_state` | the slide's own acquired embeddings | no | PASS |
| UNI embedding L2 norm | `src/acquisition/lazy_uni_store.py:93-95` | per vector | no | PASS |
| StandardScaler + LogisticRegression | `src/models/slide_classifier.py:17,23-24` inside one `Pipeline`; `src/evaluation/cross_validation.py:33-36` `clf.fit(X_tr, y_tr)`, `predict_proba(X_te)` | training fold | X_val / y_val never passed to fit | PASS |
| PCA | `slide_classifier.py:19-20` | disabled (`pca_components: 0` in `config_used.yaml`) | — | PASS |
| Saturation threshold (threshold_adaptive) | `scripts/run_adaptive_cv.py:115-130` | inner 4-fold CV on `meta.iloc[tr]` only; chosen threshold then fit on train, predict val | outer validation labels never used | PASS |
| Full-UNI mean | `src/candidates/clustering.py:57` `X.mean(0)` per slide | one slide | no | PASS |
| V1 KNN vote bank (`src/voting/knn_vote.py:53`) | training fold, leave-one-slide-out | not used by any V2 / V2.1 result | n/a |

No `fit(` call in the V2.1 path touches more than one slide except the classifier pipeline, and that is fit on
the training fold inside every outer (and inner) fold.

## 4. Patch acquisition

Validation slide path: `artifacts/cheap_features/{slide}.npz` (colour/texture/position only) →
`candidate_pool_K128` (KMeans on cheap features, per slide) → `ShortlistBuilder.build` (cheap stats, cluster
sizes, positions) → `AdaptiveRouterV21.decide` → `LazyUNIFeatureStore.acquire` → embedding. Decider state
fields actually logged (`results/decider_v3_adaptive/traces/decider_fixed_16/*.json`): calls used / max,
remaining candidates, pairwise UNI similarity/distance of acquired patches, information-gain list and window
statistics, new cheap clusters, cheap-cluster and spatial coverage; candidate fields: cheap novelty, cluster
coverage %, distance to nearest acquired, tissue fraction, entropy, representativeness, exploration score,
why-shortlisted. A scan of all states and option texts finds none of: label, responder, effective, invalid,
class_, vote, pred, target, prototype. Candidate ordering (`shortlist.py`) uses cluster fraction, cheap
novelty, spatial distance, neighbourhood coverage and tissue×entropy only. The `label` key in the candidate
npz files is loaded into memory by `load_candidates` but never read by the V2.1 scripts (grep: no access).

Lazy store: the only readers of `features_npy` are `lazy_uni_store.py:91,99` (memmap, one row copied per
`acquire`); `run_adaptive_cv.py:56` constructs the store and every embedding reaches a slide representation
through `acquire` (adaptive episodes) or `acquire_indices` (cheap fixed budgets). `np.load` of the V1
candidates file at `run_adaptive_cv.py:79` reads only `all_patch_mean` / `n_patches` for the Full-UNI baseline.
Assertion `store.total_calls == Σ uni_calls` is checked at `run_sequential_cv.py:102`.

## 5. Classifier

`LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced")`, sklearn defaults otherwise (lbfgs, L2,
no random_state — deterministic for lbfgs); preceded by `StandardScaler` in the same pipeline. Per outer fold:
`fit(X[tr], y[tr])` then `predict_proba(X[te])[:, 1]`; the fitted model is discarded afterwards. Never refit on
validation. Never tuned: `C` and the pipeline were fixed in `configs/default.yaml` from the first run and are
identical in every `config_used.yaml`.

## 6. OOF evaluation

`cross_validation.py:111-122` `summarise`: for each (method, seed) the 79 out-of-fold probabilities (one per
patient, from its own validation fold) are concatenated and `roc_auc_score` is computed once; the reported
value is the mean ± sample std over the 5 seeds. Training predictions are never written to
`oof_predictions.csv`. **0.728 ± 0.019 = mean ± std of 5 seed-level OOF AUROCs** (0.746, 0.729, 0.742, 0.699,
0.724), not a 25-fold average.

## 7. Model-selection bias (development tuning, not leakage)

* **Budget K.** WP001 fixed min 8 / target 16 / max 32 *a priori*, and every fixed-budget run reported the full
  grid (4/8/16/32[/64]). But "Decider Fixed-16 = 0.728" is the best of the four fixed budgets in
  `decider_v3_adaptive` (0.725 / 0.688 / 0.728 / 0.711) and was then singled out as the serial reference for WP003.
  Quoting the best point of a grid evaluated on the same 79 slides is optimistic → **development-set selection**.
  The pre-registered curve is the honest object.
* **Decider prompt / state.** V1→V2 state wording, the V2 separate STOP question, the V2.1 halting wording and
  the removal of the trend label were all chosen on real states from this cohort. Selection criteria were
  label-free behavioural statistics (P(SELECT) vs strength, P(STOP) vs gain ratio, stop distribution), not the
  OOF AUROC, and the acquisition policy never saw a label — but the cohort was reused across iterations →
  development tuning, MEDIUM.
* **Saturation-threshold grid** rescaled to {0.25…0.45} after inspecting the cohort's gain distribution
  (unsupervised, label-free); the final threshold is still chosen by nested CV. LOW.
* **Classifier**: never tuned. Cheap features, pool size 128, shortlist 4, window 4: set by the briefs, not searched.
* **Seeds** change only the fold split (acquisition is label-free and run once per slide; the option shuffle
  seed is fixed per slide). The ± therefore measures split variability only, not acquisition variability.

## 8. Result provenance

All result directories predate `git init` (2026-10-16 14:43); each holds `config_used.yaml`. The baseline commit
`e6556f7` is the code state at the end of that day; no code in the V2.1 path changed between the
`decider_v3_adaptive` run (14:11) and the commit, but this cannot be proven from history (LOW finding).

| Number | Result dir / file | Method key | Script | Config | Seeds / CV |
|---|---|---|---|---|---|
| Full UNI 0.679 ± 0.025 | `results/decider_v3_adaptive/summary.csv` (identical in `decider_v3_active`) | `full_uni` | `scripts/run_adaptive_cv.py` (`run_sequential_cv.py`) | `config_used.yaml` | 5 seeds × 5 folds |
| Decider Fixed-16 **0.728 ± 0.019** | `results/decider_v3_adaptive/summary.csv` | `decider_fixed_16` (V2.1 state, shuffled options, min=max=16) | `run_adaptive_cv.py` | same | same |
| Decider Fixed-16 (V2 state) 0.709 ± 0.014 | `results/decider_v3_active/budget_curve.csv` | `decider_fixed_16` (V2 state with target fields, joint routing) | `run_sequential_cv.py` | its `config_used.yaml` | same |
| Cheap Fixed-16 0.702 ± 0.022 | `decider_v3_adaptive/summary.csv` = `decider_v3_active/budget_curve.csv` `cheap_greedy_16` | `cheap_greedy` (sequential unified exploration score) | both scripts | same | same |
| Cheap Top-16 0.723 ± 0.017 | `results/decider_v3_active/budget_curve.csv` | `cheap_topk_16` (16 largest cheap clusters, non-sequential) | `run_sequential_cv.py` | same | same |
| Decider Adaptive 0.704 ± 0.032 | `decider_v3_adaptive/summary.csv` | `decider_adaptive` | `run_adaptive_cv.py` | same | same |
| Threshold Adaptive 0.718 ± 0.020 | `decider_v3_adaptive/summary.csv` | `threshold_adaptive` (nested) | `run_adaptive_cv.py` | same | same |

**0.723 vs 0.702 resolved:** two different algorithms. `cheap_cluster_topk` (`src/acquisition/sequential_agent.py`)
takes the K largest cheap clusters' representatives; `cheap_greedy` takes K sequential picks by the unified
exploration score (novelty + distance + cluster size + content). WP002's "Cheap Fixed-K" table used
`cheap_greedy` (`run_adaptive_cv.py:85`), WP001's table listed both. Same folds, same classifier, same seeds.

## 9. Findings

| ID | Severity | Finding | Affected result |
|---|---|---|---|
| F1 | HIGH | Headline "Decider Fixed-16 = 0.728" is the best of a 4-point budget grid evaluated on the same cohort; the budget was pre-specified but the choice to headline K=16 was made after seeing the grid. | 0.728; WP003 serial reference |
| F2 | MEDIUM | Decider state/prompt iterated on real states from this cohort (label-free criteria). | all Decider rows |
| F3 | MEDIUM | Two different runs both called "Decider Fixed-16" (0.709 V2 state vs 0.728 V2.1 state); reports must name the state version. | 0.728 / 0.709 |
| F4 | LOW | Results predate git; code version inferred, not recorded, for every pre-baseline result. | all |
| F5 | LOW | "Cheap Fixed-16" and "Cheap Top-16" were used in different reports for different algorithms without saying so. | 0.702 / 0.723 |
| F6 | LOW | Seed variability reflects fold splits only; acquisition is deterministic per slide. | all ± values |
| F7 | INFO | Candidate npz files carry a `label` key in memory during V2.1 runs; never read. | none |

No CRITICAL finding: no patient overlap, no scaler/PCA fit on validation data, no label or unacquired
embedding in the acquisition state, no validation data in any fit, correct per-seed OOF AUROC.

## 10. Final confidence

> Do we have evidence that the reported Decider Fixed-16 AUROC ≈ 0.728 is artificially inflated by
> train/validation leakage?

**NO.** The patient folds are disjoint and cover every patient once, every learned component (scaler,
logistic regression, saturation threshold) is fit inside the training fold, the acquisition policy is
label-agnostic and only ever reads embeddings it has acquired, and the metric is a per-seed out-of-fold
AUROC. What the number does carry is optimistic *selection* bias from quoting the best fixed budget (F1) and
from prompt iteration on the same cohort (F2): the defensible headline is the whole budget curve
(0.688–0.728 for K = 8–32, all within ~1 SD of each other and of Full UNI 0.679), not the single best point.
