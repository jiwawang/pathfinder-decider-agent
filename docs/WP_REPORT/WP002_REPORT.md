# WP002 report — V2.1 target-free adaptive STOP (run `results/decider_v3_adaptive`, 2026-10-16)

Patient-level 5-fold CV x 5 seeds, L2 logistic regression on the mean of the acquired UNI embeddings; the
acquisition policy is label-agnostic and runs once per slide. Archived runs: `decider_v3_adaptive_v0_failed`
(first target-free prompt, never stopped), `decider_v3_adaptive_v1_sharedstate` (halt question saw the
candidate shortlist, 52% forced stops).

## 1. Target count removed
Yes. The state contains `uni_calls_used` and `maximum_uni_calls` only; `tests/test_v21_adaptive.py::
test_target_count_not_in_state` scans every state and option text for target / expected / preferred.
`min_uni_calls: 4`, `max_uni_calls: 32`.

## 2. STOP prompt (final)
State (`expensive_observation_state`): calls used / max, remaining candidates, mean pairwise UNI similarity
and distance, the information gain of every acquisition so far, last-4 gains with mean / min / max / slope,
mean of the first acquisitions, mean of the last acquisitions, `recent_gain_as_percent_of_initial_gains`,
new cheap clusters in the last 4, cheap-cluster and spatial coverage. No trend label, no target, no label.

Question: *Is the evidence from expensive UNI observations saturating? Compare the recent information gains
with the initial ones.* Options: `CONTINUE` — not saturating: recent gains are comparable to the initial
gains; `STOP` — saturating: recent gains are much lower than the initial gains. Asked on the saturation
state alone (step 1); routing (A–D, shuffled) is a second call on the full state (step 2).

## 3. Information gain
`information_gain_t = 1 - max_j cos(e_t, e_j)` over embeddings acquired before `t`; 1.0 for the first.
Window 4 for the recent statistics. Measured gains plateau around 0.40 after ~12 acquisitions (mean gain at
round 4 / 8 / 16 / 24 / 32 = 0.54 / 0.48 / 0.45 / 0.42 / 0.40), so "saturation" on this cohort is a plateau,
not a decay to zero.

## 4–5. UNI-call distribution (Decider Adaptive)
mean 20.3 · median 20 · std 8.9 · min 8 · max 32 · P25 11.5 · P75 32. Stopped at min 0%; 5–8: 2.5%; 9–16:
40.5%; 17–24: 24.1%; 25–31: 6.3%; forced max: 26.6%. 73.4% of slides stopped on the agent's own decision.
`uni_calls_histogram.png`, `information_gain_examples.png`.

## 6–8. Main table
| Method | Mean UNI calls | AUROC | Balanced Acc | F1 |
|---|---:|---:|---:|---:|
| Full UNI | 25,033 | 0.679 ± 0.025 | 0.626 | 0.659 |
| Cheap Fixed-8 | 8 | 0.692 ± 0.019 | 0.622 | 0.668 |
| Cheap Fixed-16 | 16 | 0.702 ± 0.022 | 0.635 | 0.678 |
| Cheap Fixed-32 | 32 | 0.695 ± 0.024 | 0.624 | 0.673 |
| Threshold Adaptive (nested threshold, mostly 0.35) | 19.8 | 0.718 ± 0.020 | 0.644 | 0.682 |
| Decider Fixed-8 | 8 | 0.688 ± 0.025 | 0.619 | 0.668 |
| Decider Fixed-16 | 16 | 0.728 ± 0.019 | 0.676 | 0.716 |
| Decider Fixed-32 | 32 | 0.711 ± 0.026 | 0.656 | 0.704 |
| **Decider Adaptive V3** | **20.3** | **0.704 ± 0.032** | **0.640** | **0.683** |

Threshold Adaptive: calls mean 21.1, median 19, std 9.4, range 5–32, forced max 34%; threshold chosen per
training fold by inner CV: 0.35 in 17 of 25 folds (0.45 x4, 0.25 x2, 0.40 x2).

## 9–10. Routing: position vs strategy (1,603 adaptive decisions; 4,740 over all Decider runs)
| Position | % | | Strategy | % |
|---|---:|---|---|---:|
| A | 14.3 | | largest cluster | 62.5 |
| B | 24.2 | | cheap novelty | 3.1 |
| C | 35.0 | | spatial far | 17.5 |
| D | 26.5 | | under-covered | 17.0 |

## 11. STOP probability vs information gain
Mean recent gain 0.365 at STOP vs 0.497 at CONTINUE; latest gain 0.319 vs 0.468. corr(P(STOP), recent gain)
= −0.58; corr with calls used = +0.30; with pairwise similarity = +0.07. Calibration buckets of P(STOP):
0–0.2 (n=515, gain 0.57, never stops), 0.2–0.4 (n=371, gain 0.48), 0.4–0.6 (n=423, gain 0.42, 5% stop),
0.6–0.8 (n=36, gain 0.37, 100% stop). The halting probability is monotone in the measured gain.

## 12. Position bias
Yes, a moderate one: after shuffling, C is chosen 35% and A only 14% of the time (uniform would be 25%). The
strategy preference is the dominant effect (largest cluster 62.5%), and it survives the shuffle, so the
V2 "A = 60%" result was mostly a real strategy preference; but the model also favours the third position.
Reported, not hidden; the shuffle stays on so it cannot align with any strategy.

## 13. Slide-specific compute allocation
Yes. Calls range 8–32 with a broad body between 9 and 24 and a 27% tail that reaches the maximum. The
Decider and the deterministic saturation rule spend almost the same budget on average (20.3 vs 19.8) with
similar spread; the Decider's STOP is driven by the gain ratio without any coded threshold.

## 14. Differences from V2 (`decider_v3_active`)
V2: 75/79 slides stopped at exactly 16 (target in the state), AUROC 0.723. V2.1: no target, budgets 8–32,
median 20, AUROC 0.704 ± 0.032 (within one SD of every fixed-budget point; the budget curve is flat on this
cohort). Routing was shuffled, which exposed the position bias.

## 15. Failure modes observed (all kept)
* `decider_v3_adaptive_v0_failed`: "Would another observation still add meaningful new evidence?" with a
  `recent_gain_trend` label → the model reads the word only. "decreasing" ⇒ P(STOP) = 1.0 (would stop every
  slide at the minimum); without the label P(STOP) ≤ 0.24 on all 2,212 states ⇒ never stops (all 79 forced to 32).
* `decider_v3_adaptive_v1_sharedstate`: correct question, but asked together with the routing question on
  the full state (shortlist included) → 52% forced stops, mean 25.9 calls.
* A numeric anchor in the option text ("below about 60") made the model *less* decisive (P ≤ 0.41).
* Decider Fixed-8 (0.688) is below Cheap Fixed-8 (0.692) and Decider Fixed-4; noise at n = 79.
* Cheap Fixed-16 = 0.702 and Threshold Adaptive = 0.718 remain fully competitive: the cohort carries a
  strong globally distributed morphology signal (WP002 section 21).

Latency: ~151 ms per round (two sequential Decider calls when STOP is allowed); mean halting confidence 0.44.
