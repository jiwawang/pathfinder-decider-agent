# Adaptive Expensive-Observation Router — reusable Strands Decider 2B pattern

**Pattern tag:** `#context-memory`  **Model:** `#strands-decider-2b`  **Code:** `adaptive_expensive_observation_router.py`

## The decision

> Given a large set of cheap observations and a limited budget of expensive ones, repeatedly decide
> which candidate deserves the next expensive observation, and when enough evidence has been collected.

Each round a cheap heuristic nominates a shortlist (A, B, C, D), each with a one-line reason and a
few cheap statistics. The Decider answers one typed `choice` question:

*Which candidate should receive the next expensive observation?* → `A | B | C | D | STOP`

The chosen candidate is the **only** one that gets the expensive computation. The result updates the
agent's evidence state (coverage, redundancy among what was acquired) and the next round begins.
`STOP` ends the search.

| Domain | cheap observation | expensive observation |
|---|---|---|
| pathology WSI (this repo) | colour / texture / position of a 224 px patch | UNI ViT-L forward pass |
| video | frame timestamp, motion energy, thumbnail stats | large VLM call |
| RAG | BM25 / embedding retrieval score, chunk length | expensive cross-encoder rerank or LLM read |
| remote sensing | low-resolution tile statistics | high-resolution scan |
| finance | event metadata, cheap anomaly score | deep analysis |

## What the Decider sees

```json
{"acquisition_state": {"uni_patches_acquired": 7, "minimum_budget": 8, "target_budget": 16, "maximum_budget": 32,
                       "cheap_clusters_covered": 7, "total_cheap_clusters": 128, "spatial_coverage_percent": 42.0,
                       "mean_pairwise_uni_similarity_of_acquired": 0.61,
                       "target_budget_reached": false, "acquired_evidence_is_diverse": false},
 "candidates": {"A": {"cluster_coverage_percent": 8.4, "cheap_novelty": 0.73, "distance_from_nearest_acquired": 0.64,
                      "tissue_fraction": 0.91, "entropy": 0.67, "representativeness": 0.92,
                      "why_shortlisted": "largest unexplored tissue cluster", "exploration_score": 2.3},
                "B": {...}, "C": {...}, "D": {...}}}
```

Options (what the model reads next to each id): the shortlist reason plus the same cheap numbers,
e.g. `A — largest unexplored tissue cluster; coverage 8.4%, novelty 0.73, distance 0.64, tissue 0.91, entropy 0.67`,
and `STOP — target_budget_reached is true and acquired_evidence_is_diverse is true: enough evidence has been collected`.

Nothing about an *unacquired* candidate's expensive representation is ever in the state. Nothing about
the task label is ever in the state (the acquisition policy is label-agnostic).

## Guard-rails (outside the model)

* `STOP` is only offered after `min_calls` expensive calls; it is forced at `max_calls`.
* A decision below `low_confidence_threshold` falls back to the heuristic's best-scored option, never
  to `STOP`, so an unsure model keeps exploring. Every override is logged.
* The expensive resource is only reachable through `acquire(id)`, which counts calls; reading an
  unacquired item raises (`src/acquisition/lazy_uni_store.py`, tested in `tests/test_v2_acquisition.py`).

## Usage

```python
from decision_patterns.adaptive_expensive_observation_router import AdaptiveObservationRouter, Option
from src.decider.client import StrandsDeciderBackend          # or HeuristicRouter() for a model-free ablation

router = AdaptiveObservationRouter(StrandsDeciderBackend("StrandsAgents/strands-decider-2B-qwen3.5-v1-2610"),
                                   min_calls=8, target_calls=16, max_calls=32, low_confidence_threshold=0.15)
options = [Option("A", "largest unexplored cluster; coverage 8.4%, novelty 0.73, ...", payload=cand_a, score=2.3), ...]
d = router.decide(state, options, calls_used=7)
d.choice, d.probabilities, d.confidence, d.latency_ms    # "B", {...}, 0.62, 55.0
```

`src/acquisition/sequential_agent.py` is the complete loop for the pathology case, including the cheap
shortlist (`shortlist.py`) and the lazy UNI store.

## Why a decision model

A slide needs 10–30 rounds; a cohort-level experiment needs tens of thousands of decisions. The Decider
answers each in ~55 ms with calibrated probabilities, i.e. cheaper than the expensive observation it
is routing (a UNI forward pass is ~8–13 ms per patch at batch 1, but a full slide is 25,000 of them).

## V2.1: target-free halting on information gain (`src/acquisition/adaptive_agent.py`)

The first version told the model a target budget, and it stopped at exactly that number on 75 of 79
slides: the Decider was a fixed-16 counter. V2.1 removes every target or preferred count from the
state. The halting question reads evidence-saturation numbers only:

```json
{"expensive_observation_state": {"uni_calls_used": 9, "maximum_uni_calls": 32, "remaining_candidate_count": 119,
   "mean_pairwise_uni_similarity": 0.58, "mean_pairwise_uni_distance": 0.42,
   "information_gain_of_each_acquisition_so_far": [1.0, 0.71, 0.62, 0.55, 0.49, 0.44, 0.39, 0.33, 0.31],
   "latest_information_gain": 0.31, "recent_information_gains": [0.44, 0.39, 0.33, 0.31], "recent_gain_mean": 0.37,
   "recent_gain_min": 0.31, "recent_gain_max": 0.44, "recent_gain_slope": -0.045,
   "average_gain_of_first_acquisitions": 0.59, "average_gain_of_last_acquisitions": 0.37,
   "recent_gain_as_percent_of_initial_gains": 62.7, "new_cheap_clusters_found_in_last_4": 4,
   "cheap_cluster_coverage_percent": 7.0, "spatial_coverage_percent": 31.2}, "candidates": {...}}
```

`information_gain_t = 1 - max_j cos(e_t, e_j)` over the embeddings acquired before `t` (1.0 for the first).
Two questions per round, one model call. Halting: *Is the evidence from expensive UNI observations
saturating? Compare the recent information gains with the initial ones.* → `CONTINUE` ("not saturating:
recent gains are comparable to the initial gains") / `STOP` ("saturating: recent gains are much lower than
the initial gains"), offered only between `min_uni_calls` and `max_uni_calls`. Routing: A–D as before, with
option letters shuffled every round (seeded) and both position and strategy stored in the trace.

What we measured before settling on this (2,212 real states): STOP as a fifth routing option never fires;
the words "redundant / meaningful new evidence" with the same numbers keep P(STOP) ≤ 0.24 forever; a
categorical `recent_gain_trend: decreasing` field is read as a command (P(STOP) = 1.0 whenever present, so
every slide stops at the minimum). Asking the model to *compare* recent and initial gains makes P(STOP)
follow the measured ratio (corr −0.82) and gives slide-specific stopping. No gain threshold is coded
anywhere; the deterministic `SaturationThresholdRouter` baseline exists to test whether a threshold rule
does as well.
