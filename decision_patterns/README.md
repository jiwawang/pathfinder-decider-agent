# Evidence Context Selector — reusable Strands Decider 2B pattern

**Pattern tag:** `#context-memory`  **Model:** `#strands-decider-2b`

## The decision

An agent has many candidate context chunks and a small evidence budget. Instead of averaging
everything, it walks the candidates and asks the Decider one typed `choice` question per candidate:

| Option | Meaning |
|---|---|
| `SELECT` | the candidate is reliable, representative and non-redundant: add it to the evidence set |
| `SKIP` | the candidate is redundant, poorly supported or unlikely to help |
| `STOP` | the evidence set is already diverse and informative: stop searching |

The decision is made *repeatedly* (once per candidate), *changes the agent's state* (the evidence set
and every later novelty value) and *determines the final representation* (only `SELECT`ed chunks are
pooled). The LLM (Amazon Bedrock) only writes the report afterwards; it never selects evidence.

## Generic inputs

The Decider never sees raw content (no embeddings, no images, no long text). It sees a compact
structured state, rendered as JSON by `strands_decider`:

```json
{
  "candidate": {
    "evidence_support_score": 0.82,        // weak evidence the chunk is informative (domain-specific)
    "support_confidence": 0.64,            // how unanimous that support is
    "representativeness": 0.91,            // how typical the chunk is of its group / region
    "coverage_percent": 8.3,               // share of the source covered by the chunk's group
    "novelty_vs_selected_evidence": 0.76   // 1 - max similarity to already selected evidence
  },
  "current_evidence_state": {
    "selected_items": 5, "minimum_evidence": 6, "target_evidence_budget": 12,
    "maximum_evidence_budget": 16, "unique_groups_represented": 5,
    "candidates_examined": 11, "candidates_remaining": 21
  }
}
```

Any domain that can compute those five candidate numbers can reuse the pattern unchanged:

* document chunks for RAG (support = retriever score, novelty = 1 − max cosine to selected chunks)
* retrieved agent memories (support = recency/relevance, coverage = share of episodes)
* sensor observations or image regions
* pathology patches (this repository: support = weak KNN vote from training slides)

## Guard-rails that keep the agent stable

Implemented in `evidence_selector.py`, outside the model:

* `STOP` is only offered once `min_selected` is reached; forced at `max_selected`.
* Answers below `low_confidence_threshold` fall back to `SKIP` (configurable), so an uncertain model
  never pollutes the evidence set. Every low-confidence case is kept in the trace.
* Every decision records the chosen option, option probabilities, confidence and latency.

## Usage

```python
from decision_patterns.evidence_selector import EvidenceSelector, CandidateState, EvidenceState
from src.decider.client import StrandsDeciderBackend   # or ThresholdPolicy() for a model-free ablation

selector = EvidenceSelector(StrandsDeciderBackend("StrandsAgents/strands-decider-2B-qwen3.5-v1-2610"),
                            min_selected=6, target_selected=12, max_selected=16,
                            low_confidence_threshold=0.3)
decision = selector.decide(CandidateState(support=0.82, support_confidence=0.64, representativeness=0.91,
                                          coverage=0.083, novelty=0.76),
                           EvidenceState(selected_count=5, min_selected=6, target_selected=12,
                                         max_selected=16, unique_groups=5, examined_count=11, remaining_count=21))
decision.action, decision.probabilities, decision.confidence, decision.latency_ms
```

`src/decider/episode.py` shows the full loop for one slide, including novelty bookkeeping and tracing.

## Why a decision model and not an LLM

One slide needs ~30 decisions; a cohort-level cross-validation needs tens of thousands. The Decider
answers each in ~65 ms on a single RTX 4090 with calibrated probabilities, which makes the agent
loop cheap enough to run inside every CV fold, and makes its confidence usable as a guard-rail.
