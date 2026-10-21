"""Amazon Bedrock agent report (README section 18).

Bedrock never sees the response label and never decides the prediction. It receives the slide
prediction, the decider trace summary and selection statistics, and writes a short, plain-language
report of what the agent did. Model id and region come from environment variables only.

If AWS credentials / Bedrock access are unavailable, `generate_report` falls back to a deterministic
template so the pipeline and demo still run; the returned dict says which path was used.
"""
from __future__ import annotations

import json
import os
from typing import Any

SYSTEM_PROMPT = (
    "You are the reporting component of PathFinder, a research prototype that studies how an "
    "evidence-selection agent searches a gigapixel pathology slide. Write a concise report (120-180 "
    "words, plain prose, no bullet points) describing what the agent examined, what it selected, why "
    "it stopped, and the classifier's output. Describe class-1 as 'responder-like' and class-0 as "
    "'non-responder-like' patterns learned from training slides. Never give a diagnosis or clinical "
    "recommendation; state explicitly that this is a research prototype, not a clinical tool."
)


def summarise_episode(slide_id: str, prediction: dict[str, Any], episode: dict[str, Any],
                      class_names: dict[int, str]) -> dict[str, Any]:
    st = episode["stats"]
    trace = episode["trace"]
    sel = [t for t in trace if t.get("decision") == "SELECT"]
    votes = [t["state"]["candidate"].get("weak_class_1_vote") for t in sel if "state" in t]
    return {
        "slide_id": slide_id,
        "prediction": {"predicted_class": int(prediction["pred_label"]),
                       "predicted_class_name": class_names[int(prediction["pred_label"])],
                       "probability_class_1": round(float(prediction["pred_probability"]), 3)},
        "search": {"candidate_regions_available": int(st["num_candidates"]),
                   "candidates_examined": int(st["num_examined"]),
                   "patches_selected": int(st["num_selected"]),
                   "tissue_clusters_covered": int(st["clusters_covered"]),
                   "stopped_by_agent": bool(st["stopped_by_model"]),
                   "action_counts": st["action_counts"],
                   "mean_decision_confidence": round(float(st["mean_confidence"]), 3),
                   "low_confidence_decision_rate": round(float(st["low_confidence_rate"]), 3),
                   "mean_decision_latency_ms": round(float(st["mean_latency_ms"]), 1)},
        "selected_evidence": {"mean_weak_class_1_vote": round(float(sum(votes) / len(votes)), 3) if votes else None,
                              "mean_novelty_at_selection": round(float(sum(t["state"]["candidate"]["novelty_vs_selected_evidence"] for t in sel if "state" in t) / max(1, len(sel))), 3)},
    }


def template_report(s: dict[str, Any]) -> str:
    p, q = s["prediction"], s["search"]
    if "uni_calls" in q:   # V2 sequential acquisition summary
        stop = "stopped on its own once the acquired evidence was judged diverse enough" if q["stopped_by_agent"] \
            else "reached the maximum UNI budget before stopping"
        return (f"The agent searched a slide of {q['tissue_patches_in_slide']:,} tissue patches through a cheap candidate pool "
                f"of {q['candidate_pool']} regions. Over {q['rounds']} rounds it chose which region deserved the next UNI "
                f"inference and spent {q['uni_calls']} UNI calls in total, avoiding {q['uni_compute_avoided_percent']}% of the "
                f"encoder compute a full-slide pass would need; the acquired patches cover {q['cheap_clusters_covered']} distinct "
                f"tissue clusters. The slide classifier predicted class {p['predicted_class']} ({p['predicted_class_name']}) "
                f"with probability {p['probability_class_1']} for class 1. The agent {stop}; mean decision confidence was "
                f"{q['mean_decision_confidence']} at {q['mean_decision_latency_ms']} ms per decision and "
                f"{q['mean_uni_call_latency_ms']} ms per UNI call. This is a research prototype for treatment-response "
                f"prediction, not a diagnostic or clinical tool.")
    e = s["selected_evidence"]
    lean = "responder-like (class-1)" if (e["mean_weak_class_1_vote"] or 0.5) >= 0.5 else "non-responder-like (class-0)"
    stop = "stopped on its own once the evidence set was judged sufficient" if q["stopped_by_agent"] \
        else "examined every candidate region before the budget rules ended the search"
    return (f"The agent examined {q['candidates_examined']} of {q['candidate_regions_available']} candidate regions "
            f"and selected {q['patches_selected']} evidence patches spanning {q['tissue_clusters_covered']} tissue clusters. "
            f"The selected evidence showed stronger similarity to {lean} training-slide patterns "
            f"(mean weak vote {e['mean_weak_class_1_vote']}). The slide classifier predicted class "
            f"{p['predicted_class']} ({p['predicted_class_name']}) with probability {p['probability_class_1']} for class 1. "
            f"The agent {stop}; mean decision confidence was {q['mean_decision_confidence']} at "
            f"{q['mean_decision_latency_ms']} ms per decision. This is a research prototype for treatment-response "
            f"prediction, not a diagnostic or clinical tool.")


def generate_report(summary: dict[str, Any], *, max_tokens: int = 600) -> dict[str, Any]:
    model_id = os.environ.get("BEDROCK_MODEL_ID")
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    if not model_id or not region:
        return {"source": "template", "reason": "BEDROCK_MODEL_ID / AWS_REGION not set", "report": template_report(summary)}
    try:
        import boto3
        client = boto3.client("bedrock-runtime", region_name=region)
        resp = client.converse(
            modelId=model_id,
            system=[{"text": SYSTEM_PROMPT}],
            messages=[{"role": "user", "content": [{"text": "Agent run summary (JSON):\n" + json.dumps(summary, indent=1)}]}],
            inferenceConfig={"maxTokens": int(max_tokens), "temperature": 0.2},
        )
        text = "".join(c.get("text", "") for c in resp["output"]["message"]["content"]).strip()
        return {"source": "bedrock", "model_id": model_id, "region": region, "report": text,
                "usage": resp.get("usage", {})}
    except Exception as e:  # credentials, access, throttling: degrade gracefully, say so
        return {"source": "template", "reason": f"bedrock call failed: {type(e).__name__}: {e}",
                "report": template_report(summary)}
