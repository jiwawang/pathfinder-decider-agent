"""WP007 smoke tests 1-3 (section 22-24). Run on the EC2 instance (or locally for a dry run).
  python scripts/aws/smoke_test.py --test bedrock|decider|uni|all [--reference-npz <local reference>]
Writes results/aws_smoke/smoke_<test>.json. Never prints credentials.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common.config import load_config, resolve  # noqa: E402

OUT = resolve("results/aws_smoke")


def test_bedrock() -> dict:
    import boto3
    region, model = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION"), os.environ.get("BEDROCK_MODEL_ID")
    if not region or not model:
        return {"status": "BLOCKED", "reason": "AWS_REGION / BEDROCK_MODEL_ID not set"}
    client = boto3.client("bedrock-runtime", region_name=region)
    t0 = time.perf_counter()
    r = client.converse(modelId=model, messages=[{"role": "user", "content": [{"text": "Reply with the single word PATHFINDER."}]}], inferenceConfig={"maxTokens": 16, "temperature": 0.0})
    dt = time.perf_counter() - t0
    text = "".join(c.get("text", "") for c in r["output"]["message"]["content"])
    return {"status": "PASS" if text.strip() else "FAIL", "region": region, "model_id": model, "latency_s": round(dt, 2), "request_id": r["ResponseMetadata"].get("RequestId"),
            "usage": r.get("usage"), "response_text": text[:80]}


def test_decider(device: str) -> dict:
    from src.decider.client import StrandsDeciderBackend
    demo = load_config("configs/aws_demo.yaml")["demo"]
    t0 = time.perf_counter(); be = StrandsDeciderBackend(demo["decider_model"], device); load = time.perf_counter() - t0
    state = {"expensive_observation_state": {"uni_calls_used": 3, "maximum_uni_calls": 16, "remaining_candidate_count": 125, "mean_pairwise_uni_similarity": 0.41},
             "candidates": {"A": {"cheap_novelty": 0.9, "distance_from_nearest_acquired": 0.7, "tissue_fraction": 0.95, "entropy": 0.8, "why_shortlisted": "largest cluster"},
                            "B": {"cheap_novelty": 0.2, "distance_from_nearest_acquired": 0.1, "tissue_fraction": 0.3, "entropy": 0.4, "why_shortlisted": "cheap novelty"},
                            "C": {"cheap_novelty": 0.6, "distance_from_nearest_acquired": 0.9, "tissue_fraction": 0.9, "entropy": 0.7, "why_shortlisted": "spatial far"},
                            "D": {"cheap_novelty": 0.5, "distance_from_nearest_acquired": 0.4, "tissue_fraction": 0.8, "entropy": 0.6, "why_shortlisted": "under-covered"}}}
    opts = {k: f"{v['why_shortlisted']}; novelty {v['cheap_novelty']}, distance {v['distance_from_nearest_acquired']}, tissue {v['tissue_fraction']}, entropy {v['entropy']}" for k, v in state["candidates"].items()}
    lat = []
    for _ in range(5):
        t0 = time.perf_counter(); choice, probs, conf = be.choose(state, "Which candidate should receive the next expensive UNI inference?", opts); lat.append((time.perf_counter() - t0) * 1000)
    return {"status": "PASS" if choice in opts else "FAIL", "model": demo["decider_model"], "load_s": round(load, 1), "choice": choice, "probabilities": probs, "confidence": conf,
            "latency_ms": {"mean": round(float(np.mean(lat[1:])), 1), "median": round(float(np.median(lat[1:])), 1)}, "mapped_patch": "A->patch id resolved by the coordinator at runtime"}


def test_uni(device: str, reference_npz: str | None) -> dict:
    """Encoder smoke test (name kept for the CLI): loads the configured encoder (default H-optimus-0), encodes one raw patch."""
    import torch
    from PIL import Image
    from src.encoders.registry import build_encoder, encode
    demo = load_config("configs/aws_demo.yaml")["demo"]; enc = os.environ.get("PATHFINDER_ENCODER", demo.get("encoder", "hoptimus0"))
    root = Path(demo["local_root"]) if not os.environ.get("PATHFINDER_CACHE") else Path(os.environ["PATHFINDER_CACHE"])
    slide_dir = next((root / "slides").iterdir()); cands = json.load(open(slide_dir / "candidates.json"))
    t0 = time.perf_counter(); m, tf, sp = build_encoder(enc, device); load = time.perf_counter() - t0
    c = cands[0]; img = Image.open(slide_dir / c["png"]).convert("RGB"); lat = []
    for _ in range(5):
        t0 = time.perf_counter(); e = encode(m, sp, tf(img).unsqueeze(0), device, True)[0]; lat.append((time.perf_counter() - t0) * 1000)
    out = {"status": "PASS" if e.shape == (sp.dim,) and np.isfinite(e).all() else "FAIL", "encoder": enc, "model": sp.hub, "license": sp.license, "load_s": round(load, 1), "shape": list(e.shape), "finite": bool(np.isfinite(e).all()),
           "latency_ms_batch1": {"mean": round(float(np.mean(lat[1:])), 1), "median": round(float(np.median(lat[1:])), 1)}, "patch": c["png"], "slide": slide_dir.name}
    ref_path = Path(reference_npz) if reference_npz else slide_dir / f"reference_embedding_{enc}.npz"
    if ref_path.exists():
        ref = np.load(ref_path); cos = float(ref["embedding"] @ e / (np.linalg.norm(ref["embedding"]) * np.linalg.norm(e)))
        out["cosine_vs_local_reference"] = round(cos, 6); out["reference_host"] = str(ref.get("host", ""))
    else:
        np.savez(ref_path, embedding=e, host=platform.node(), patch=c["png"]); out["reference_written"] = str(ref_path)
    if device.startswith("cuda"):
        out["gpu_memory_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--test", default="all"); ap.add_argument("--device", default="cuda"); ap.add_argument("--reference-npz", default=None)
    args = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    tests = ["bedrock", "decider", "uni"] if args.test == "all" else [args.test]
    for t in tests:
        try:
            res = {"bedrock": lambda: test_bedrock(), "decider": lambda: test_decider(args.device), "uni": lambda: test_uni(args.device, args.reference_npz)}[t]()
        except Exception as e:
            res = {"status": "FAIL", "error": f"{type(e).__name__}: {str(e)[:200]}"}
        res.update(test=t, timestamp_utc=datetime.now(timezone.utc).isoformat(), host=platform.node(), region=os.environ.get("AWS_REGION", ""))
        (OUT / f"smoke_{t}.json").write_text(json.dumps(res, indent=1, default=float)); print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
