"""PathFinder cloud entrypoint (WP007 sections 12-15): the real Decider -> UNI -> evidence -> classifier -> Bedrock loop.

  python scripts/aws/run_agent.py --slide-id <id> --budget 16 --mode live --source s3|local

source=s3    : artifacts are pulled from s3://$S3_BUCKET/$S3_PREFIX/slides/<slide_id>/ (private bucket, instance role)
source=local : the same layout from configs/aws_demo.yaml demo.local_root (used for the local dry run)
mode=live    : UNI runs on the GPU on the selected raw patch only; unselected candidates are never encoded
mode=adaptive: same loop with the frozen V2.1 adaptive STOP (min 4 / max 32)

The acquisition code is the frozen src/acquisition (AdaptiveRouterV21 + run_adaptive_episode); this file only
provides the artifact loader, a PNG-backed on-demand UNI store, the frozen classifier and the Bedrock report.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.acquisition.adaptive_agent import AdaptiveRouterV21, run_adaptive_episode  # noqa: E402
from src.acquisition.lazy_uni_store import _StoreBase  # noqa: E402
from src.bedrock.reporter import generate_report  # noqa: E402
from src.common.config import load_config, resolve  # noqa: E402


class PNGPatchUNIStore(_StoreBase):
    """On-demand encoder over the candidates' raw PNG patches: a row exists only after acquire().
    The encoder is configurable (hackathon default H-optimus-0, Apache-2.0); see src/encoders/registry.py."""

    def __init__(self, slide_dir: Path, candidates: list[dict], encoder: str, device: str = "cuda"):
        super().__init__()
        from src.encoders.registry import build_encoder
        self.dir, self.cands, self.device, self.encoder = slide_dir, {c["patch_index"]: c for c in candidates}, device, encoder
        t0 = time.perf_counter()
        self.model, self.tf, self.spec = build_encoder(encoder, device)
        self.load_s = time.perf_counter() - t0

    def _row(self, slide_id: str, patch_index: int) -> np.ndarray:
        from src.encoders.registry import encode
        img = Image.open(self.dir / self.cands[int(patch_index)]["png"]).convert("RGB")
        return encode(self.model, self.spec, self.tf(img).unsqueeze(0), self.device, True)[0]

    def n_patches(self, slide_id: str) -> int:
        return len(self.cands)


def fetch_from_s3(bucket: str, prefix: str, slide_id: str, dest: Path) -> float:
    import boto3
    s3 = boto3.client("s3"); t0 = time.perf_counter(); n = 0
    for key_prefix in (f"{prefix}/slides/{slide_id}/", f"{prefix}/classifiers/", f"{prefix}/demo_manifest.json"):
        pages = s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=key_prefix)
        for page in pages:
            for obj in page.get("Contents", []):
                rel = obj["Key"][len(prefix) + 1:]; target = dest / rel
                if target.exists() and target.stat().st_size == obj["Size"]:
                    continue
                target.parent.mkdir(parents=True, exist_ok=True); s3.download_file(bucket, obj["Key"], str(target)); n += 1
    return time.perf_counter() - t0


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=str(Path(__file__).resolve().parents[2])).strip()
    except Exception:
        return os.environ.get("PATHFINDER_GIT_COMMIT", "unknown")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/aws_demo.yaml")
    ap.add_argument("--base-config", default="configs/default.yaml")
    ap.add_argument("--slide-id", required=True)
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--mode", choices=["live", "adaptive"], default="live")
    ap.add_argument("--source", choices=["s3", "local"], default="s3" if os.environ.get("S3_BUCKET") else "local")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-bedrock", action="store_true", help="skip the Bedrock call (template report, source=template)")
    args = ap.parse_args()
    demo = load_config(args.config)["demo"]; cfg = load_config(args.base_config); ad = cfg["adaptive"]
    budget = args.budget or int(demo["budget"])
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or ""
    t_all = time.perf_counter(); timings = {}

    if args.source == "s3":
        bucket, prefix = os.environ["S3_BUCKET"], os.environ.get("S3_PREFIX", demo["s3_prefix_default"])
        root = Path(os.environ.get("PATHFINDER_CACHE", "/tmp/pathfinder-demo"))
        timings["s3_fetch_s"] = round(fetch_from_s3(bucket, prefix, args.slide_id, root), 2)
    else:
        root = Path(demo["local_root"]); bucket = prefix = None
    sd = root / "slides" / args.slide_id
    slide = json.load(open(sd / "slide.json")); cands = json.load(open(sd / "candidates.json"))
    pool = {k: v for k, v in np.load(sd / "candidates.npz", allow_pickle=True).items()}
    encoder = os.environ.get("PATHFINDER_ENCODER", demo.get("encoder", "hoptimus0"))
    clf_name = slide.get("classifier_artifacts", {}).get(encoder) or slide["classifier_artifact"]
    clf = joblib.load(root / "classifiers" / f"{clf_name}.joblib")
    clf_prov = json.load(open(root / "classifiers" / f"{clf_name}.json"))
    assert slide["slide_id"] not in clf_prov["training_patient_ids"] and all(not slide["slide_id"].startswith(p) for p in clf_prov["training_patient_ids"]), "demo slide inside the classifier's training fold"

    t0 = time.perf_counter()
    from src.decider.client import StrandsDeciderBackend
    backend = StrandsDeciderBackend(demo["decider_model"], args.device); timings["decider_load_s"] = round(time.perf_counter() - t0, 1)
    store = PNGPatchUNIStore(sd, cands, encoder, args.device); timings["uni_load_s"] = round(store.load_s, 1)
    assert clf_prov.get("encoder_name", "uni") == encoder, f"classifier {clf_name} was trained on {clf_prov.get('encoder_name','uni')} embeddings, demo encoder is {encoder}"
    assert clf.n_features_in_ == store.spec.dim, "classifier feature dimension does not match the encoder"
    gpu = None
    if args.device.startswith("cuda"):
        import torch
        gpu = {"name": torch.cuda.get_device_name(0), "memory_after_load_gb": round(torch.cuda.memory_allocated() / 1e9, 2)}
    mn, mx = (budget, budget) if args.mode == "live" else (int(demo["adaptive_min"]), int(demo["adaptive_max"]))
    router = AdaptiveRouterV21(backend, min_calls=mn, max_calls=mx, low_confidence_threshold=float(ad["low_confidence_threshold"]))
    print(f"[agent] run {run_id} slide {args.slide_id} budget {mn}-{mx} candidates {len(cands)} source {args.source}", flush=True)

    t0 = time.perf_counter()
    ep = run_adaptive_episode(args.slide_id, pool, store, router, ad, shuffle_seed=int(ad["shuffle_seed"]), shortlist_size=int(ad["shortlist_size"]))
    timings["acquisition_loop_s"] = round(time.perf_counter() - t0, 2)
    assert store.total_calls == ep["stats"]["uni_calls"] == len(ep["acquired_patch_index"]), "UNI accounting mismatch"
    assert set(store.acquired[args.slide_id]) == set(ep["acquired_patch_index"]), "an unselected candidate was encoded"
    for t in ep["trace"]:
        if t.get("acquired_patch_index") is not None:
            print(f"[agent] round {t['round']:2d}: {t['selected_option']} ({t['selected_strategy']}, conf {t['route_confidence']:.2f}) -> UNI call #{t['uni_calls']+1} patch {t['acquired_patch_index']} gain {t['information_gain']:.3f}", flush=True)
        elif t["decision"] == "STOP":
            print(f"[agent] round {t['round']:2d}: STOP ({t.get('overridden_by') or 'model'})", flush=True)

    X = ep["embedding"].mean(0, keepdims=True); t0 = time.perf_counter(); P = clf.predict_proba(X)[0]; timings["classifier_s"] = round(time.perf_counter() - t0, 4)
    inv = {v: k for k, v in clf_prov["label_map"].items()}; pred = int(P.argmax())
    dec_lat = [t["latency_ms"] for t in ep["trace"] if t.get("model_queried")]
    summary = {"slide_id": args.slide_id, "mode": "aws_live_on_demand_uni" if args.source == "s3" else "local_dry_run_on_demand_uni",
               "prediction": {"predicted_class": pred, "predicted_class_name": inv[pred], "probabilities": {inv[i]: round(float(p), 4) for i, p in enumerate(P)}},
               "search": {"tissue_patches_in_slide": slide["n_tissue_patches"], "candidate_pool": len(cands), "rounds": ep["stats"]["rounds"], "uni_calls": ep["stats"]["uni_calls"],
                          "uni_compute_avoided_percent": round(100 * (1 - ep["stats"]["uni_calls"] / slide["n_tissue_patches"]), 3), "cheap_clusters_covered": ep["stats"]["clusters_covered"],
                          "stopped_by_agent": ep["stats"]["stopped_by_model"], "strategy_counts": ep["stats"]["strategy_counts"], "mean_decision_confidence": round(ep["stats"]["mean_halt_confidence"] or 0.0, 3),
                          "mean_decision_latency_ms": round(float(np.mean(dec_lat)), 1) if dec_lat else 0.0, "mean_uni_call_latency_ms": round(float(np.mean(store.latency_ms)), 1),
                          "wall_time_s": timings["acquisition_loop_s"], "final_mean_pairwise_uni_similarity": round(ep["stats"]["final_mean_pairwise_similarity"], 3),
                          "information_gains": [round(g, 3) for g in ep["information_gains"]]},
               "classifier": {"artifact": clf_name, "dataset": clf_prov["dataset"], "fold": clf_prov["fold"], "n_train": clf_prov["n_train"], "git_commit": clf_prov["git_commit"]},
               "true_label_hidden_from_reporter": True}
    t0 = time.perf_counter()
    report = {"source": "template", "reason": "--no-bedrock", "report": ""} if args.no_bedrock else generate_report(summary, max_tokens=512)
    timings["bedrock_s"] = round(time.perf_counter() - t0, 2)
    if report.get("source") != "bedrock" and not args.no_bedrock:
        from src.bedrock.reporter import template_report
        report["report"] = template_report(summary)
    timings["end_to_end_s"] = round(time.perf_counter() - t_all, 2)
    trace = {"run_id": run_id, "timestamp_utc": datetime.now(timezone.utc).isoformat(), "aws_region": region, "source": args.source, "s3_bucket_set": bool(bucket), "host": platform.node(),
             "git_commit": git_commit(), "decider_model": demo["decider_model"], "encoder": encoder, "encoder_hub": store.spec.hub, "encoder_license": store.spec.license, "encoder_dim": store.spec.dim, "classifier_artifact": clf_name, "bedrock_model_id": os.environ.get("BEDROCK_MODEL_ID", ""),
             "gpu": gpu, "budget": {"min": mn, "max": mx, "mode": args.mode}, "summary": summary, "report": report, "timings": timings,
             "decider_calls": len(dec_lat), "uni_calls": store.total_calls, "decider_latency_ms": {"mean": round(float(np.mean(dec_lat)), 1), "median": round(float(np.median(dec_lat)), 1)} if dec_lat else None,
             "uni_latency_ms": {"mean": round(float(np.mean(store.latency_ms)), 1), "median": round(float(np.median(store.latency_ms)), 1)},
             "acquired_patch_index": ep["acquired_patch_index"], "decision_history": ep["trace"]}
    out_dir = resolve(demo["runs_dir"]) / run_id; out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "trace.json").write_text(json.dumps(trace, indent=1, default=float))
    if bucket:
        import boto3
        boto3.client("s3").upload_file(str(out_dir / "trace.json"), bucket, f"{prefix}/runs/{run_id}/trace.json")
    print(json.dumps({k: trace[k] for k in ("run_id", "aws_region", "git_commit", "uni_calls", "decider_calls", "decider_latency_ms", "uni_latency_ms", "timings")}, indent=1))
    print(f"[agent] prediction: {summary['prediction']} | report_source = {report.get('source')}")
    print(f"[agent] report: {report.get('report', '')[:600]}")
    print(f"[agent] trace -> {out_dir/'trace.json'}")


if __name__ == "__main__":
    main()
