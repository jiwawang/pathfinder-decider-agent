"""Render video frames from a REAL PathFinder run trace (scripts/aws/run_agent.py) over the demo slide thumbnail.
No inference is simulated: every box, number and probability comes from the saved trace.json.

  python scripts/submission/render_trace_frames.py --trace results/aws_smoke/<run_id>/trace.json \
      --demo-root <ARTIFACT_ROOT>/aws_demo/pathfinder-demo --out results/submission_frames/<run_id>
Outputs frame_00.png (candidates only) … frame_16.png (all acquisitions), plus final.png with the prediction,
and a contact sheet of the acquired raw patches (evidence_sheet.png).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

COLORS = {"largest_cluster": (31, 119, 180), "cheap_novelty": (255, 127, 14), "spatial_far": (44, 160, 44), "undercovered_region": (214, 39, 40)}


def font(size):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True); ap.add_argument("--demo-root", required=True); ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tr = json.load(open(args.trace)); sid = tr["summary"]["slide_id"]
    sd = Path(args.demo_root) / "slides" / sid; slide = json.load(open(sd / "slide.json")); cands = json.load(open(sd / "candidates.json"))
    thumb = Image.open(sd / "thumbnail.jpg").convert("RGB"); W, H = thumb.size
    # thumbnail scale: thumbnail long side 1024 over level-0 size; level-0 size is not in slide.json -> derive from cheap_features coords
    import numpy as np
    z = np.load(sd / "cheap_features.npz"); coords = z["coords"]; win = int(slide["window_px"])
    w0 = int(coords[:, 0].max() + win); h0 = int(coords[:, 1].max() + win); sc = min(W / w0, H / h0)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True); f14, f20, f26 = font(14), font(20), font(26)
    acq = [d for d in tr["decision_history"] if d.get("acquired_coord")]
    panel_w = 460; CH = max(H, 96 + 30 * len(acq) + 150); oy = (CH - H) // 2   # canvas tall enough for the full list
    def frame(k: int, final: bool = False):
        im = Image.new("RGB", (W + panel_w, CH), (18, 18, 22)); im.paste(thumb, (0, oy)); d = ImageDraw.Draw(im)
        for c in cands:   # all 128 cheap candidates: thin grey boxes
            x, y = c["x"] * sc, c["y"] * sc + oy; d.rectangle([x, y, x + win * sc, y + win * sc], outline=(170, 170, 170), width=1)
        for i, a in enumerate(acq[:k]):   # acquisitions so far: coloured by strategy, numbered
            x, y = a["acquired_coord"][0] * sc, a["acquired_coord"][1] * sc + oy; col = COLORS.get(a["selected_strategy"], (255, 255, 255))
            d.rectangle([x - 2, y - 2, x + win * sc + 2, y + win * sc + 2], outline=col, width=4); d.text((x + 3, y + 1), str(i + 1), fill=col, font=f14)
        px = W + 16; d.text((px, 14), "PathFinder - real run trace", fill=(255, 255, 255), font=f20)
        d.text((px, 44), f"slide {sid[:23]}  ({slide['n_tissue_patches']:,} tissue patches)", fill=(200, 200, 200), font=f14)
        d.text((px, 64), f"{len(cands)} cheap candidates (grey) - no encoder yet", fill=(200, 200, 200), font=f14)
        y = 96
        for i, a in enumerate(acq[:k]):
            col = COLORS.get(a["selected_strategy"], (255, 255, 255))
            d.text((px, y), f"{i+1:2d}. Decider -> {a['selected_option']} ({a['selected_strategy'].replace('_', ' ')}, conf {a['route_confidence']:.2f})", fill=col, font=f14)
            d.text((px + 20, y + 15), f"UNI call #{i+1}  patch {a['acquired_patch_index']}  info gain {a['information_gain']:.2f}", fill=(220, 220, 220), font=f14); y += 30
        H_ = CH
        d.text((px, H_ - 110), f"UNI calls: {k} / {slide['n_tissue_patches']:,}   Decider calls: {k}", fill=(255, 255, 255), font=f14)
        d.text((px, H_ - 90), f"encoder compute avoided: {100*(1-k/slide['n_tissue_patches']):.2f}%", fill=(255, 255, 255), font=f14)
        if final:
            p = tr["summary"]["prediction"]; d.text((px, H_ - 60), f"prediction: {p['predicted_class_name']}  p={max(p['probabilities'].values()):.2f}", fill=(255, 215, 0), font=f20)
            d.text((px, H_ - 32), f"loop {tr['timings']['acquisition_loop_s']} s  |  Decider {tr['decider_latency_ms']['median']:.0f} ms  |  UNI {tr['uni_latency_ms']['median']:.0f} ms", fill=(200, 200, 200), font=f14)
        return im
    for k in range(len(acq) + 1):
        frame(k).save(out / f"frame_{k:02d}.png")
    frame(len(acq), final=True).save(out / "final.png")
    # evidence sheet: the acquired raw patches in acquisition order
    n = len(acq); cols = 8; rows = (n + cols - 1) // cols; tile = 160
    sheet = Image.new("RGB", (cols * tile, rows * (tile + 22)), (18, 18, 22)); d = ImageDraw.Draw(sheet)
    byidx = {c["patch_index"]: c for c in cands}
    for i, a in enumerate(acq):
        im = Image.open(sd / byidx[a["acquired_patch_index"]]["png"]).convert("RGB").resize((tile, tile)); x, y = (i % cols) * tile, (i // cols) * (tile + 22)
        sheet.paste(im, (x, y)); d.text((x + 4, y + tile + 3), f"#{i+1} {a['selected_strategy'].replace('_', ' ')} g={a['information_gain']:.2f}", fill=(230, 230, 230), font=f14)
    sheet.save(out / "evidence_sheet.png")
    print(f"[frames] {len(acq)+1} frames + final.png + evidence_sheet.png -> {out}")


if __name__ == "__main__":
    main()
