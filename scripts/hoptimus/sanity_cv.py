"""Sanity check of the retrained classifier head: 5x5 patient-level CV on the mean of the frozen Decider-Fixed-16
acquisitions embedded with the chosen encoder (same protocol as the benchmark classifier). This is a check that
the head generalises, NOT a benchmark re-run (acquisitions were made under the historical UNI run).
  python scripts/hoptimus/sanity_cv.py --config configs/tcga_task1.yaml --encoder hoptimus0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common.config import load_config, resolve  # noqa: E402
from src.evaluation.splits import make_folds  # noqa: E402
from src.tcga.evaluation import fit_predict_multi, multiclass_metrics  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/tcga_task1.yaml"); ap.add_argument("--base-config", default="configs/default.yaml")
    ap.add_argument("--encoder", default="hoptimus0"); ap.add_argument("--method", default="decider_fixed_16"); args = ap.parse_args()
    tcfg = load_config(args.config)["tcga"]; cfg = load_config(args.base_config); run = resolve(tcfg["results_dir"])
    meta = pd.read_csv(run / "manifest_snapshot.csv").sort_values("slide_id").reset_index(drop=True); y = meta.label.values.astype(int)
    X = np.stack([np.load(Path(cp).parent.parent / f"{args.encoder}_acquired_{args.method}" / f"{s}.npy").mean(0) for s, cp in zip(meta.slide_id, meta.cheap_path)])
    res = []
    for seed in cfg["evaluation"]["seeds"]:
        P = np.zeros((len(y), int(y.max() + 1)))
        for tr, te in make_folds(meta, int(cfg["evaluation"]["folds"]), seed):
            P[te] = fit_predict_multi(cfg["classifier"], X[tr], y[tr], X[te])
        res.append(multiclass_metrics(y, P))
    df = pd.DataFrame(res); out = {"encoder": args.encoder, "method": args.method, "n_slides": int(len(y)), "dim": int(X.shape[1]),
                                  **{k: {"mean": round(float(df[k].mean()), 4), "std": round(float(df[k].std()), 4)} for k in ("auroc", "balanced_accuracy", "f1", "accuracy")}}
    (run / f"sanity_cv_{args.encoder}_{args.method}.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
