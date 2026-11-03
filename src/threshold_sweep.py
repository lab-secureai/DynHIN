#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
threshold_sweep.py — Separates the advantage due to a THRESHOLD SHIFT from the part due to REPRESENTATION.

Question to answer
------------------
Compare `R-HGT Strict Control (No Fine-Tuning)` with `Proposed R-HGT` on clean
data (measured from raw, 10 seeds):

    F1       : 0.8974 -> 0.9997   (+0.1023)
    PR-AUC   : 0.9975 -> 1.0000   (+0.0025)     <-- does NOT depend on the threshold

The F1 gain is 41 times the PR-AUC gain. Since PR-AUC is invariant to the
threshold, most of the F1 gain is almost certainly an operating-point shift —
because the fine-tuning step is the ONLY place in the pipeline that uses class
weights. The signature of Strict Control (Precision 0.9985 but Recall 0.8185) is
exactly that of a model that ranks well but cuts at the wrong place.

This script measures it directly instead of speculating: for each model it finds
the optimal threshold and reports the F1 at that threshold.

Threshold-selection protocol (important)
----------------------------------------
Choosing the threshold on the test set and reporting on the same test set is
cheating. Here we use a NESTED SPLIT-HALF: split the test set into two stratified
halves, choose tau on half A and score it on half B and vice versa, then average.
The resulting number is an unbiased estimate of "the F1 achieved if the threshold
had been properly calibrated".

Usage
-----
    python3 threshold_sweep.py --dump-dir experiments_.../prob_dumps \\
                               --out-dir threshold_analysis

Input: .npz files produced by pipeline v2 (RHGT_SAVE_PROBS=1).
Requirements: numpy, pandas, scikit-learn.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    """ROC-AUC via the Mann–Whitney rank statistic (scikit-learn not needed)."""
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=np.float64)
    ranks[order] = np.arange(1, len(p) + 1, dtype=np.float64)
    # Average rank for tied values.
    ps = p[order]
    i = 0
    while i < len(ps):
        j = i
        while j + 1 < len(ps) and ps[j + 1] == ps[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def average_precision(y: np.ndarray, p: np.ndarray) -> float:
    """Average precision — sum weighted by the recall increments, no interpolation.

    Preferred over auc(recall, precision) because the trapezoidal rule on the
    PR curve has a known bias.
    """
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-p, kind="mergesort")
    ys = y[order]
    tp = np.cumsum(ys)
    precision = tp / np.arange(1, len(ys) + 1)
    return float((precision * ys).sum() / ys.sum())


def f1_at(y: np.ndarray, p: np.ndarray, tau: float) -> float:
    pred = (p >= tau).astype(np.int8)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0


def best_tau(y: np.ndarray, p: np.ndarray, grid: np.ndarray) -> Tuple[float, float]:
    scores = [f1_at(y, p, t) for t in grid]
    k = int(np.argmax(scores))
    return float(grid[k]), float(scores[k])


def stratified_halves(y: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    a, b = [], []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        cut = len(idx) // 2
        a.append(idx[:cut])
        b.append(idx[cut:])
    return np.concatenate(a), np.concatenate(b)


def analyse(path: str, grid: np.ndarray) -> Dict:
    z = np.load(path, allow_pickle=False)
    y = z["y_true"].astype(np.int8)
    prob = z["y_prob"]
    p = prob[:, 1] if (prob.ndim > 1 and prob.shape[1] > 1) else prob.ravel()
    regime = str(z["regime"][0]) if "regime" in z.files else ""

    m = re.match(r"clean_seed(\d+)_(.+)\.npz$", os.path.basename(path))
    seed = int(m.group(1)) if m else -1
    model = (m.group(2).replace("_", " ").strip() if m else os.path.basename(path))

    roc, ap = roc_auc(y, p), average_precision(y, p)

    # Nested split-half: choose tau on one half, score on the other.
    oof = []
    taus = []
    for rep in range(5):
        ia, ib = stratified_halves(y, seed=1000 * rep + max(seed, 0))
        ta, _ = best_tau(y[ia], p[ia], grid)
        tb, _ = best_tau(y[ib], p[ib], grid)
        oof.append(f1_at(y[ib], p[ib], ta))
        oof.append(f1_at(y[ia], p[ia], tb))
        taus += [ta, tb]

    tau_in, f1_in = best_tau(y, p, grid)      # optimistically biased, for reference only
    return {
        "Model": model, "Seed": seed, "Regime": regime, "n_test": int(y.size),
        "Prevalence": float(y.mean()),
        "F1 @ tau=0.5": f1_at(y, p, 0.5),
        "F1 @ tau* (out-of-fold)": float(np.mean(oof)),
        "F1 @ tau* (in-sample, biased)": f1_in,
        "tau* mean": float(np.mean(taus)),
        "ROC-AUC": roc, "Average Precision": ap,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Decomposition: threshold vs representation.")
    ap.add_argument("--dump-dir", required=True,
                    help="prob_dumps directory produced by pipeline v2.")
    ap.add_argument("--out-dir", default="threshold_analysis")
    ap.add_argument("--grid", type=int, default=1001)
    ap.add_argument("--pair", nargs=2,
                    default=["R-HGT Strict Control (No Fine-Tuning)",
                             "Multimodal + HIN R-HGT"],
                    help="The two models to decompose (matched approximately by name).")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dump_dir, "clean_seed*_*.npz")))
    if not files:
        sys.exit(f"No .npz file found in {args.dump_dir}.\n"
                 f"Run pipeline v2 with RHGT_SAVE_PROBS=1 first.")
    os.makedirs(args.out_dir, exist_ok=True)
    grid = np.linspace(0.0, 1.0, args.grid)

    print(f"Processing {len(files)} probability files ...")
    rows = [analyse(f, grid) for f in files]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, "threshold_sweep_raw.csv"), index=False)

    num = ["F1 @ tau=0.5", "F1 @ tau* (out-of-fold)",
           "F1 @ tau* (in-sample, biased)", "tau* mean",
           "ROC-AUC", "Average Precision"]
    summ = df.groupby("Model")[num].agg(["mean", "std"]).round(4)
    flat = pd.DataFrame(index=summ.index)
    for c in num:
        flat[c] = [f"{a:.4f} ± {(b if np.isfinite(b) else 0):.4f}"
                   for a, b in zip(summ[(c, "mean")], summ[(c, "std")])]
    flat["Gain from threshold calibration"] = (
        df.groupby("Model")["F1 @ tau* (out-of-fold)"].mean()
        - df.groupby("Model")["F1 @ tau=0.5"].mean()).round(4)
    flat = flat.reset_index().sort_values("Model")
    flat.to_csv(os.path.join(args.out_dir, "threshold_sweep_summary.csv"), index=False)

    print("\n=== F1 at the default threshold vs the calibrated threshold ===")
    print(flat.to_string(index=False))

    # ---- decomposition for the model pair of interest ----
    def match(name: str) -> str:
        key = "".join(ch for ch in name.lower() if ch.isalnum())
        best, score = None, -1
        for m in df["Model"].unique():
            k2 = "".join(ch for ch in m.lower() if ch.isalnum())
            s = len(os.path.commonprefix([key, k2]))
            if k2 in key or key in k2:
                s += 100
            if s > score:
                best, score = m, s
        return best

    a_name, b_name = match(args.pair[0]), match(args.pair[1])
    if a_name is None or b_name is None or a_name == b_name:
        print("\n[!] Could not identify the model pair to decompose.")
        return

    a = df[df["Model"] == a_name]
    b = df[df["Model"] == b_name]
    d_f1_default = b["F1 @ tau=0.5"].mean() - a["F1 @ tau=0.5"].mean()
    d_f1_tuned = (b["F1 @ tau* (out-of-fold)"].mean()
                  - a["F1 @ tau* (out-of-fold)"].mean())
    d_ap = b["Average Precision"].mean() - a["Average Precision"].mean()
    thr_part = d_f1_default - d_f1_tuned
    share = (thr_part / d_f1_default * 100) if abs(d_f1_default) > 1e-12 else float("nan")

    lines = [
        "",
        "=" * 74,
        "DECOMPOSITION: threshold or representation?",
        "=" * 74,
        f"  A (baseline) : {a_name}",
        f"  B (proposed) : {b_name}",
        "",
        f"  F1 difference at default threshold 0.5: {d_f1_default:+.4f}",
        f"  F1 difference with BOTH tau-calibrated: {d_f1_tuned:+.4f}   <-- true advantage",
        f"  Average Precision difference (tau-invariant): {d_ap:+.4f}",
        "",
        f"  Part due to threshold shift           : {thr_part:+.4f}  ({share:.1f}% of the original difference)",
        f"  Part due to better representation     : {d_f1_tuned:+.4f}  ({100 - share:.1f}%)",
        "",
    ]
    if np.isfinite(share) and share > 60:
        lines += [
            "  ⚠️  Most of the advantage comes from the OPERATING POINT, not from the",
            "      ability to separate classes. The paper must: (a) report both models at the",
            "      calibrated threshold, (b) state clearly that only the fine-tune branch uses class weights,",
            "      (c) lower the claim about the contribution of strict fine-tuning to match",
            "      the numbers above.",
        ]
    else:
        lines += [
            "  ✅ The advantage remains after both models are threshold-calibrated —",
            "      this is a genuine representation improvement.",
            "      Report the threshold-calibrated comparison alongside the default threshold.",
        ]

    text = "\n".join(lines)
    print(text)
    with open(os.path.join(args.out_dir, "DECOMPOSITION.md"), "w", encoding="utf-8") as f:
        f.write("# Decomposition: threshold vs representation\n\n```")
        f.write(text)
        f.write("\n```\n\n## Details by model\n\n")
        f.write(flat.to_string(index=False))
        f.write("\n")
    print(f"\n✅ Wrote {args.out_dir}/DECOMPOSITION.md")


if __name__ == "__main__":
    main()
