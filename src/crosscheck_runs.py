#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
crosscheck_runs.py — measures reproducibility between two machines running the SAME code with the SAME seed.

Background. When runs are executed in parallel on several Kaggle accounts, some
configurations end up running on more than one machine. merge_runs.py calls this
a "conflict" and forces a choice of one source — correctly, since a results
table must not hold two numbers for the same cell.

But the overlapping part is NOT garbage. It is something rarely available in a
paper: the same source code, the same seed, two independent machines. The
difference between the two sides is the reproducibility floor of the whole
experiment — the number that every published effect must exceed to be credible.

Why the difference is non-zero even with fixed seeds: seed_everything() sets
cudnn.deterministic=True, but that only constrains cuDNN's convolution kernels.
The scatter-add / index_add reductions used by every message-passing layer on
graphs accumulate with atomicAdd on the GPU, and the summation order depends on
block scheduling, so it is NOT deterministic; torch.use_deterministic_algorithms(True)
is not enabled (and with PyG it usually slows things down considerably or raises
an error about missing deterministic kernels). In addition, two Kaggle accounts
may be assigned different GPU types (T4 / P100 / L4), in which case even cuBLAS
selects different kernels. Both causes produce differences at the level of
accumulated floating-point error, not differences in method.

Usage:

    python3 crosscheck_runs.py --a m1/experiments --b m3/experiments \\
        --label-a m1 --label-b m3 --out crosscheck.md

Only keys present in BOTH sources are compared; the rest are ignored.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

KINDS = {
    "overall": ("robustness_global", "overall_robustness_raw_10seed.csv",
                ["Config", "Run", "Architecture", "Noise_Type", "Noise_Level"]),
    "perarch": ("robustness_per_architecture",
                "per_architecture_robustness_raw_10seed.csv",
                ["Config", "Run", "Architecture", "Noise_Type", "Noise_Level"]),
    "lofo": ("lofo_zeroday_generalization",
             "lofo_zeroday_generalization_raw_10seed.csv",
             ["Config", "Seed", "Held_Out_Family", "Architecture"]),
}
METRICS = ["F1", "Accuracy", "ROC-AUC", "Precision", "Recall"]


def find(root: str, sub: str, name: str):
    for p in (os.path.join(root, sub, name), os.path.join(root, name)):
        if os.path.isfile(p):
            return p
    return None


def gpu_of(root: str) -> str:
    for n in ("run_config.json", "run_env.json"):
        p = os.path.join(root, n)
        if os.path.isfile(p):
            try:
                return json.load(open(p, encoding="utf-8")).get("gpu", "?")
            except Exception:
                pass
    return "not recorded"


def compare(da: pd.DataFrame, db: pd.DataFrame, keys, la: str, lb: str):
    keys = [k for k in keys if k in da.columns and k in db.columns]
    ka = da[keys].astype(str).agg("|".join, axis=1)
    kb = db[keys].astype(str).agg("|".join, axis=1)
    da = da.assign(__k=ka).drop_duplicates("__k")
    db = db.assign(__k=kb).drop_duplicates("__k")
    both = sorted(set(da["__k"]) & set(db["__k"]))
    if not both:
        return None
    a = da.set_index("__k").loc[both]
    b = db.set_index("__k").loc[both]

    rows = []
    for m in METRICS:
        if m not in a.columns or m not in b.columns:
            continue
        d = (pd.to_numeric(a[m], errors="coerce")
             - pd.to_numeric(b[m], errors="coerce")).abs().dropna()
        if d.empty:
            continue
        rows.append({
            "Metric": m, "n keys": len(d),
            "mean |Δ|": d.mean(), "median |Δ|": d.median(),
            "p95 |Δ|": float(np.percentile(d, 95)), "max |Δ|": d.max(),
            "keys with |Δ|>0.01": int((d > 0.01).sum()),
        })
    out = pd.DataFrame(rows)

    # Standard deviation across seeds, for comparison: a difference between the
    # two machines is only worrying if it is of the same size as the inherent
    # variation between seeds.
    sd = np.nan
    if {"Config", "F1"} <= set(a.columns) and "Run" in a.columns:
        g = a.groupby("Config")["F1"].apply(
            lambda s: pd.to_numeric(s, errors="coerce").std(ddof=1))
        sd = float(np.nanmedian(g.values)) if len(g) else np.nan

    worst = None
    if "F1" in a.columns:
        d = (pd.to_numeric(a["F1"], errors="coerce")
             - pd.to_numeric(b["F1"], errors="coerce")).abs()
        worst = d.sort_values(ascending=False).head(10)
    return out, sd, worst, len(both)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--min-effect", type=float, default=0.0131,
                    help="SMALLEST effect the paper reports. Default "
                         "0.0131 = margin by which R-HGT beats Late Fusion on the DarkNexus family. "
                         "The reproducibility floor must be smaller than this number.")
    ap.add_argument("--out", default="crosscheck.md")
    args = ap.parse_args()

    L = []

    def say(s=""):
        print(s)
        L.append(s)

    say("# Cross-check between two machines")
    say()
    say(f"- `{args.label_a}` ← `{args.a}` — GPU: {gpu_of(args.a)}")
    say(f"- `{args.label_b}` ← `{args.b}` — GPU: {gpu_of(args.b)}")
    say()
    say("Only keys present in BOTH sources are compared: same source code, same "
        "seed, two independent machines.")
    say()

    floor = 0.0
    for kind, (sub, name, keys) in KINDS.items():
        pa, pb = find(args.a, sub, name), find(args.b, sub, name)
        if not pa or not pb:
            say(f"## {kind}\n\nOne of the two sources has no data — skipped.\n")
            continue
        r = compare(pd.read_csv(pa), pd.read_csv(pb), keys,
                    args.label_a, args.label_b)
        if r is None:
            say(f"## {kind}\n\nNo keys overlap — skipped.\n")
            continue
        out, sd, worst, n = r
        say(f"## {kind} — {n} overlapping keys")
        say()
        say(out.to_markdown(index=False, floatfmt=".6f"))
        say()
        if not np.isnan(sd):
            say(f"Standard deviation of F1 across seeds (median over configurations): "
                f"**{sd:.6f}**")
        mx = float(out.loc[out["Metric"] == "F1", "max |Δ|"].iloc[0]) \
            if (out["Metric"] == "F1").any() else 0.0
        floor = max(floor, mx)
        if not np.isnan(sd) and sd > 0:
            say(f"Ratio (largest difference between machines) / (sd across seeds): "
                f"**{mx / sd:.2f}×**")
        say()
        say("The 10 keys with the largest F1 difference:")
        say()
        say("```")
        for k, v in worst.items():
            say(f"  {v:.6f}  {k}")
        say("```")
        say()

    say("## Conclusion")
    say()
    say(f"Reproducibility floor between the two machines (max |ΔF1|): **{floor:.6f}**")
    say(f"Smallest effect reported in the paper: **{args.min_effect:.4f}**")
    say()
    if floor < args.min_effect:
        say(f"✅ The reproducibility floor is {args.min_effect / max(floor, 1e-12):.1f} "
            f"times smaller than the smallest reported effect. All conclusions in "
            f"the paper are robust to hardware variation.")
    else:
        say("❌ The reproducibility floor is NOT smaller than the smallest reported effect. "
            "Any conclusion thinner than this floor must be withdrawn, or everything "
            "must be re-run on a SINGLE machine.")
    say()
    say("Cause of the differences: scatter-add reductions on the GPU use "
        "atomicAdd, so the summation order is non-deterministic, and the two accounts may "
        "be assigned different GPU types. This is accumulated floating-point "
        "error, not a difference in method — but it must be MEASURED, "
        "not assumed.")

    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L) + "\n")
    print(f"\n✅ {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
