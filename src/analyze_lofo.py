#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_lofo.py — analysis of zero-day (leave-one-family-out) results.

Produces four tables for the paper, all using PAIRED tests by seed:

  L-A  Test-set size and trivial baseline inferred from the metrics themselves
  L-B  F1 per held-out family, with a "margin over the trivial baseline" column
  L-C  Paired comparison: proposed model vs each baseline (Holm-Bonferroni)
  L-D  Protocol gap under zero-day conditions

Reuses the pure-numpy statistics functions of analyze_phase1.py (no scipy needed).

    python3 analyze_lofo.py --lofo-dir <directory> --out-dir <directory>/analysis
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from analyze_phase1 import paired_t, wilcoxon_exact, holm      # noqa: E402

# Number of malicious samples per family in the whole dataset (dataset statistics table).
FAMILY_MALWARE_N = {"Mirai": 7368, "DarkNexus": 219, "Gafgyt": 60,
                    "Generic": 47, "Tsunami": 3}

PROPOSED = "R-HGT Stage 6 (+ Strict-Inductive Fine-Tuning)"


# --------------------------------------------------------------------------- #
# L-A: test-set size and trivial baseline
# --------------------------------------------------------------------------- #
def infer_counts(row, n_mal):
    """Infer n_benign from (Precision, Recall, Accuracy) and the known n_mal.

    TP = R·n_mal ; FN = n_mal − TP ; FP = TP/P − TP (P > 0)
    A·(n_mal + n_ben) = TP + TN = TP + (n_ben − FP)
      ⇒ n_ben = (TP − FP − A·n_mal) / (A − 1)
    """
    P, R, A = row["Precision"], row["Recall"], row["Accuracy"]
    if not (0 < P <= 1) or not np.isfinite(R) or not np.isfinite(A) or A >= 1:
        return np.nan
    tp = R * n_mal
    fp = tp / P - tp
    n_ben = (tp - fp - A * n_mal) / (A - 1.0)
    return n_ben


def table_A(d, out_dir):
    rows = []
    for fam, g in d.groupby("Held_Out_Family"):
        n_mal = FAMILY_MALWARE_N.get(fam)
        if n_mal is None:
            continue
        est = g.apply(lambda r: infer_counts(r, n_mal), axis=1)
        est = est[np.isfinite(est) & (est > 0)]
        n_ben = float(np.median(est)) if len(est) else np.nan
        n_tot = n_mal + n_ben
        prev = n_mal / n_tot
        rows.append({
            "Held_Out_Family": fam,
            "n_malware": n_mal,
            "n_benign_est": round(n_ben),
            "n_total": round(n_tot),
            "prevalence": prev,
            "trivial_F1": 2 * prev / (1 + prev),
            "trivial_Accuracy": prev,
            "n_benign_iqr": float(np.percentile(est, 75) - np.percentile(est, 25)) if len(est) else np.nan,
        })
    T = pd.DataFrame(rows).sort_values("n_total", ascending=False)
    T.round(6).to_csv(os.path.join(out_dir, "LA_test_set_sizes.csv"), index=False)
    return T


# --------------------------------------------------------------------------- #
# L-B: F1 per family + margin over the trivial baseline
# --------------------------------------------------------------------------- #
def table_B(d, TA, out_dir):
    triv = dict(zip(TA["Held_Out_Family"], TA["trivial_F1"]))
    g = (d.groupby(["Config", "Held_Out_Family"])
           .agg(F1=("F1", "mean"), F1_sd=("F1", "std"),
                Precision=("Precision", "mean"), Recall=("Recall", "mean"),
                AP=("PR-AUC", "mean"),
                ZeroDayRecall=("ZeroDay_Detection_Rate (Recall)", "mean"),
                n_seed=("Seed", "nunique"))
           .reset_index())
    g["trivial_F1"] = g["Held_Out_Family"].map(triv)
    g["margin_over_trivial"] = g["F1"] - g["trivial_F1"]
    g = g.sort_values(["Held_Out_Family", "F1"], ascending=[True, False])
    g.round(6).to_csv(os.path.join(out_dir, "LB_f1_by_family.csv"), index=False)
    return g


# --------------------------------------------------------------------------- #
# L-C: paired comparison of the proposed model vs baselines
# --------------------------------------------------------------------------- #
BASELINES = [
    "Multimodal Late Fusion (E2E)",
    "Multimodal Non-HIN GNN (Res-GraphSAGE)",
    "Multimodal + HIN R2-RGCN",
    "Multimodal + HIN HAN",
    "Multimodal + HIN RGCN",
    "Multimodal + HIN GAT",
    "Multimodal + HIN MAGNN",
    "Multimodal + HIN HGT",
    "R-HGT Strict Control (No Fine-Tuning)",
    "Branch 1 Only (PCAP MLP)",
]


def table_C(d, out_dir, proposed=PROPOSED):
    out = []
    for fam, g in d.groupby("Held_Out_Family"):
        w = g.pivot_table(index="Seed", columns="Config", values="F1")
        if proposed not in w.columns:
            continue
        rows, praw = [], []
        for b in BASELINES:
            if b not in w.columns:
                continue
            md, t, p, lo, hi, n = paired_t(w[proposed], w[b])
            _, pw = wilcoxon_exact(w[proposed], w[b])
            rows.append({"Held_Out_Family": fam, "baseline": b, "n_seed": n,
                         "F1_proposed": w[proposed].mean(), "F1_baseline": w[b].mean(),
                         "Delta_F1": md, "CI95_low": lo, "CI95_high": hi,
                         "t": t, "p_paired_t": p, "p_wilcoxon": pw})
            praw.append(p)
        if rows:
            sub = pd.DataFrame(rows)
            sub["p_holm"] = holm(praw)
            out.append(sub)
    T = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    if len(T):
        T = T.sort_values(["Held_Out_Family", "Delta_F1"], ascending=[True, False])
    T.round(6).to_csv(os.path.join(out_dir, "LC_proposed_vs_baselines.csv"), index=False)
    return T


# --------------------------------------------------------------------------- #
# L-D: protocol gap under zero-day conditions
# --------------------------------------------------------------------------- #
PROTOCOL_PAIRS_LOFO = [
    ("RGCN", "HIN RGCN [shared eval]", "Multimodal + HIN RGCN"),
    ("R2-RGCN", "HIN R2-RGCN [shared eval]", "Multimodal + HIN R2-RGCN"),
    ("MAGNN*", "HIN MAGNN [shared eval]", "Multimodal + HIN MAGNN"),
    ("HAN", "HIN HAN [shared eval]", "Multimodal + HIN HAN"),
    ("GAT", "HIN GAT [shared eval]", "Multimodal + HIN GAT"),
    ("HGT (vanilla)", "R-HGT Stage 1 (Vanilla HGT)", "Multimodal + HIN HGT"),
    ("R-HGT S2", "R-HGT Stage 2 (+ Residual + LayerNorm)", "R-HGT Stage 2 [strict eval]"),
    ("R-HGT S3", "R-HGT Stage 3 (+ Node Reliability)", "R-HGT Stage 3 [strict eval]"),
    ("R-HGT S4", "R-HGT Stage 4 (+ Relation Reliability / Gating)", "R-HGT Stage 4 [strict eval]"),
    ("R-HGT S5", "R-HGT Stage 5 (+ Robust Training)", "R-HGT Strict Control (No Fine-Tuning)"),
]


def table_D(d, out_dir):
    out = []
    for fam, g in d.groupby("Held_Out_Family"):
        w = g.pivot_table(index="Seed", columns="Config", values="F1")
        wa = g.pivot_table(index="Seed", columns="Config", values="PR-AUC")
        rows, praw = [], []
        for lab, cs, ci in PROTOCOL_PAIRS_LOFO:
            if cs not in w.columns or ci not in w.columns:
                continue
            md, t, p, lo, hi, n = paired_t(w[ci], w[cs])
            mda, *_ = paired_t(wa[ci], wa[cs])
            rows.append({"Held_Out_Family": fam, "Architecture": lab, "n_seed": n,
                         "F1_shared": w[cs].mean(), "F1_strict": w[ci].mean(),
                         "Delta_F1": md, "CI95_low": lo, "CI95_high": hi,
                         "t": t, "p_paired_t": p, "Delta_AP": mda})
            praw.append(p)
        if rows:
            sub = pd.DataFrame(rows)
            sub["p_holm"] = holm(praw)
            out.append(sub)
    T = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    if len(T):
        T = T.sort_values(["Held_Out_Family", "Delta_F1"])
    T.round(6).to_csv(os.path.join(out_dir, "LD_protocol_gap_lofo.csv"), index=False)
    return T


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lofo-dir", default=".")
    ap.add_argument("--out-dir", default="./analysis")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    pd.set_option("display.width", 260); pd.set_option("display.max_columns", 30)

    f = os.path.join(a.lofo_dir, "lofo_zeroday_generalization_raw_10seed.csv")
    d = pd.read_csv(f)
    print(f"Loaded {len(d)} rows · {d['Config'].nunique()} configurations · "
          f"{d['Seed'].nunique()} seed · families: {sorted(d['Held_Out_Family'].unique())}\n")

    print("=" * 110)
    print("L-A  TEST-SET SIZE AND TRIVIAL BASELINE (inferred from Precision/Recall/Accuracy)")
    print("=" * 110)
    TA = table_A(d, a.out_dir)
    print(TA.round(4).to_string(index=False))

    print("\n" + "=" * 110)
    print("L-B  F1 PER HELD-OUT FAMILY, WITH MARGIN OVER THE TRIVIAL BASELINE")
    print("=" * 110)
    TB = table_B(d, TA, a.out_dir)
    for fam in sorted(TB["Held_Out_Family"].unique()):
        s = TB[TB["Held_Out_Family"] == fam].head(12)
        print(f"\n— Held out {fam} (trivial baseline F1 = {s['trivial_F1'].iloc[0]:.4f}) —")
        print(s[["Config", "F1", "F1_sd", "Precision", "Recall",
                 "ZeroDayRecall", "AP", "margin_over_trivial"]]
              .round(4).to_string(index=False))

    print("\n" + "=" * 110)
    print("L-C  PROPOSED MODEL vs BASELINE (paired, Holm)")
    print("=" * 110)
    TC = table_C(d, a.out_dir)
    if len(TC):
        print(TC[["Held_Out_Family", "baseline", "F1_proposed", "F1_baseline",
                  "Delta_F1", "CI95_low", "CI95_high", "t", "p_holm"]]
              .round(4).to_string(index=False))

    print("\n" + "=" * 110)
    print("L-D  PROTOCOL GAP UNDER ZERO-DAY CONDITIONS")
    print("=" * 110)
    TD = table_D(d, a.out_dir)
    if len(TD):
        print(TD[["Held_Out_Family", "Architecture", "F1_shared", "F1_strict",
                  "Delta_F1", "CI95_low", "CI95_high", "Delta_AP", "p_holm"]]
              .round(4).to_string(index=False))

    print(f"\n✅ Wrote CSV files to {os.path.abspath(a.out_dir)}")


if __name__ == "__main__":
    main()
