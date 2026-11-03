#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_tables.py — Regenerates ALL result tables for the HIN-Dynamic paper
                 directly from the raw CSV files. No number is typed by hand.

    python3 make_tables.py --raw-dir raw --out-dir tables_generated

Inputs (required, in --raw-dir):
    overall_robustness_raw_10seed.csv
    per_architecture_robustness_raw_10seed.csv
    lofo_zeroday_generalization_raw_10seed.csv

Outputs (in --out-dir):
    csv/*.csv        — raw numbers of each table (for checking / plotting)
    md/*.md          — Markdown version to paste into the draft
    tex/*.tex        — LaTeX booktabs version to paste into the paper
    CONSISTENCY_REPORT.md — warnings detected automatically

Dependencies: pandas, numpy. scipy is NOT required (statistics are implemented here).

Design notes
----------------
1.  The script ONLY reads the raw files. Every table is derived, so tables and
    data cannot drift apart.
2.  COMBINED noise conditions (e.g. "PGD .20 + EdgeDrop .20") do NOT exist
    in the raw data: the pipeline never ran them. The script emits a warning
    instead of inventing numbers. See CONSISTENCY_REPORT.md.
3.  The LOFO test-set sizes are INFERRED from the quantisation of Recall/Accuracy,
    not hard-coded.
4.  The model parameter count is NOT in the raw data — use measure_params.py to measure it.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from itertools import product
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# =========================================================================
# 0. CONFIGURATION
# =========================================================================

PROPOSED = "Multimodal + HIN R-HGT"          # name of the proposed model in the raw data
STRICT_CONTROL = "R-HGT Strict Control (No Fine-Tuning)"

# Minimum sample size for a malware family to enter the main LOFO table.
MIN_FAMILY_POSITIVES = 200

# Canonical display names. The raw data uses two different naming sets: English
# in the robustness scenario and Vietnamese in the LOFO scenario — merged into one set.
CANONICAL_NAME: Dict[str, str] = {
    # --- single-modality / fusion baselines ---
    "Branch 1: PCAP MLP": "PCAP MLP",
    "Branch 1 Only (PCAP MLP)": "PCAP MLP",
    "Branch 2: SAR Encoder": "SAR Encoder",
    "Branch 2 Only (SAR Encoder)": "SAR Encoder",
    "Branch 3: STRACE CNN-LSTM": "STRACE CNN-LSTM",
    "Branch 3 Only (STRACE CNN-LSTM)": "STRACE CNN-LSTM",
    "Multimodal Late Fusion": "Multimodal Late Fusion",
    "Multimodal Late Fusion (E2E)": "Multimodal Late Fusion",
    "Multimodal Non-HIN GNN": "Res-GraphSAGE (non-HIN)",
    "Multimodal Non-HIN GNN (Res-GraphSAGE)": "Res-GraphSAGE (non-HIN)",
    # --- external HIN baselines ---
    "Multimodal + HIN RGCN": "HIN-RGCN",
    # [v3] R2-RGCN uses the three R-HGT mechanisms VERBATIM (node reliability,
    # residual gate, LayerNorm) — only the message-passing core differs. This is a
    # core-replacement ablation, not an external baseline; the display name reflects that.
    "Multimodal + HIN R2-RGCN": "R-HGT (RGCN backbone)",
    "Multimodal + HIN MAGNN": "HIN-MAGNN",
    "Multimodal + HIN HAN": "HIN-HAN",
    "Multimodal + HIN GAT": "HIN-GAT",
    # [v3] now an alias of R-HGT Stage 1 — same checkpoint, strict mode.
    "Multimodal + HIN HGT": "HIN-HGT (= R-HGT S1, strict)",
    # --- ablation chain ---
    "R-HGT Stage 1 (Vanilla HGT)": "R-HGT S1 (vanilla HGT)",
    "R-HGT Stage 2 (+ Residual + LayerNorm)": "R-HGT S2 (+res+LN)",
    "R-HGT Stage 3 (+ Node Reliability)": "R-HGT S3 (+node rel.)",
    "R-HGT Stage 4 (+ Relation Reliability / Gating)": "R-HGT S4 (+rel. gating)",
    "R-HGT Stage 5 (+ Robust Training)": "R-HGT S5 (+adv. training)",
    "R-HGT Stage 6 (+ Strict-Inductive Fine-Tuning)": "R-HGT S6 (+strict FT)",
    STRICT_CONTROL: "R-HGT strict control (no FT)",
    PROPOSED: "Proposed R-HGT",
    # --- group A: eval-mode aliases (v3) ---
    "HIN RGCN [shared eval]": "HIN-RGCN [shared]",
    "HIN R2-RGCN [shared eval]": "R-HGT (RGCN backbone) [shared]",
    "HIN MAGNN [shared eval]": "HIN-MAGNN [shared]",
    "HIN HAN [shared eval]": "HIN-HAN [shared]",
    "HIN GAT [shared eval]": "HIN-GAT [shared]",
    "R-HGT Stage 2 [strict eval]": "R-HGT S2 [strict]",
    "R-HGT Stage 3 [strict eval]": "R-HGT S3 [strict]",
    "R-HGT Stage 4 [strict eval]": "R-HGT S4 [strict]",
    # --- group B: baselines given the full pipeline of the proposed model (v3) ---
    "HIN RGCN (robust)": "HIN-RGCN +adv",
    "HIN RGCN (robust + strict FT)": "HIN-RGCN +adv +strictFT",
    "HIN R2-RGCN (robust)": "R-HGT(RGCN) +adv",
    "HIN R2-RGCN (robust + strict FT)": "R-HGT(RGCN) +adv +strictFT",
    "HIN MAGNN (robust)": "HIN-MAGNN +adv",
    "HIN MAGNN (robust + strict FT)": "HIN-MAGNN +adv +strictFT",
    "HIN HAN (robust)": "HIN-HAN +adv",
    "HIN HAN (robust + strict FT)": "HIN-HAN +adv +strictFT",
    "HIN GAT (robust)": "HIN-GAT +adv",
    "HIN GAT (robust + strict FT)": "HIN-GAT +adv +strictFT",
    "HIN HGT (robust)": "HIN-HGT +adv",
    "HIN HGT (robust + strict FT)": "HIN-HGT +adv +strictFT",
    # --- group C: 2x2 ablation separating class weighting from strict FT (v3) ---
    "R-HGT Stage 6a (FT, no class weight)": "R-HGT S6a (FT, no class weight)",
    "R-HGT Stage 5b (class-weighted, no FT)": "R-HGT S5b (class weight, no FT)",
}

# Configurations that SHARE weights (aliases in hin_model_specs). Their numbers are
# identical — the script warns and suggests merging them when presenting.
KNOWN_WEIGHT_ALIASES: List[Tuple[str, str]] = [
    (PROPOSED, "R-HGT Stage 6 (+ Strict-Inductive Fine-Tuning)"),
    (STRICT_CONTROL, "R-HGT Stage 5 (+ Robust Training)"),
    # [v3] Multimodal + HIN HGT is now an alias of Stage 1 (same checkpoint,
    # different eval mode) — no longer two independent training runs.
    ("Multimodal + HIN HGT", "R-HGT Stage 1 (Vanilla HGT)"),
]

# Order of appearance in the ablation chain.
ABLATION_ORDER = [
    "R-HGT Stage 1 (Vanilla HGT)",
    "R-HGT Stage 2 (+ Residual + LayerNorm)",
    "R-HGT Stage 3 (+ Node Reliability)",
    "R-HGT Stage 4 (+ Relation Reliability / Gating)",
    "R-HGT Stage 5 (+ Robust Training)",
    STRICT_CONTROL,
    "R-HGT Stage 6 (+ Strict-Inductive Fine-Tuning)",
]

# Inference mode -> presentation label.
REGIME_LABEL = {
    "strict_isolated_ego": "strict",
    "shared_entity_ablation": "shared",
    "tabular_inductive": "tabular",
    "self_loop_inductive": "self-loop",
}

METRICS = ["Accuracy", "Precision", "Recall", "Specificity",
           "F1", "MCC", "ROC-AUC", "PR-AUC"]

WARNINGS: List[str] = []


def warn(msg: str) -> None:
    WARNINGS.append(msg)
    print(f"  [!] {msg}", file=sys.stderr)


# =========================================================================
# 1. PURE-PYTHON STATISTICS (no scipy needed)
# =========================================================================

def _betacf(a: float, b: float, x: float, itmax: int = 200, eps: float = 3e-16) -> float:
    """Continued fraction for the incomplete beta function (Numerical Recipes)."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < 1e-300:
        d = 1e-300
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log1p(-x))
    front = math.exp(lbeta)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_sf_two_sided(t: float, df: int) -> float:
    """Two-sided p-value of the Student-t distribution."""
    if df <= 0 or not math.isfinite(t):
        return float("nan")
    return _betai(df / 2.0, 0.5, df / (df + t * t))


def paired_t(a: Sequence[float], b: Sequence[float]) -> dict:
    """Paired t-test + Cohen's d_z + 95% CI of the mean difference."""
    d = np.asarray(a, float) - np.asarray(b, float)
    n = d.size
    md = float(d.mean())
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    if sd == 0.0:
        return dict(n=n, mean_diff=md, t=float("inf") if md else 0.0,
                    p=0.0 if md else 1.0, dz=float("inf") if md else 0.0,
                    ci_low=md, ci_high=md)
    se = sd / math.sqrt(n)
    t = md / se
    # Two-sided 5% critical t values for small df; for df>30 use the normal approximation.
    tcrit = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
             7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
             13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110,
             18: 2.101, 19: 2.093, 20: 2.086}.get(n - 1, 1.960)
    return dict(n=n, mean_diff=md, t=t, p=t_sf_two_sided(t, n - 1),
                dz=md / sd, ci_low=md - tcrit * se, ci_high=md + tcrit * se)


def wilcoxon_exact(a: Sequence[float], b: Sequence[float]) -> float:
    """Wilcoxon signed-rank test, EXACT two-sided p by enumeration (n <= 20).

    Suitable for n = 10 seeds. Pairs with zero difference are dropped (Wilcoxon convention).
    """
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[d != 0]
    n = d.size
    if n == 0:
        return 1.0
    if n > 20:
        return float("nan")
    ranks = pd.Series(np.abs(d)).rank().to_numpy()
    w_obs = float(ranks[d > 0].sum())
    total = float(ranks.sum())
    w_obs = min(w_obs, total - w_obs)
    # Null distribution: all sign combinations are equally likely.
    count = 0
    for signs in product([0.0, 1.0], repeat=n):
        w = float(np.dot(ranks, signs))
        if min(w, total - w) <= w_obs + 1e-9:
            count += 1
    return min(1.0, count / (2.0 ** n))


def holm_bonferroni(pvals: Sequence[float], alpha: float = 0.05) -> Tuple[List[float], List[bool]]:
    """Holm–Bonferroni multiple-comparison correction (step-down)."""
    p = np.asarray(pvals, float)
    m = p.size
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for i, idx in enumerate(order):
        val = (m - i) * p[idx]
        running = max(running, val)
        adj[idx] = min(1.0, running)
    return adj.tolist(), (adj <= alpha).tolist()


# =========================================================================
# 2. INFER TEST-SET SIZES FROM THE QUANTISATION OF THE METRICS
# =========================================================================

def infer_denominator(values: np.ndarray, max_n: int = 30000,
                      tol: float = 1e-3) -> Optional[int]:
    """Find the smallest n such that every value is approximately k/n.

    Recall = TP/(TP+FN), so the denominator is the number of POSITIVE test samples.
    Accuracy = (TP+TN)/N, so the denominator is the TOTAL number of test samples.
    """
    v = np.unique(values[np.isfinite(values)])
    v = v[(v >= 0.0) & (v <= 1.0)]
    if v.size == 0:
        return None
    # Drop 0 and 1: they match every n and pollute the search.
    v_int = v[(v > 1e-12) & (v < 1 - 1e-12)]
    if v_int.size == 0:
        return None
    for n in range(2, max_n + 1):
        scaled = v_int * n
        if np.max(np.abs(scaled - np.round(scaled))) < tol:
            return n
    return None


def infer_lofo_test_sizes(lofo: pd.DataFrame) -> pd.DataFrame:
    """Infer (n_positive, n_total) for each retained malware family.

    The number of test malware samples of a family is FIXED (all samples of that family), but the
    number of benign test samples = min(n_malware, |benign_te|), and |benign_te| varies with the
    seed because of the group split. Hence n_total must be inferred PER SEED and only then
    aggregated; otherwise no common denominator exists.
    """
    rows = []
    for fam, sub in lofo.groupby("Held_Out_Family"):
        n_pos = infer_denominator(sub["Recall"].to_numpy())

        totals = []
        for _, per_seed in sub.groupby("Seed"):
            t = infer_denominator(per_seed["Accuracy"].to_numpy())
            if t:
                totals.append(t)
        n_tot = int(round(float(np.median(totals)))) if totals else None
        tot_min = min(totals) if totals else None
        tot_max = max(totals) if totals else None

        if n_pos and n_tot and n_tot < n_pos:
            warn(f"LOFO/{fam}: inferred n_total ({n_tot}) < n_positive ({n_pos}) — "
                 f"check manually.")
        prevalence = (n_pos / n_tot) if (n_pos and n_tot) else float("nan")
        rows.append({
            "Held_Out_Family": fam,
            "n_positive": n_pos,
            "n_total": n_tot,
            "n_total_range": (f"{tot_min}–{tot_max}" if totals and tot_min != tot_max
                              else (str(tot_min) if totals else "?")),
            "n_negative": (n_tot - n_pos) if (n_pos and n_tot) else None,
            "prevalence": prevalence,
            # "Always predict malware" classifier: P = prevalence, R = 1
            "F1_trivial_all_positive": (2 * prevalence / (1 + prevalence)
                                        if np.isfinite(prevalence) else float("nan")),
            "n_distinct_recall": int(np.unique(sub["Recall"].to_numpy()).size),
            "sufficient_power": bool(n_pos and n_pos >= MIN_FAMILY_POSITIVES),
        })
    out = pd.DataFrame(rows).sort_values("n_positive", ascending=False)
    for _, r in out.iterrows():
        if not r["sufficient_power"]:
            warn(f"LOFO/{r['Held_Out_Family']}: only {r['n_positive']} positive samples "
                 f"({r['n_distinct_recall']} distinct Recall values) — "
                 f"below the threshold {MIN_FAMILY_POSITIVES}, NOT statistically powered.")
    return out


# =========================================================================
# 3. LOAD AND NORMALISE THE DATA
# =========================================================================

REGIME_BY_CANON = {
    "PCAP MLP": "tabular_inductive",
    "SAR Encoder": "tabular_inductive",
    "STRACE CNN-LSTM": "tabular_inductive",
    "Multimodal Late Fusion": "tabular_inductive",
    "Res-GraphSAGE (non-HIN)": "self_loop_inductive",
}


def load(raw_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = {
        "overall": os.path.join(raw_dir, "overall_robustness_raw_10seed.csv"),
        "perarch": os.path.join(raw_dir, "per_architecture_robustness_raw_10seed.csv"),
        "lofo": os.path.join(raw_dir, "lofo_zeroday_generalization_raw_10seed.csv"),
    }
    for k, p in paths.items():
        if not os.path.exists(p):
            sys.exit(f"ERROR: file not found: {p}")
    ov = pd.read_csv(paths["overall"])
    pa = pd.read_csv(paths["perarch"])
    lo = pd.read_csv(paths["lofo"])

    for name, df in (("overall", ov), ("per-arch", pa), ("lofo", lo)):
        unknown = sorted(set(df["Config"]) - set(CANONICAL_NAME))
        if unknown:
            warn(f"{name}: configuration(s) missing from CANONICAL_NAME: {unknown}")
        df["Method"] = df["Config"].map(CANONICAL_NAME).fillna(df["Config"])

    # LOFO lacks Inference_Regime for the 5 non-HIN baselines. This is exactly the bug
    # that made groupby(dropna=True) drop them from the old summary table.
    if "Inference_Regime" in lo.columns:
        missing = lo["Inference_Regime"].isna()
        if missing.any():
            affected = sorted(lo.loc[missing, "Method"].unique())
            warn(f"LOFO: {int(missing.sum())} rows lack Inference_Regime "
                 f"({affected}). Filled automatically — the old summary table SILENTLY "
                 f"dropped these baselines (groupby defaults to dropna=True).")
            lo.loc[missing, "Inference_Regime"] = (
                lo.loc[missing, "Method"].map(REGIME_BY_CANON).fillna("tabular_inductive"))
    else:
        lo["Inference_Regime"] = lo["Method"].map(REGIME_BY_CANON).fillna("unknown")

    if "Seed" not in lo.columns:
        lo["Seed"] = lo["Run"].astype(str).str.extract(r"S(\d+)_").astype(float)
    for df in (ov, pa):
        if "Seed" not in df.columns:
            df["Seed"] = df["Run"]

    return ov, pa, lo


# =========================================================================
# 4. FORMATTING / EXPORT UTILITIES
# =========================================================================

def ms(series: pd.Series, nd: int = 4) -> str:
    """'mean ± std' with the given number of digits."""
    m, s = series.mean(), series.std(ddof=1)
    if not np.isfinite(m):
        return "—"
    if not np.isfinite(s):
        s = 0.0
    return f"{m:.{nd}f} ± {s:.{nd}f}"


def agg_mean_std(df: pd.DataFrame, group: List[str], metrics: List[str],
                 nd: int = 4) -> pd.DataFrame:
    g = df.groupby(group, dropna=False)
    out = g[metrics].agg(["mean", "std"])
    flat = pd.DataFrame(index=out.index)
    for m in metrics:
        flat[m] = [f"{a:.{nd}f} ± {(b if np.isfinite(b) else 0.0):.{nd}f}"
                   for a, b in zip(out[(m, "mean")], out[(m, "std")])]
    flat["n_seeds"] = g.size()
    return flat.reset_index()


LATEX_ESC = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
             "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
             "^": r"\textasciicircum{}", "±": r"$\pm$", "→": r"$\to$",
             "↑": r"$\uparrow$", "—": "---"}


def tex_escape(s: str) -> str:
    out = []
    for ch in str(s):
        out.append(LATEX_ESC.get(ch, ch))
    return "".join(out)


def to_latex(df: pd.DataFrame, caption: str, label: str) -> str:
    cols = list(df.columns)
    align = "l" + "r" * (len(cols) - 1)
    lines = [r"\begin{table}[htbp]", r"\centering",
             rf"\caption{{{tex_escape(caption)}}}", rf"\label{{{label}}}",
             r"\resizebox{\textwidth}{!}{%",
             rf"\begin{{tabular}}{{{align}}}", r"\toprule",
             " & ".join(tex_escape(c) for c in cols) + r" \\", r"\midrule"]
    for _, row in df.iterrows():
        lines.append(" & ".join(tex_escape(v) for v in row.tolist()) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def df_to_markdown(df: pd.DataFrame) -> str:
    """Markdown table without depending on the `tabulate` package."""
    cols = [str(c) for c in df.columns]
    rows = [[("" if (v is None or (isinstance(v, float) and not np.isfinite(v)))
              else str(v)) for v in r] for r in df.astype(object).values]
    widths = [max(len(cols[i]), *(len(r[i]) for r in rows)) if rows else len(cols[i])
              for i in range(len(cols))]
    out = ["| " + " | ".join(c.ljust(w) for c, w in zip(cols, widths)) + " |",
           "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    for r in rows:
        out.append("| " + " | ".join(v.ljust(w) for v, w in zip(r, widths)) + " |")
    return "\n".join(out)


class Emitter:
    """Write each table in three formats and keep an index."""

    def __init__(self, out_dir: str):
        self.out = out_dir
        for sub in ("csv", "md", "tex"):
            os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
        self.index: List[Tuple[str, str, str]] = []

    def emit(self, key: str, df: pd.DataFrame, caption: str,
             label: Optional[str] = None) -> None:
        label = label or f"tab:{key}"
        df = df.copy()
        df.to_csv(os.path.join(self.out, "csv", f"{key}.csv"), index=False)
        with open(os.path.join(self.out, "md", f"{key}.md"), "w", encoding="utf-8") as f:
            f.write(f"### {caption}\n\n")
            f.write(df_to_markdown(df))
            f.write("\n")
        with open(os.path.join(self.out, "tex", f"{key}.tex"), "w", encoding="utf-8") as f:
            f.write(to_latex(df, caption, label))
        self.index.append((key, caption, label))
        print(f"  ✓ {key:34s} {df.shape[0]:>3d} rows × {df.shape[1]:>2d} cols")


# =========================================================================
# 5. THE TABLES
# =========================================================================

def table_overall_clean(ov: pd.DataFrame, em: Emitter) -> None:
    """Tables IV/V — performance on clean data.

    Separates the two inference modes. Configurations run in 'shared' mode are NOT
    directly comparable with 'strict' mode: shared mode still allows second-order
    test-to-test propagation through shared entity nodes.
    """
    c = ov[ov["Noise_Type"] == "Clean"].copy()
    c["Regime"] = c["Inference_Regime"].map(REGIME_LABEL).fillna(c["Inference_Regime"])

    full = agg_mean_std(c, ["Method", "Regime"], METRICS)
    full = full.sort_values("F1", ascending=False)
    em.emit("T04_overall_clean_all_regimes", full,
            "Detection performance on clean data, all configurations "
            "(mean ± std over 10 seeds). The Regime column gives the inference "
            "mode; 'shared' still has test-to-test propagation.")

    strict = c[c["Inference_Regime"] != "shared_entity_ablation"]
    tbl = agg_mean_std(strict, ["Method", "Regime"], METRICS)
    tbl = tbl.sort_values("F1", ascending=False)
    em.emit("T05_overall_clean_strict_only", tbl,
            "Performance on clean data, ONLY strict-inductive / tabular "
            "inference configurations (shared_entity_ablation excluded). "
            "This is the fair comparison table.")


def table_protocol_gap(ov: pd.DataFrame, em: Emitter) -> None:
    """Evaluation-protocol gap table — the main proposed claim.

    Compares the same checkpoint under two evaluation modes.
    """
    c = ov[ov["Noise_Type"] == "Clean"]
    piv = c.groupby(["Config", "Seed"])["F1"].mean().unstack(0)

    rows = []
    pairs = [("R-HGT Stage 5 (+ Robust Training)", STRICT_CONTROL,
              "R-HGT Stage 5 — same checkpoint, two protocols")]
    for shared_cfg, strict_cfg, note in pairs:
        if shared_cfg not in piv.columns or strict_cfg not in piv.columns:
            continue
        st = paired_t(piv[strict_cfg], piv[shared_cfg])
        rows.append({
            "Checkpoint": note,
            "F1 (shared eval)": f"{piv[shared_cfg].mean():.4f} ± {piv[shared_cfg].std(ddof=1):.4f}",
            "F1 (strict eval)": f"{piv[strict_cfg].mean():.4f} ± {piv[strict_cfg].std(ddof=1):.4f}",
            "Gap": f"{st['mean_diff']:+.4f}",
            "CI 95%": f"[{st['ci_low']:+.4f}, {st['ci_high']:+.4f}]",
        })

    # All HIN baselines under strict mode: shows the Recall drop.
    hin_strict = c[(c["Inference_Regime"] == "strict_isolated_ego")]
    extra = (hin_strict.groupby("Method")[["F1", "Precision", "Recall"]]
             .mean().round(4).reset_index().sort_values("F1", ascending=False))
    extra.columns = ["Method", "F1 (strict)", "Precision (strict)", "Recall (strict)"]

    if rows:
        em.emit("T06a_protocol_gap", pd.DataFrame(rows),
                "Evaluation-protocol gap: the same checkpoint is "
                "evaluated under shared-entity and strict-inductive modes.")
    em.emit("T06b_hin_baselines_under_strict", extra,
            "All HIN baselines under strict-inductive inference. "
            "Recall drops sharply when test-to-test propagation is blocked.")


def table_robustness(ov: pd.DataFrame, em: Emitter) -> None:
    """Table VI — F1 at the strongest noise/attack level of each type."""
    worst = {"Gaussian_Noise": 0.5, "Feature_Masking": 0.5,
             "Edge_Perturbation": 0.5, "FGSM_Attack": 0.2,
             "BIM_Attack": 0.2, "PGD_Attack": 0.2}
    label = {"Gaussian_Noise": "Gaussian σ=0.5", "Feature_Masking": "Mask p=0.5",
             "Edge_Perturbation": "EdgeDrop p=0.5", "FGSM_Attack": "FGSM ε=0.2",
             "BIM_Attack": "BIM ε=0.2", "PGD_Attack": "PGD ε=0.2"}

    out = {}
    clean = ov[ov["Noise_Type"] == "Clean"].groupby("Method")["F1"]
    out["Clean"] = {m: ms(g) for m, g in clean}
    for nt, lvl in worst.items():
        sub = ov[(ov["Noise_Type"] == nt) & (np.isclose(ov["Noise_Level"], lvl))]
        out[label[nt]] = {m: ms(g) for m, g in sub.groupby("Method")["F1"]}

    tbl = pd.DataFrame(out)
    tbl.index.name = "Method"
    tbl = tbl.fillna("—").reset_index()
    order = (ov[ov["Noise_Type"] == "Clean"].groupby("Method")["F1"].mean()
             .sort_values(ascending=False).index)
    tbl["__o"] = tbl["Method"].map({m: i for i, m in enumerate(order)})
    tbl = tbl.sort_values("__o").drop(columns="__o")
    em.emit("T07_robustness_strongest", tbl,
            "F1 at the strongest noise / attack level of each type "
            "(mean ± std over 10 seeds). '—' = not applicable "
            "(the model does not use the graph).")

    # Full curves, for plotting.
    curve = (ov.groupby(["Method", "Noise_Type", "Noise_Level"])["F1"]
             .agg(["mean", "std", "count"]).reset_index()
             .rename(columns={"mean": "F1_mean", "std": "F1_std", "count": "n_seeds"}))
    em.emit("T07b_robustness_curves_longform", curve.round(6),
            "Long form of all robustness curves — input "
            "for Figures 4 and 5.")


def table_lofo(lofo: pd.DataFrame, sizes: pd.DataFrame, em: Emitter) -> None:
    """Table VII — zero-day generalisation (LOFO), with sample sizes and a trivial baseline."""
    em.emit("T08a_lofo_test_set_sizes",
            sizes.assign(
                prevalence=sizes["prevalence"].round(4),
                F1_trivial_all_positive=sizes["F1_trivial_all_positive"].round(4))
            .rename(columns={
                "Held_Out_Family": "Held-out family", "n_positive": "Test malware",
                "n_negative": "Benign test", "n_total": "Test total (median)",
                "n_total_range": "Test total (range over 10 seeds)",
                "prevalence": "Prevalence", "n_distinct_recall": "Distinct Recall values",
                "F1_trivial_all_positive": "Trivial F1",
                "sufficient_power": "Sufficient power"}),
            "LOFO test-set size per family, INFERRED from the quantisation of "
            "Recall and Accuracy. 'Trivial F1' is the F1 of the classifier that "
            "always predicts malware: 2p/(1+p).")

    keep = set(sizes.loc[sizes["sufficient_power"], "Held_Out_Family"])
    dropped = sorted(set(sizes["Held_Out_Family"]) - keep)

    n_by_fam = dict(zip(sizes["Held_Out_Family"], sizes["n_positive"]))
    per_fam = lofo.pivot_table(index="Method", columns="Held_Out_Family",
                               values="F1", aggfunc="mean").round(4)

    def _weighted(row: pd.Series, fams: set) -> float:
        num = den = 0.0
        for f in fams:
            n = n_by_fam.get(f) or 0
            if f in row.index and np.isfinite(row[f]):
                num += row[f] * n
                den += n
        return num / den if den else float("nan")

    main = per_fam[sorted(keep)].copy()
    main["Mean F1 (sufficient power)"] = main[sorted(keep)].mean(axis=1).round(4)
    main["Weighted F1"] = per_fam.apply(lambda r: _weighted(r, keep), axis=1).round(4)
    main["Worst F1 (sufficient power)"] = main[sorted(keep)].min(axis=1).round(4)
    rec = lofo.groupby("Method")["ZeroDay_Detection_Rate (Recall)"].mean().round(4)
    main["Zero-day Recall"] = rec
    main = main.sort_values("Weighted F1", ascending=False).reset_index()
    em.emit("T08b_lofo_main", main,
            f"Zero-day generalisation (LOFO) — ONLY families with ≥ "
            f"{MIN_FAMILY_POSITIVES} positive samples. Families {dropped} are excluded because "
            f"they lack statistical power (see T08a).")

    full = per_fam.copy()
    full["Mean F1 (all families)"] = per_fam.mean(axis=1).round(4)
    full["Worst F1 (all families)"] = per_fam.min(axis=1).round(4)
    full = full.sort_values("Mean F1 (all families)", ascending=False).reset_index()
    em.emit("T08c_lofo_all_families_APPENDIX", full,
            "APPENDIX — LOFO over all 5 families. WARNING: families "
            + ", ".join(f"{f} (n={n_by_fam.get(f)})" for f in dropped)
            + " have too few samples; those columns are sampling noise, not "
              "results. Do not use for the main table.")

    # Direct comparison with the trivial baseline.
    rows = []
    for fam in sorted(per_fam.columns):
        triv = float(sizes.loc[sizes["Held_Out_Family"] == fam,
                               "F1_trivial_all_positive"].iloc[0])
        n = n_by_fam.get(fam)
        prop = per_fam.loc[CANONICAL_NAME[PROPOSED], fam] if CANONICAL_NAME[PROPOSED] in per_fam.index else np.nan
        rows.append({"Held-out family": fam, "n malware": n,
                     "Trivial F1": round(triv, 4),
                     "F1 Proposed R-HGT": round(float(prop), 4),
                     "Margin": round(float(prop) - triv, 4),
                     "Sufficient power": "yes" if fam in keep else "NO"})
    em.emit("T08d_lofo_vs_trivial", pd.DataFrame(rows),
            "Proposed model versus the trivial 'always predict malware' classifier "
            "per family. This is the real margin.")


def table_ablation(ov: pd.DataFrame, lofo: pd.DataFrame,
                   sizes: pd.DataFrame, em: Emitter) -> None:
    """Table VIII — 6-stage ablation, annotated with the evaluation mode."""
    keep = set(sizes.loc[sizes["sufficient_power"], "Held_Out_Family"])
    lo_ok = lofo[lofo["Held_Out_Family"].isin(keep)]

    regime = (ov[ov["Noise_Type"] == "Clean"]
              .groupby("Config")["Inference_Regime"].first())

    rows = []
    for cfg in ABLATION_ORDER:
        sub = ov[ov["Config"] == cfg]
        if sub.empty:
            continue

        def pick(nt, lvl):
            s = sub[(sub["Noise_Type"] == nt) & (np.isclose(sub["Noise_Level"], lvl))]["F1"]
            return ms(s, 3) if len(s) else "—"

        lo_sub = lo_ok[lo_ok["Config"] == cfg]
        lo_fam = lo_sub.groupby("Held_Out_Family")["F1"].mean()
        rows.append({
            "Stage": CANONICAL_NAME.get(cfg, cfg),
            "Eval mode": REGIME_LABEL.get(regime.get(cfg, ""), "?"),
            "Clean F1": ms(sub[sub["Noise_Type"] == "Clean"]["F1"], 3),
            "FGSM 0.2": pick("FGSM_Attack", 0.2),
            "PGD 0.2": pick("PGD_Attack", 0.2),
            "EdgeDrop 0.5": pick("Edge_Perturbation", 0.5),
            "LOFO F1": f"{lo_fam.mean():.3f}" if len(lo_fam) else "—",
            "Worst-family F1": f"{lo_fam.min():.3f}" if len(lo_fam) else "—",
        })
    tbl = pd.DataFrame(rows)
    em.emit("T09_ablation_stages", tbl,
            "Stage-wise ablation of R-HGT. NOTE: the 'Eval mode' column "
            "is NOT uniform across stages — S1–S5 run in shared mode, "
            "S6 and the strict control run in strict mode; the S5→S6 difference therefore "
            "mixes two variables. LOFO is computed only over sufficiently powered families "
            f"({sorted(keep)}).")


def table_threshold_decomposition(ov: pd.DataFrame, em: Emitter) -> None:
    """Decomposition: does the advantage come from a threshold shift or from a better representation?

    ROC-AUC and PR-AUC are threshold-independent. If F1 rises much more than PR-AUC,
    then most of the gain is an operating-point effect (due to class weighting),
    not separability.
    """
    c = ov[ov["Noise_Type"] == "Clean"]
    a = c[c["Config"] == STRICT_CONTROL]
    b = c[c["Config"] == PROPOSED]
    if a.empty or b.empty:
        warn("Strict Control / Proposed pair not found — skipping the threshold decomposition.")
        return

    rows = []
    for metric, thr_dep in [("Precision", True), ("Recall", True), ("F1", True),
                            ("ROC-AUC", False), ("PR-AUC", False)]:
        rows.append({
            "Metric": metric,
            "Threshold-dependent": "yes" if thr_dep else "NO",
            "Strict control (no FT)": f"{a[metric].mean():.4f}",
            "Proposed (FT + class weight)": f"{b[metric].mean():.4f}",
            "Δ": f"{b[metric].mean() - a[metric].mean():+.4f}",
        })
    d_f1 = b["F1"].mean() - a["F1"].mean()
    d_pr = b["PR-AUC"].mean() - a["PR-AUC"].mean()
    ratio = d_f1 / d_pr if abs(d_pr) > 1e-12 else float("inf")
    rows.append({"Metric": "ΔF1 / ΔPR-AUC", "Threshold-dependent": "—",
                 "Strict control (no FT)": "—",
                 "Proposed (FT + class weight)": "—",
                 "Δ": f"{ratio:.1f}×"})
    em.emit("T10_threshold_vs_representation", pd.DataFrame(rows),
            "Decomposition of the advantage of strict fine-tuning. ROC-AUC and PR-AUC are not "
            "threshold-dependent; if ΔF1 ≫ ΔPR-AUC then most of the F1 gain is an "
            "operating-point shift effect (class weighting), not a representation "
            "improvement. A direct threshold sweep is NEEDED to conclude.")
    warn(f"Threshold decomposition: ΔF1 = {d_f1:+.4f} but ΔPR-AUC is only {d_pr:+.4f} "
         f"(ratio {ratio:.1f}×). An F1-at-optimal-threshold experiment must be added "
         f"for BOTH models before attributing the gain to fine-tuning.")


def table_per_architecture(pa: pd.DataFrame, em: Emitter) -> None:
    """Tables IX/X — consistency across CPU architectures."""
    c = pa[pa["Noise_Type"] == "Clean"]
    tbl = agg_mean_std(c, ["Method", "Architecture"], METRICS)
    em.emit("T11a_per_arch_clean_long", tbl.sort_values(["Method", "Architecture"]),
            "Performance on clean data per CPU architecture (long form).")

    wide = {}
    for arch in ["ARM", "MIPS", "Mipsel", "x86"]:
        cl = c[c["Architecture"] == arch].groupby("Method")["F1"]
        wide[f"{arch} Clean"] = {m: ms(g, 4) for m, g in cl}
        pg = pa[(pa["Architecture"] == arch) & (pa["Noise_Type"] == "PGD_Attack")
                & (np.isclose(pa["Noise_Level"], 0.2))].groupby("Method")["F1"]
        wide[f"{arch} PGD.2"] = {m: ms(g, 4) for m, g in pg}
    t = pd.DataFrame(wide).fillna("—")
    t.index.name = "Method"
    t = t.reset_index()
    em.emit("T11b_per_arch_wide", t,
            "F1 per CPU architecture, clean data and under PGD ε=0.2 attack.")

    # Detect anomalies across architectures.
    mean_f1 = c.groupby(["Method", "Architecture"])["F1"].mean().unstack()
    for m, row in mean_f1.iterrows():
        if row.max() - row.min() > 0.30:
            lo_a, hi_a = row.idxmin(), row.idxmax()
            warn(f"Cross-architecture anomaly — {m}: F1 = {row.min():.4f} on "
                 f"{lo_a} but {row.max():.4f} on {hi_a} "
                 f"(difference {row.max()-row.min():.4f}). Investigate before publishing.")


def table_efficiency(ov: pd.DataFrame, em: Emitter) -> None:
    """Table XI — computational cost, MEASURED ON CLEAN DATA.

    NEVER average the inference time over all scenarios: attack
    scenarios include the cost of GENERATING the attack (PGD 20 steps × 5 restarts),
    which is the attacker's work, not the detector's.
    """
    c = ov[ov["Noise_Type"] == "Clean"]
    f1 = c.groupby("Method")["F1"].mean()
    infer = c.groupby("Method")["Inference (ms/sample)"].agg(["mean", "std"])
    train = c.groupby("Method")["Training Time (s)"].agg(["mean", "std"])
    regime = c.groupby("Method")["Inference_Regime"].first().map(REGIME_LABEL)

    tbl = pd.DataFrame({
        "Method": f1.index,
        "Eval mode": regime.reindex(f1.index).values,
        "Clean F1": f1.round(4).values,
        "Train (s)": [f"{m:.1f} ± {s:.1f}" for m, s in zip(train["mean"], train["std"])],
        "Inference (ms/sample, clean)": [f"{m:.4f} ± {s:.4f}"
                                         for m, s in zip(infer["mean"], infer["std"])],
        "Parameters": "MEASURE WITH measure_params.py",
        "FLOPs": "MEASURE WITH measure_params.py",
    }).sort_values("Clean F1", ascending=False)
    em.emit("T12_efficiency", tbl,
            "Computational cost. Inference time is measured ON CLEAN DATA — "
            "attack scenarios are not pooled, because the cost of generating the attack belongs "
            "to the attacker. The Peak GPU Memory column is dropped: the values in the raw data "
            "are not per-model measurements (see CONSISTENCY_REPORT).")

    # Illustrates the gap if all scenarios are wrongly pooled.
    allcond = ov.groupby("Method")["Inference (ms/sample)"].mean()
    cmp = pd.DataFrame({
        "Method": f1.index,
        "Correct — clean only (ms)": infer["mean"].reindex(f1.index).round(4).values,
        "WRONG — pooled over all scenarios (ms)": allcond.reindex(f1.index).round(4).values,
    })
    cmp["Inflation factor"] = (cmp["WRONG — pooled over all scenarios (ms)"]
                               / cmp["Correct — clean only (ms)"]).round(1)
    cmp = cmp.sort_values("Inflation factor", ascending=False)
    em.emit("T12b_inference_timing_pitfall", cmp,
            "Why filtering by Clean is required: averaging over all scenarios "
            "adds the PGD/BIM attack-generation cost to the inference time.")

    # Check the Peak GPU Memory column.
    mem = c.groupby("Method")["Peak GPU Memory (MB)"].mean().round(4)
    dup = mem[mem.duplicated(keep=False)]
    if len(dup):
        groups = dup.groupby(dup.values).apply(lambda s: list(s.index))
        for val, names in groups.items():
            if len(names) > 1:
                warn(f"Peak GPU Memory = {val} MB identical for {names}. "
                     f"torch.cuda.max_memory_allocated() measures the whole process "
                     f"and memory is not released (bug C1) — this column is not "
                     f"usable for cross-model comparison.")


def table_significance(ov: pd.DataFrame, lofo: pd.DataFrame,
                       sizes: pd.DataFrame, em: Emitter) -> None:
    """Table XII — paired statistical tests with Holm–Bonferroni correction."""
    keep = set(sizes.loc[sizes["sufficient_power"], "Held_Out_Family"])

    scen = {}
    cl = ov[ov["Noise_Type"] == "Clean"]
    scen["Clean F1"] = cl.groupby(["Config", "Seed"])["F1"].mean().unstack(0)
    for nt, lvl, lab in [("PGD_Attack", 0.2, "PGD ε=0.2 F1"),
                         ("FGSM_Attack", 0.2, "FGSM ε=0.2 F1"),
                         ("Edge_Perturbation", 0.5, "EdgeDrop p=0.5 F1")]:
        s = ov[(ov["Noise_Type"] == nt) & (np.isclose(ov["Noise_Level"], lvl))]
        scen[lab] = s.groupby(["Config", "Seed"])["F1"].mean().unstack(0)
    lo_ok = lofo[lofo["Held_Out_Family"].isin(keep)]
    scen["LOFO F1 (sufficiently powered families)"] = lo_ok.groupby(["Config", "Seed"])["F1"].mean().unstack(0)

    comparators = [STRICT_CONTROL, "Multimodal + HIN R2-RGCN", "Multimodal + HIN HAN",
                   "Multimodal Non-HIN GNN", "Multimodal Non-HIN GNN (Res-GraphSAGE)",
                   "Multimodal Late Fusion", "Multimodal Late Fusion (E2E)",
                   "Multimodal + HIN HGT", "Multimodal + HIN GAT",
                   "Multimodal + HIN RGCN", "Multimodal + HIN MAGNN"]

    rows = []
    for lab, piv in scen.items():
        if PROPOSED not in piv.columns:
            continue
        for cfg in comparators:
            if cfg not in piv.columns:
                continue
            a, b = piv[PROPOSED].dropna(), piv[cfg].dropna()
            idx = a.index.intersection(b.index)
            if len(idx) < 3:
                continue
            st = paired_t(a.loc[idx], b.loc[idx])
            rows.append({
                "Scenario": lab,
                "Versus": CANONICAL_NAME.get(cfg, cfg),
                "n seed": st["n"],
                "Mean diff": round(st["mean_diff"], 4),
                "CI 95%": f"[{st['ci_low']:+.4f}, {st['ci_high']:+.4f}]",
                "t": round(st["t"], 2) if np.isfinite(st["t"]) else "inf",
                "Cohen d_z": round(st["dz"], 2) if np.isfinite(st["dz"]) else "inf",
                "p (paired t)": st["p"],
                "p (Wilcoxon)": wilcoxon_exact(a.loc[idx], b.loc[idx]),
            })

    if not rows:
        warn("Could not build the statistical test table.")
        return

    df = pd.DataFrame(rows)
    adj, sig = holm_bonferroni(df["p (paired t)"].to_numpy())
    df["p adjusted (Holm)"] = [f"{v:.2e}" for v in adj]
    df["Significant (α=0.05)"] = ["yes" if s else "NO" for s in sig]
    df["p (paired t)"] = df["p (paired t)"].map(lambda v: f"{v:.2e}")
    df["p (Wilcoxon)"] = df["p (Wilcoxon)"].map(
        lambda v: f"{v:.4f}" if np.isfinite(v) else "—")
    em.emit("T13_statistical_significance", df,
            "Paired tests across seeds, proposed model versus each "
            "baseline. Both the paired t-test and the exact Wilcoxon signed-rank test are included; "
            "p is corrected for multiple comparisons with Holm–Bonferroni. NOTE: the variance "
            "across seeds is very small, so most comparisons are 'significant' — "
            "interpret by the MAGNITUDE of the difference and the confidence interval.")

    tiny = df[(pd.to_numeric(df["Mean diff"]) .abs() < 0.02)
              & (df["Significant (α=0.05)"] == "yes")]
    if len(tiny):
        warn(f"{len(tiny)} comparisons are 'statistically significant' but the absolute "
             f"difference is < 0.02 — because the variance across seeds is extremely small. "
             f"Report the PRACTICAL significance, not just the p-value.")


# =========================================================================
# 6. CONSISTENCY CHECKS
# =========================================================================

def consistency_checks(ov: pd.DataFrame, pa: pd.DataFrame, lofo: pd.DataFrame) -> None:
    # (a) No combined noise conditions exist.
    types = set(ov["Noise_Type"].unique())
    compound = [t for t in types if "+" in str(t)]
    if not compound:
        warn("There are NO COMBINED noise conditions in the raw data "
             "(only: " + ", ".join(sorted(types)) + "). Any table of the form "
             "'PGD .20 + EdgeDrop .20' in the draft is NOT a measured result "
             "— check: if they equal the average of the two neighbouring columns "
             "then it is interpolation and must be removed or the experiment actually run.")

    # (b) Configurations that share weights.
    cl = ov[ov["Noise_Type"] == "Clean"]
    for x, y in KNOWN_WEIGHT_ALIASES:
        ax = cl[cl["Config"] == x]["F1"]
        ay = cl[cl["Config"] == y]["F1"]
        if len(ax) and len(ay) and abs(ax.mean() - ay.mean()) < 1e-9:
            warn(f"'{CANONICAL_NAME.get(x, x)}' and '{CANONICAL_NAME.get(y, y)}' "
                 f"give IDENTICAL numbers (shared checkpoint). Merge them into one "
                 f"row when presenting, or note it explicitly in the caption.")

    # (c) The Parameters column does not exist.
    if "Parameters" not in ov.columns:
        warn("There is no 'Parameters' column in any raw file — every parameter "
             "count in the draft is a hand estimate. Re-measure with "
             "measure_params.py.")

    # (d) Saturation.
    sat = cl.groupby("Method")[["F1", "ROC-AUC"]].mean()
    n_sat = int(((sat["F1"] > 0.999) & (sat["ROC-AUC"] > 0.9999)).sum())
    if n_sat >= 3:
        warn(f"{n_sat} methods reach F1 > 0.999 AND ROC-AUC > 0.9999 on clean "
             f"data. The task is saturated — a near-duplicate analysis "
             f"between train/test and a shortcut-probing experiment are needed (the benign class of "
             f"CIC-YNU-IoTMal is LLM-generated).")

    # (e) Number of seeds.
    for name, df in (("overall", ov), ("per-arch", pa), ("lofo", lofo)):
        counts = df.groupby("Config")["Seed"].nunique()
        if counts.nunique() > 1:
            warn(f"{name}: the number of seeds is uneven across configurations "
                 f"({counts.min()}–{counts.max()}).")


def write_report(out_dir: str, em: Emitter, sizes: pd.DataFrame) -> None:
    p = os.path.join(out_dir, "CONSISTENCY_REPORT.md")
    with open(p, "w", encoding="utf-8") as f:
        f.write("# Consistency report\n\n")
        f.write(f"Generated automatically by `make_tables.py`. "
                f"Total warnings: **{len(WARNINGS)}**\n\n")
        f.write("## Warnings\n\n")
        if WARNINGS:
            for i, w in enumerate(WARNINGS, 1):
                f.write(f"{i}. {w}\n")
        else:
            f.write("No warnings.\n")
        f.write("\n## LOFO test-set sizes (inferred from the data)\n\n")
        f.write(df_to_markdown(sizes))
        f.write("\n\n## Index of generated tables\n\n")
        f.write("| Key | LaTeX label | Content |\n|---|---|---|\n")
        for k, cap, lab in em.index:
            f.write(f"| `{k}` | `{lab}` | {cap.splitlines()[0]} |\n")
        f.write("\n## Remaining work outside the scope of this script\n\n")
        f.write(
            "- Tables I / II (dataset statistics) and Table III (HIN schema): need the "
            "original parquet files, which are not in `raw/`.\n"
            "- Parameter count, FLOPs, model size: run `measure_params.py`.\n"
            "- F1 at the optimal threshold: needs per-sample `y_prob` to be saved — modify the "
            "pipeline and rerun.\n"
            "- Combined noise conditions: the real experiment must be run, not interpolated.\n")
    print(f"\n  ✓ {p}")


# =========================================================================
# 7. MAIN
# =========================================================================

def main() -> None:
    global MIN_FAMILY_POSITIVES

    ap = argparse.ArgumentParser(
        description="Regenerate all result tables from the raw CSVs.")
    ap.add_argument("--raw-dir", default="raw")
    ap.add_argument("--out-dir", default="tables_generated")
    ap.add_argument("--min-family-positives", type=int, default=MIN_FAMILY_POSITIVES,
                    help="Sample-size threshold for a family to enter the main LOFO table.")
    args = ap.parse_args()

    MIN_FAMILY_POSITIVES = args.min_family_positives

    print(f"Reading raw data from: {args.raw_dir}")
    ov, pa, lofo = load(args.raw_dir)
    print(f"  overall  {ov.shape}\n  per-arch {pa.shape}\n  lofo     {lofo.shape}\n")

    print("Inferring LOFO test-set sizes from the quantisation of the metrics...")
    sizes = infer_lofo_test_sizes(lofo)
    print(sizes.to_string(index=False), "\n")

    os.makedirs(args.out_dir, exist_ok=True)
    em = Emitter(args.out_dir)

    print("Building tables:")
    table_overall_clean(ov, em)
    table_protocol_gap(ov, em)
    table_robustness(ov, em)
    table_lofo(lofo, sizes, em)
    table_ablation(ov, lofo, sizes, em)
    table_threshold_decomposition(ov, em)
    table_per_architecture(pa, em)
    table_efficiency(ov, em)
    table_significance(ov, lofo, sizes, em)

    print("\nConsistency checks:")
    consistency_checks(ov, pa, lofo)

    write_report(args.out_dir, em, sizes)
    print(f"\nDone: {len(em.index)} tables × 3 formats in '{args.out_dir}/'")
    print(f"Warnings: {len(WARNINGS)} — see {args.out_dir}/CONSISTENCY_REPORT.md")


if __name__ == "__main__":
    main()
