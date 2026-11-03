#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_claims.py — re-derive the reported numbers of the paper from the released results.

    python verify_claims.py               # from the repository root
    python verify_claims.py --root <path>
    python verify_claims.py --quiet       # print only failures and the summary

Every check prints the value recomputed from the CSV/TXT files in results/ next to the value
printed in the manuscript, grouped by table/section. The script exits with a non-zero code if any
value deviates by more than the rounding of the printed number, so it can be used as a regression
test. It needs only numpy and pandas (statistics come from src/analyze_phase1.py, pure numpy).

Statistics follow Section 5.5 of the paper: paired differences over the ten seeds, the
Nadeau–Bengio corrected resampled t statistic with n_test/n_train = 0.25,
SE_corr = sd * sqrt(1/n + 0.25), two-sided p-values with n-1 degrees of freedom, and Holm
adjustment within each comparison family.

Numbers that cannot be re-derived from this release are listed at the end of the run
(NOT_RELEASED) instead of being silently skipped.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import re
import sys

import numpy as np
import pandas as pd

FAILS: list[str] = []
NCHECK = 0
QUIET = False

S6 = "R-HGT Stage 6 (+ Strict-Inductive Fine-Tuning)"
S6A = "R-HGT Stage 6a (FT, no class weight)"
S5B = "R-HGT Stage 5b (class-weighted, no FT)"
SCTRL = "R-HGT Strict Control (No Fine-Tuning)"   # S5 checkpoint under isolated inference
ARCHS = ["ARM", "MIPS", "Mipsel", "x86"]
N_ARCH = {"ARM": 4792, "MIPS": 4045, "Mipsel": 3924, "x86": 4262}

NOT_RELEASED = [
    "Table 20 (per-encoder cross-run maxima): needs the raw CSVs of both independent runs; "
    "only the summary results/05-audit/crosscheck_l3_l5.md is released.",
    "Section 6.2, upper end of '76-82%' (optimal-threshold F1 0.9838 / 0.9992): from a second run "
    "whose probability dumps are not released; the lower end (~76%) is checked from results/07-threshold.",
    "Section 7.3 trivial all-malicious F1 (0.8877 / 0.1902): needs per-split benign test counts.",
]


# --------------------------------------------------------------------------- helpers
def load_stats(root: str):
    p = os.path.join(root, "src", "analyze_phase1.py")
    if not os.path.isfile(p):
        sys.exit(f"Not found: {p}")
    spec = importlib.util.spec_from_file_location("ap_stats", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _report(name: str, good: bool, got: str, exp: str) -> None:
    global NCHECK
    NCHECK += 1
    if not good:
        FAILS.append(f"{name}: measured {got}, paper {exp}")
    if not good or not QUIET:
        print(f"  {'✓' if good else '✗'} {name:62s} measured {got:>12s}   paper {exp:>12s}")


def chk(name: str, got: float, exp: float, dec: int = 4) -> None:
    """Value printed with `dec` decimals: pass if |got - exp| <= 0.6 * 10^-dec."""
    good = abs(got - exp) <= 0.6 * 10 ** (-dec) + 1e-12
    _report(name, good, f"{got:.{dec}f}", f"{exp:.{dec}f}")


def chk_sig(name: str, got: float, exp: float, sig: int = 3) -> None:
    """Value printed with `sig` significant digits (p-values, small numbers)."""
    if exp == 0:
        good = abs(got) < 1e-12
    else:
        e = math.floor(math.log10(abs(exp)))
        good = abs(got - exp) <= 0.6 * 10 ** (e - sig + 1)
    _report(name, good, f"{got:.{sig - 1}e}", f"{exp:.{sig - 1}e}")


def chk_int(name: str, got: int, exp: int) -> None:
    _report(name, got == exp, f"{got:,d}", f"{exp:,d}")


def chk_true(name: str, cond: bool, got: str, exp: str) -> None:
    _report(name, bool(cond), got, exp)


def head(t: str) -> None:
    if not QUIET:
        print(f"\n{'=' * 100}\n{t}\n{'=' * 100}")


# --------------------------------------------------------------------------- main
def main() -> int:
    global QUIET
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    QUIET = a.quiet
    R = a.root
    st = load_stats(R)

    def path(*parts):
        p = os.path.join(R, "results", *parts)
        if not os.path.isfile(p):
            sys.exit(f"Not found: {p}")
        return p

    def rd(*parts):
        return pd.read_csv(path(*parts))

    def tcrit(df: int) -> float:
        lo, hi = 0.0, 100.0
        for _ in range(200):
            mid = (lo + hi) / 2
            if st.t_sf(mid, df) > 0.05:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    def corrected(d):
        """Nadeau-Bengio corrected resampled t: (mean, sd, ci_lo, ci_hi, p_two_sided)."""
        d = np.asarray(d, float)
        n = len(d)
        md, sd = float(d.mean()), float(d.std(ddof=1))
        se = sd * math.sqrt(1.0 / n + 0.25)
        p = st.t_sf(md / se, n - 1) if se > 0 else 0.0
        h = tcrit(n - 1) * se
        return md, sd, md - h, md + h, p

    main_o = rd("01-main", "robustness_global", "overall_robustness_raw_10seed.csv")
    main_p = rd("01-main", "robustness_per_architecture", "per_architecture_robustness_raw_10seed.csv")
    lofo = rd("01-main", "lofo_zeroday_generalization", "lofo_zeroday_generalization_raw_10seed.csv")
    rob_o = rd("02-robustness", "robustness_global", "overall_robustness_raw_10seed.csv")
    rob_p = rd("02-robustness", "robustness_per_architecture", "per_architecture_robustness_raw_10seed.csv")
    t1 = rd("01-main", "dataset_statistics", "table1_architecture_summary.csv")
    ded_s = rd("03-dedup", "dedup_contamination_summary.csv")
    ded_r = rd("03-dedup", "dedup_per_config_per_seed.csv")
    cost = rd("04-cost", "bench_cost.csv").set_index("Model")
    p1 = rd("05-audit", "P1_provenance_shortcut.csv")
    p1b = rd("05-audit", "P1b_single_feature_shortcut.csv").set_index("feature")
    p2 = rd("05-audit", "P2_architecture_leakage.csv")
    p2f = rd("05-audit", "P2_family_by_arch.csv").set_index("arch")
    p3 = rd("05-audit", "P3_near_duplicate.csv").set_index("percentile")
    p4 = rd("05-audit", "P4_family_counts.csv").set_index("family")
    thr = rd("07-threshold", "threshold_sweep_summary.csv")

    clean = main_o[main_o.Noise_Type == "Clean"]
    pclean = main_p[main_p.Noise_Type == "Clean"]

    def v(cfg, df=clean, m="F1"):
        q = df[df.Config == cfg].sort_values("Run")
        if q.empty:
            sys.exit(f"Configuration not found: {cfg}")
        return q[m].to_numpy(float)

    def rv(cfg, nt, lv, df=rob_o, m="F1"):
        q = df[(df.Config == cfg) & (df.Noise_Type == nt) & np.isclose(df.Noise_Level, lv)].sort_values("Run")
        return q[m].to_numpy(float)

    # ------------------------------------------------------------------ release integrity
    head("0. RELEASE INTEGRITY (not a paper number)")
    chk_int("01-main: rows", len(main_o), 2890)
    chk_int("01-main: seeds", main_o.Run.nunique(), 10)
    chk_int("01-main: configurations", main_o.Config.nunique(), 41)
    cells = main_o.groupby(["Config", "Noise_Type", "Noise_Level"]).Run.nunique()
    chk_int("01-main: cells without all 10 seeds", int((cells != 10).sum()), 0)
    chk_int("01-main: LOFO rows", len(lofo), 820)
    chk_int("02-robustness: rows", len(rob_o), 2410)

    # ------------------------------------------------------------------ Tables 2, 3, 18
    head("Tables 2, 3, 18 — dataset composition")
    t1 = t1.set_index("Architecture")
    for arch, b, m in (("ARM", 1945, 2847), ("MIPS", 2433, 1612), ("MIPSEL", 2375, 1549), ("x86", 2570, 1692)):
        chk_int(f"Table 2 {arch}: benign", int(t1.loc[arch, "Benign"]), b)
        chk_int(f"Table 2 {arch}: malware", int(t1.loc[arch, "Malware"]), m)
    chk_int("Table 2: total samples", int(t1.loc["Total", "Total"]), 17023)
    mal = int(t1.loc["Total", "Malware"])
    for fam, n, pct in (("Mirai", 7368, 95.69), ("DarkNexus", 219, 2.84), ("Gafgyt", 60, 0.78),
                        ("Generic", 47, 0.61), ("Tsunami", 3, 0.04)):
        chk_int(f"Table 3 {fam}: samples", int(p4.loc[fam, "n_samples"]), n)
        chk(f"Table 3 {fam}: % of malware", 100 * p4.loc[fam, "n_samples"] / mal, pct, 2)
    chk_int("Table 3 Other (Agent/Rudedevil)", int(p4.loc["Agent", "n_samples"] + p4.loc["Rudedevil", "n_samples"]), 3)
    chk("Section 8.4: Mirai share of all samples (%)", float(p4.loc["Mirai", "pct"]), 43.28, 2)
    for arch, rate in (("ARM", 59.41), ("MIPS", 39.85), ("MIPSEL", 39.48), ("x86", 39.70)):
        chk(f"Table 18 {arch}: malware rate (%)",
            100 * t1.loc[arch, "Malware"] / t1.loc[arch, "Total"], rate, 2)
    for arch, n in (("arm", 2721), ("mips", 1544), ("mipsel", 1483), ("x86", 1620)):
        chk_int(f"Section 8.2: Mirai samples on {arch}", int(p2f.loc[arch, "Mirai"]), n)

    # ------------------------------------------------------------------ Table 7
    head("Table 7 — clean detection, mean ± sd over ten seeds")
    T7 = [
        ("Branch 1: PCAP MLP", "PCAP MLP", (0.9555, 0.9484, 0.9120, 0.9927, 0.9919), (0.0025, 0.0032, 0.0042, 0.0005, 0.0006)),
        ("Branch 2: SAR Encoder", "SAR Encoder", (0.9281, 0.9189, 0.8592, 0.9894, 0.9876), (0.0076, 0.0117, 0.0128, 0.0007, 0.0011)),
        ("Branch 3: STRACE CNN-LSTM", "STRACE CNN-LSTM", (0.9416, 0.8852, 0.8731, 0.9456, 0.9351), (0.1378, 0.3111, 0.3069, 0.1614, 0.1866)),
        ("Multimodal Late Fusion", "Late Fusion", (0.9949, 0.9944, 0.9897, 0.9994, 0.9994), (0.0108, 0.0120, 0.0219, 0.0012, 0.0013)),
        ("Multimodal Non-HIN GNN", "Non-HIN GNN", (0.9981, 0.9979, 0.9963, 0.9999, 0.9999), (0.0011, 0.0012, 0.0022, 0.0001, 0.0002)),
        ("HIN GAT (robust + strict FT)", "GAT", (0.9999, 0.9999, 0.9998, 1.0000, 1.0000), (0.0002, 0.0002, 0.0003, 0.0000, 0.0000)),
        ("HIN R2-RGCN (robust + strict FT)", "R2-RGCN", (0.9998, 0.9998, 0.9996, 1.0000, 1.0000), (0.0002, 0.0002, 0.0004, 0.0000, 0.0000)),
        ("HIN RGCN (robust + strict FT)", "R-GCN", (0.9997, 0.9997, 0.9994, 0.9996, 0.9997), (0.0003, 0.0003, 0.0006, 0.0004, 0.0002)),
        ("HIN HAN (robust + strict FT)", "HAN", (0.9997, 0.9996, 0.9993, 1.0000, 1.0000), (0.0004, 0.0005, 0.0009, 0.0000, 0.0000)),
        ("HIN HGT (robust + strict FT)", "HGT", (0.9996, 0.9996, 0.9992, 0.9994, 0.9996), (0.0005, 0.0005, 0.0010, 0.0008, 0.0005)),
        ("HIN MAGNN (robust + strict FT)", "MAGNN", (0.9554, 0.9482, 0.9122, 0.9933, 0.9926), (0.0023, 0.0026, 0.0043, 0.0005, 0.0005)),
        (S6, "R-HGT (final)", (0.9997, 0.9997, 0.9994, 1.0000, 1.0000), (0.0002, 0.0003, 0.0005, 0.0000, 0.0000)),
    ]
    for cfg, lab, means, sds in T7:
        for m, mu, sd in zip(("Accuracy", "F1", "MCC", "ROC-AUC", "PR-AUC"), means, sds):
            x = v(cfg, m=m)
            chk(f"Table 7 {lab}: {m} mean", x.mean(), mu)
            chk(f"Table 7 {lab}: {m} sd", x.std(ddof=1), sd)
    head("Section 6.1 — clean ranking among aligned encoders")
    chk("GAT clean F1", v("HIN GAT (robust + strict FT)").mean(), 0.99987, 5)
    chk("R2-RGCN clean F1", v("HIN R2-RGCN (robust + strict FT)").mean(), 0.99977, 5)
    chk("R-HGT clean F1", v(S6).mean(), 0.99967, 5)
    chk("HGT clean F1", v("HIN HGT (robust + strict FT)").mean(), 0.99957, 5)
    chk("MAGNN clean F1", v("HIN MAGNN (robust + strict FT)").mean(), 0.94821, 5)
    # exact value 0.2607; the manuscript prints 0.260 -> checked at two significant digits
    chk_sig("GAT vs R-HGT clean: corrected p", corrected(v("HIN GAT (robust + strict FT)") - v(S6))[4], 0.260, 2)

    # ------------------------------------------------------------------ Table 8
    head("Table 8 — fixed-checkpoint protocol gap (ΔF1 = F1_I − F1_S)")
    PAIRS = [  # (Protocol S config, Protocol I config, label, S, I, Δ, SD, CI lo, CI hi, p_corr, p_corr+Holm)
        ("HIN RGCN [shared eval]", "Multimodal + HIN RGCN", "R-GCN", 0.9987, 0.7955, -0.2032, 0.0380, -0.2540, -0.1523, 8.29e-6, 6.64e-5),
        ("HIN MAGNN [shared eval]", "Multimodal + HIN MAGNN", "MAGNN", 0.9867, 0.7965, -0.1901, 0.0618, -0.2728, -0.1075, 5.62e-4, 3.37e-3),
        ("R-HGT Stage 1 (Vanilla HGT)", "Multimodal + HIN HGT", "HGT (S1)", 0.9988, 0.8356, -0.1632, 0.0383, -0.2144, -0.1120, 5.03e-5, 3.52e-4),
        ("R-HGT Stage 2 (+ Residual + LayerNorm)", "R-HGT Stage 2 [strict eval]", "R-HGT S2", 0.9998, 0.8848, -0.1150, 0.0091, -0.1272, -0.1028, 5.18e-9, 4.67e-8),
        ("R-HGT Stage 3 (+ Node Reliability)", "R-HGT Stage 3 [strict eval]", "R-HGT S3", 0.9998, 0.8875, -0.1123, 0.0038, -0.1174, -0.1073, 2.39e-12, 2.39e-11),
        ("HIN GAT [shared eval]", "Multimodal + HIN GAT", "GAT", 0.8985, 0.7962, -0.1024, 0.0990, -0.2349, 0.0302, 0.1146, 0.3438),
        ("R-HGT Stage 5 (+ Robust Training)", SCTRL, "R-HGT S5", 0.9996, 0.9159, -0.0837, 0.0474, -0.1471, -0.0203, 0.0153, 0.0764),
        ("R-HGT Stage 4 (+ Relation Reliability / Gating)", "R-HGT Stage 4 [strict eval]", "R-HGT S4", 0.9994, 0.9291, -0.0704, 0.0548, -0.1437, 0.0030, 0.0582, 0.2329),
        ("HIN R2-RGCN [shared eval]", "Multimodal + HIN R2-RGCN", "R2-RGCN", 0.9995, 0.9980, -0.0015, 0.0017, -0.0038, 0.0008, 0.1731, 0.3463),
        ("HIN HAN [shared eval]", "Multimodal + HIN HAN", "HAN", 0.9668, 0.9671, 0.0003, 0.0007, -0.0007, 0.0013, 0.4697, 0.4697),
    ]
    p_ord, p_cor = [], []
    for sh, iso, lab, eS, eI, eD, eSD, eLo, eHi, eP, _ in PAIRS:
        x, y = v(sh), v(iso)
        md, sd, lo, hi, p = corrected(y - x)
        p_ord.append(st.paired_t(y, x)[2])
        p_cor.append(p)
        chk(f"Table 8 {lab}: Protocol S", x.mean(), eS)
        chk(f"Table 8 {lab}: Protocol I", y.mean(), eI)
        chk(f"Table 8 {lab}: ΔF1", md, eD)
        chk(f"Table 8 {lab}: SD of Δ", sd, eSD)
        chk(f"Table 8 {lab}: corrected 95% CI low", lo, eLo)
        chk(f"Table 8 {lab}: corrected 95% CI high", hi, eHi)
        chk_sig(f"Table 8 {lab}: p_corr", p, eP, 3)
    holm_cor = st.holm(p_cor)
    for (_, _, lab, *_, eH), h in zip(PAIRS, holm_cor):
        chk_sig(f"Table 8 {lab}: p_corr + Holm", float(h), eH, 3)
    chk_int("Section 6.2: significant after ordinary paired t + Holm", int((st.holm(p_ord) < 0.05).sum()), 9)
    chk_int("Abstract/6.2: significant after corrected t + Holm", int((holm_cor < 0.05).sum()), 5)

    head("Section 6.2 — protocol gap by architecture (mean over the ten pairs)")
    per = {a: [] for a in ARCHS}
    for sh, iso, *_ in PAIRS:
        for arch in ARCHS:
            q = pclean[pclean.Architecture == arch]
            per[arch].append(float((v(iso, q) - v(sh, q)).mean()))
    for arch, exp in zip(ARCHS, (-0.3330, -0.0104, -0.0109, -0.0122)):
        chk(f"mean gap on {arch}", float(np.mean(per[arch])), exp)
    tw = sum(N_ARCH.values())
    contrib = {k: N_ARCH[k] / tw * float(np.mean(per[k])) for k in ARCHS}
    tot = sum(contrib.values())
    chk("sample-weighted reconstruction of the gap", tot, -0.1018)
    chk("ARM share of the reconstructed gap (%)", 100 * contrib["ARM"] / tot, 92.1, 1)
    chk("ARM share of the corpus (%)", 100 * N_ARCH["ARM"] / tw, 28.15, 2)
    hgt = [per[k][2] for k in ("MIPS", "Mipsel", "x86")]
    chk_true("vanilla HGT loses ~0.09-0.11 F1 on each non-ARM architecture",
             all(-0.115 <= g <= -0.085 for g in hgt), " ".join(f"{g:.3f}" for g in hgt), "0.09-0.11")

    head("Section 6.2 — threshold diagnostic (lower end of the 76-82% range)")
    ts = thr.set_index("Model")
    s5_default = float(ts.loc["R-HGT Strict Control  No Fine-Tuning", "F1 @ tau=0.5"].split("±")[0])
    s5_opt = float(ts.loc["R-HGT Strict Control  No Fine-Tuning", "F1 @ tau* (out-of-fold)"].split("±")[0])
    s6_default = float(ts.loc["R-HGT Stage 6    Strict-Inductive Fine-Tuning", "F1 @ tau=0.5"].split("±")[0])
    share = (s5_opt - s5_default) / (s6_default - s5_default)
    chk("share of the clean F1 recovery from operating-point movement (%)", 100 * share, 76.0, 0)

    # ------------------------------------------------------------------ Table 9
    head("Table 9 — strict-inductive alignment from the robust checkpoints")
    T9 = [("HIN GAT", "GAT", 0.7604, 0.0251, 0.9999, 0.0002, 0.2395),
          ("HIN HAN", "HAN", 0.9473, 0.0186, 0.9996, 0.0005, 0.0523),
          ("HIN HGT", "HGT", 0.6919, 0.0926, 0.9996, 0.0005, 0.3076),
          ("HIN MAGNN", "MAGNN", 0.8091, 0.0347, 0.9482, 0.0026, 0.1391),
          ("HIN R2-RGCN", "R2-RGCN", 0.9984, 0.0014, 0.9998, 0.0002, 0.0013),
          ("HIN RGCN", "R-GCN", 0.8047, 0.0267, 0.9997, 0.0003, 0.1950)]
    for cfg, lab, b, bs, f, fs, d in T9:
        x, y = v(f"{cfg} (robust)"), v(f"{cfg} (robust + strict FT)")
        chk(f"Table 9 {lab}: before", x.mean(), b)
        chk(f"Table 9 {lab}: before sd", x.std(ddof=1), bs)
        chk(f"Table 9 {lab}: after", y.mean(), f)
        chk(f"Table 9 {lab}: after sd", y.std(ddof=1), fs)
        chk(f"Table 9 {lab}: ΔF1", y.mean() - x.mean(), d)
    x, y = v(SCTRL), v(S6)
    chk("Table 9 R-HGT: before", x.mean(), 0.9159)
    chk("Table 9 R-HGT: before sd", x.std(ddof=1), 0.0473)
    chk("Table 9 R-HGT: after", y.mean(), 0.9997)
    chk("Table 9 R-HGT: after sd", y.std(ddof=1), 0.0003)
    chk("Table 9 R-HGT: ΔF1", y.mean() - x.mean(), 0.0838)
    _, _, lo, hi, p = corrected(y - x)
    chk("Section 6.3 R-HGT S6 effect: corrected CI low", lo, 0.020, 3)
    chk("Section 6.3 R-HGT S6 effect: corrected CI high", hi, 0.147, 3)
    chk("Section 6.3 R-HGT S6 effect: corrected p", p, 0.015, 3)
    # These differences are stated in the text from the rounded table values.
    r4 = lambda cfg: round(float(v(cfg).mean()), 4)
    hgt_std, gat_std = r4("Multimodal + HIN HGT"), r4("Multimodal + HIN GAT")
    chk("Section 6.3 standard vs robust HGT checkpoint (isolated)", r4("HIN HGT (robust)") - hgt_std, -0.1437)
    chk("Section 6.3 standard vs robust GAT checkpoint (isolated)", r4("HIN GAT (robust)") - gat_std, -0.0358)
    chk("Section 6.3 standard HGT → aligned", r4("HIN HGT (robust + strict FT)") - hgt_std, 0.1640)
    chk("Section 6.3 standard GAT → aligned", r4("HIN GAT (robust + strict FT)") - gat_std, 0.2037)

    # ------------------------------------------------------------------ Table 10
    head("Table 10 — R-HGT component analysis under strict isolated inference")
    T10 = [("R-HGT Stage 2 [strict eval]", "S2", 0.8848, 0.0091, 0.8183, 0.9751, 0.9720, 0.9675),
           ("R-HGT Stage 3 [strict eval]", "S3", 0.8875, 0.0037, 0.7984, 0.9992, 0.9688, 0.9614),
           ("R-HGT Stage 4 [strict eval]", "S4", 0.9291, 0.0545, 0.9152, 0.9563, 0.9989, 0.9987),
           (SCTRL, "S5", 0.9159, 0.0473, 0.9963, 0.8513, 0.9983, 0.9979),
           (S5B, "S5b", 0.8957, 0.0352, 0.9987, 0.8138, 0.9974, 0.9971),
           (S6A, "S6a", 0.9997, 0.0002, 0.9999, 0.9995, 1.0000, 1.0000),
           (S6, "S6", 0.9997, 0.0003, 0.9999, 0.9995, 1.0000, 1.0000)]
    for cfg, lab, f1, sd, pr, rc, roc, pra in T10:
        chk(f"Table 10 {lab}: F1", v(cfg).mean(), f1)
        chk(f"Table 10 {lab}: F1 sd", v(cfg).std(ddof=1), sd)
        chk(f"Table 10 {lab}: precision", v(cfg, m="Precision").mean(), pr)
        chk(f"Table 10 {lab}: recall", v(cfg, m="Recall").mean(), rc)
        chk(f"Table 10 {lab}: ROC-AUC", v(cfg, m="ROC-AUC").mean(), roc)
        chk(f"Table 10 {lab}: PR-AUC", v(cfg, m="PR-AUC").mean(), pra)
    seq = ["R-HGT Stage 2 [strict eval]", "R-HGT Stage 3 [strict eval]", "R-HGT Stage 4 [strict eval]", SCTRL, S6]
    for (a_, b_), d in zip(zip(seq, seq[1:]), (0.0027, 0.0415, -0.0132, 0.0838)):
        chk(f"Table 10 ΔF1 {a_.split(' [')[0]} → {b_.split(' (')[0]}", v(b_).mean() - v(a_).mean(), d)
    # exact value 0.2190; the manuscript prints 0.218 -> checked at two significant digits
    chk_sig("Section 6.4 S3 → S4: corrected p", corrected(v("R-HGT Stage 4 [strict eval]") - v("R-HGT Stage 3 [strict eval]"))[4], 0.218, 2)
    chk("Section 6.4 S6a F1 (5 decimals)", v(S6A).mean(), 0.99974, 5)
    chk("Section 6.4 S6a F1 sd", v(S6A).std(ddof=1), 0.00021, 5)
    chk("Section 6.4 S6 F1 (5 decimals)", v(S6).mean(), 0.99967, 5)
    chk("Section 6.4 S6 F1 sd", v(S6).std(ddof=1), 0.00027, 5)
    chk_sig("Section 6.4 S6 − S6a", float((v(S6) - v(S6A)).mean()), -6.5e-5, 2)
    chk("Section 6.4 ROC-AUC before alignment", v(SCTRL, m="ROC-AUC").mean(), 0.99825, 5)
    chk("Section 6.4 ROC-AUC after alignment", v(S6, m="ROC-AUC").mean(), 0.99999, 5)
    chk("Section 6.4 PR-AUC before alignment", v(SCTRL, m="PR-AUC").mean(), 0.99795, 5)
    chk("Section 6.4 PR-AUC after alignment", v(S6, m="PR-AUC").mean(), 0.99999, 5)

    # ------------------------------------------------------------------ Tables 11-14
    head("Table 11 — non-adversarial perturbations (02-robustness, own clean reference)")
    T11 = [("Multimodal Late Fusion", "Late Fusion", (0.9984, 0.0008), (0.8866, 0.0163), (0.8563, 0.0430), None),
           ("Multimodal Non-HIN GNN", "Non-HIN GNN", (0.9979, 0.0011), (0.9971, 0.0010), (0.9771, 0.0029), (0.9980, 0.0010)),
           ("HIN GAT (robust + strict FT)", "GAT", (0.9997, 0.0005), (0.9993, 0.0006), (0.9981, 0.0009), (0.9786, 0.0166)),
           ("HIN HAN (robust + strict FT)", "HAN", (0.9996, 0.0004), (0.9994, 0.0005), (0.9991, 0.0008), (0.9912, 0.0040)),
           (S6, "R-HGT", (0.9997, 0.0002), (0.9995, 0.0004), (0.9992, 0.0005), (0.9873, 0.0026))]
    for cfg, lab, *cols in T11:
        for (nt, lv), e in zip((("Clean", 0.0), ("Gaussian_Noise", 0.5), ("Feature_Masking", 0.5), ("Edge_Perturbation", 0.5)), cols):
            if e is None:
                continue
            x = rv(cfg, nt, lv)
            chk(f"Table 11 {lab}: {nt} mean", x.mean(), e[0])
            chk(f"Table 11 {lab}: {nt} sd", x.std(ddof=1), e[1])
    chk("Section 7.1 unaligned R-HGT clean reference", rv(SCTRL, "Clean", 0.0).mean(), 0.9181)

    head("Table 12 — gradient attacks at ε = 0.20")
    T12 = [("Multimodal Late Fusion", "Late Fusion", (0.3071, 0.0991), (0.1861, 0.1228), (0.1461, 0.1059)),
           ("Multimodal Non-HIN GNN", "Non-HIN GNN", (0.8930, 0.0304), (0.8656, 0.0315), (0.8299, 0.0291)),
           ("HIN GAT (robust + strict FT)", "GAT", (0.9642, 0.0112), (0.9410, 0.0185), (0.9141, 0.0238)),
           ("HIN HAN (robust + strict FT)", "HAN", (0.9676, 0.0080), (0.9511, 0.0085), (0.9374, 0.0100)),
           (S6, "R-HGT", (0.9705, 0.0085), (0.9458, 0.0166), (0.9228, 0.0211))]
    for cfg, lab, *cols in T12:
        for nt, e in zip(("FGSM_Attack", "BIM_Attack", "PGD_Attack"), cols):
            x = rv(cfg, nt, 0.2)
            chk(f"Table 12 {lab}: {nt.split('_')[0]} mean", x.mean(), e[0])
            chk(f"Table 12 {lab}: {nt.split('_')[0]} sd", x.std(ddof=1), e[1])
    for nt, seq in (("PGD_Attack", (0.9997, 0.9995, 0.9985, 0.9228)), ("BIM_Attack", (0.9997, 0.9995, 0.9986, 0.9458))):
        for lv, e in zip((0.01, 0.05, 0.10, 0.20), seq):
            chk(f"Section 7.2 R-HGT {nt.split('_')[0]} ε={lv}", rv(S6, nt, lv).mean(), e)
    for nt, e in (("FGSM_Attack", 0.9987), ("BIM_Attack", 0.9963), ("PGD_Attack", 0.9930)):
        chk(f"Section 7.2 R-HGT {nt.split('_')[0]} ε=0.2 ROC-AUC", rv(S6, nt, 0.2, m="ROC-AUC").mean(), e)
    chk("Section 7.2 HAN vs R-HGT under PGD ε=0.2: corrected p",
        corrected(rv("HIN HAN (robust + strict FT)", "PGD_Attack", 0.2) - rv(S6, "PGD_Attack", 0.2))[4], 0.196, 3)
    ps = []
    for nt, lv in (("FGSM_Attack", 0.2), ("BIM_Attack", 0.2), ("PGD_Attack", 0.2),
                   ("Gaussian_Noise", 0.5), ("Feature_Masking", 0.5), ("Edge_Perturbation", 0.5)):
        for opp in ("HIN HAN (robust + strict FT)", "HIN GAT (robust + strict FT)"):
            ps.append(corrected(rv(S6, nt, lv) - rv(opp, nt, lv))[4])
    chk_int("Section 5.5 robustness family: comparisons", len(ps), 12)
    chk_int("Section 5.5 robustness family: significant after Holm", int((st.holm(ps) < 0.05).sum()), 0)

    head("Table 13 — same R-HGT with and without strict realignment at ε = 0.20")
    # Holm-corrected p for FGSM: the data give 9.46e-6; the manuscript prints 9.0e-6,
    # so the Holm p-values are checked at one significant digit.
    T13 = [("FGSM_Attack", 0.9705, 0.8188, 0.1517, 0.0253, 0.1179, 0.1855, 9.0e-6, 0.0292, 0.0993),
           ("BIM_Attack", 0.9458, 0.7953, 0.1504, 0.0308, 0.1092, 0.1917, 3.5e-5, 0.0539, 0.1227),
           ("PGD_Attack", 0.9228, 0.7770, 0.1458, 0.0313, 0.1039, 0.1877, 3.5e-5, 0.0769, 0.1411)]
    praw = []
    for nt, al, un, d, sd, lo, hi, _, dga, dgu in T13:
        x, y = rv(S6, nt, 0.2), rv(SCTRL, nt, 0.2)
        md, s, l_, h_, p = corrected(x - y)
        praw.append(p)
        lab = nt.split("_")[0]
        chk(f"Table 13 {lab}: aligned", x.mean(), al)
        chk(f"Table 13 {lab}: unaligned", y.mean(), un)
        chk(f"Table 13 {lab}: ΔF1", md, d)
        chk(f"Table 13 {lab}: SD", s, sd)
        chk(f"Table 13 {lab}: CI low", l_, lo)
        chk(f"Table 13 {lab}: CI high", h_, hi)
        chk(f"Section 7.2 {lab}: degradation, aligned", rv(S6, "Clean", 0.0).mean() - x.mean(), dga)
        chk(f"Section 7.2 {lab}: degradation, unaligned", rv(SCTRL, "Clean", 0.0).mean() - y.mean(), dgu)
    for (nt, *_r), h in zip(T13, st.holm(praw)):
        chk_sig(f"Table 13 {nt.split('_')[0]}: Holm-corrected p", float(h), _r[6], 1)

    head("Table 14 — R-HGT by architecture under PGD ε = 0.20")
    T14 = [("ARM", (0.9964, 0.0040), (0.9970, 0.0033), (0.9996, 0.0012)),
           ("MIPS", (0.9236, 0.0278), (0.9109, 0.0310), (0.9942, 0.0031)),
           ("Mipsel", (0.8799, 0.0250), (0.8667, 0.0243), (0.9942, 0.0033)),
           ("x86", (0.8920, 0.0462), (0.8738, 0.0497), (0.9773, 0.0124))]
    for arch, *cols in T14:
        q = rob_p[rob_p.Architecture == arch]
        for m, e in zip(("Accuracy", "F1", "ROC-AUC"), cols):
            x = rv(S6, "PGD_Attack", 0.2, df=q, m=m)
            chk(f"Table 14 {arch}: {m} mean", x.mean(), e[0])
            chk(f"Table 14 {arch}: {m} sd", x.std(ddof=1), e[1])

    # ------------------------------------------------------------------ Table 15
    head("Table 15 — family-held-out generalization")
    T15 = {"Mirai": [("HIN GAT (robust + strict FT)", "GAT", (0.9553, 0.0042), (0.9712, 0.0028), (0.9954, 0.0009), (0.9441, 0.0053)),
                     ("HIN HAN (robust + strict FT)", "HAN", (0.9631, 0.0100), (0.9763, 0.0065), (0.9975, 0.0019), (0.9542, 0.0134)),
                     ("HIN HGT (robust + strict FT)", "HGT", (0.9480, 0.0057), (0.9663, 0.0038), (0.9963, 0.0022), (0.9349, 0.0071)),
                     ("HIN R2-RGCN (robust + strict FT)", "R2-RGCN", (0.9458, 0.0023), (0.9648, 0.0015), (0.9992, 0.0005), (0.9321, 0.0028)),
                     (S6, "R-HGT", (0.9597, 0.0043), (0.9741, 0.0029), (0.9997, 0.0002), (0.9496, 0.0054))],
           "DarkNexus": [("HIN GAT (robust + strict FT)", "GAT", (0.9998, 0.0003), (0.9991, 0.0016), (1.0000, 0.0001), (1.0000, 0.0000)),
                         ("HIN HAN (robust + strict FT)", "HAN", (0.9996, 0.0005), (0.9982, 0.0023), (1.0000, 0.0000), (1.0000, 0.0000)),
                         ("HIN HGT (robust + strict FT)", "HGT", (0.9999, 0.0002), (0.9993, 0.0011), (1.0000, 0.0000), (1.0000, 0.0000)),
                         ("HIN R2-RGCN (robust + strict FT)", "R2-RGCN", (0.9998, 0.0003), (0.9989, 0.0016), (1.0000, 0.0000), (1.0000, 0.0000)),
                         (S6, "R-HGT", (0.9999, 0.0003), (0.9993, 0.0015), (1.0000, 0.0000), (1.0000, 0.0000))]}
    mets = ("Accuracy", "F1", "ROC-AUC", "ZeroDay_Detection_Rate (Recall)")
    for fam, rows in T15.items():
        q = lofo[lofo.Held_Out_Family == fam]
        for cfg, lab, *cols in rows:
            for m, e in zip(mets, cols):
                x = v(cfg, q, m)
                short = "held-out recall" if m.startswith("ZeroDay") else m
                chk(f"Table 15 {fam} {lab}: {short} mean", x.mean(), e[0])
                chk(f"Table 15 {fam} {lab}: {short} sd", x.std(ddof=1), e[1])
    qm, qd = lofo[lofo.Held_Out_Family == "Mirai"], lofo[lofo.Held_Out_Family == "DarkNexus"]
    chk("Section 7.3 Mirai, no fine-tuning: F1", v(SCTRL, qm).mean(), 0.9262)
    chk("Section 7.3 Mirai, no fine-tuning: F1 sd", v(SCTRL, qm).std(ddof=1), 0.0358)
    chk("Section 7.3 Mirai, no fine-tuning: HFR", v(SCTRL, qm, mets[3]).mean(), 0.8776)
    chk("Section 7.3 Mirai, no fine-tuning: HFR sd", v(SCTRL, qm, mets[3]).std(ddof=1), 0.0803)
    chk("Section 7.3 DarkNexus, no fine-tuning: F1", v(SCTRL, qd).mean(), 0.8547)
    chk("Section 7.3 DarkNexus, no fine-tuning: F1 sd", v(SCTRL, qd).std(ddof=1), 0.0644)
    chk("Section 7.3 DarkNexus, no fine-tuning: HFR", v(SCTRL, qd, mets[3]).mean(), 0.7671)
    d = v(S6, qm) - v("HIN GAT (robust + strict FT)", qm)
    chk("Section 7.3 R-HGT − GAT on Mirai: ΔF1", d.mean(), 0.0029)
    chk("Section 7.3 R-HGT − GAT on Mirai: corrected p", corrected(d)[4], 0.221, 3)

    # ------------------------------------------------------------------ Table 6
    head("Table 6 — cost (Tesla T4) and representative performance")
    T6 = [("HIN-HAN", "HAN", 68674, 52.8, 5.58), ("HIN-MAGNN", "MAGNN", 86466, 44.2, 2.63),
          ("HIN-RGCN", "R-GCN", 137410, 40.9, 3.71), ("HIN-GAT", "GAT", 165186, 67.2, 10.79),
          ("HIN-HGT", "HGT", 208642, 109.8, 11.01), ("HIN-R2-RGCN", "R2-RGCN", 221400, 70.7, 7.51),
          ("R-HGT S4 / Proposed", "R-HGT", 349976, 130.3, 17.67)]
    for key, lab, prm, mem, lat in T6:
        chk_int(f"Table 6 {lab}: parameters", int(cost.loc[key, "Parameters"]), prm)
        chk(f"Table 6 {lab}: peak memory (MB)", float(cost.loc[key, "Mem train (MB)"]), mem, 1)
        chk(f"Table 6 {lab}: latency (ms)", float(cost.loc[key, "Latency I (ms/sample)"]), lat, 2)
    chk("Table 6 R-GCN: held-out Mirai F1", v("HIN RGCN (robust + strict FT)", qm).mean(), 0.9654)
    chk("Table 6 MAGNN: held-out Mirai F1", v("HIN MAGNN (robust + strict FT)", qm).mean(), 0.9410)
    g_, r_ = cost.loc["HIN-GAT"], cost.loc["R-HGT S4 / Proposed"]
    chk("Section 5.5 GAT / R-HGT parameters", g_["Parameters"] / r_["Parameters"], 0.47, 2)
    chk("Section 5.5 GAT / R-HGT memory", g_["Mem train (MB)"] / r_["Mem train (MB)"], 0.52, 2)
    chk("Section 5.5 GAT / R-HGT latency", g_["Latency I (ms/sample)"] / r_["Latency I (ms/sample)"], 0.61, 2)

    # ------------------------------------------------------------------ Tables 16-17, Section 8
    head("Tables 16-17, Section 8 — benchmark audits")
    for mdl, lab, acc, f1, roc in (("LogisticRegression", "Logistic regression", (0.99932, 0.00017), (0.99924, 0.00019), 1.0),
                                    ("RandomForest", "Random forest", (0.99990, 0.00017), (0.99989, 0.00019), 1.0)):
        q = p1[p1.model == mdl]
        chk(f"Table 16 {lab}: accuracy", q.accuracy.mean(), acc[0], 5)
        chk(f"Table 16 {lab}: accuracy sd", q.accuracy.std(ddof=1), acc[1], 5)
        chk(f"Table 16 {lab}: F1", q.f1.mean(), f1[0], 5)
        chk(f"Table 16 {lab}: F1 sd", q.f1.std(ddof=1), f1[1], 5)
        chk(f"Table 16 {lab}: ROC-AUC", q.roc_auc.mean(), roc, 5)
    for feat, f1 in (("Call_set_tid_address", 0.9704), ("Telnet", 0.9291), ("Call_mmap2", 0.9188),
                     ("Call_writev", 0.9148), ("Call_fcntl64", 0.9067), ("syn_flag_number", 0.8922),
                     ("Header_Length", 0.8895)):
        chk(f"Table 17 stump {feat}: F1", float(p1b.loc[feat, "f1_stump"]), f1)
    best = p1b.f1_stump.idxmax()
    chk_true("Table 17 strongest single feature", best == "Call_set_tid_address", best, "Call_set_tid_address")
    for (seed, acc) in zip((10, 42, 100), (0.99941, 0.99971, 0.99971)):
        chk(f"Section 8.2 architecture accuracy, split {seed}", float(p2[p2.seed == seed].arch_accuracy.iloc[0]), acc, 5)
    chk("Section 8.2 architecture accuracy mean", p2.arch_accuracy.mean(), 0.99961, 5)
    chk("Section 8.2 architecture accuracy sd", p2.arch_accuracy.std(ddof=1), 0.00017, 5)
    for pc_, e in ((50, 0.9660), (90, 0.9985), (99, 0.9997)):
        chk(f"Section 8.3 max cosine to train, percentile {pc_}", float(p3.loc[pc_, "max_cosine_to_train"]), e)
    for seed, n in zip((10, 42, 100), (343, 322, 306)):
        chk_int(f"Section 8.3 test samples above cosine 0.999, seed {seed}",
                int(ded_s[ded_s.seed == seed].n_contaminated.iloc[0]), n)
    chk_int("Section 8.3 occurrences across the three seeds", int(ded_s.n_contaminated.sum()), 971)
    chk_int("Section 8.3 occurrences sharing the label of the nearest train sample",
            int(ded_s.twin_same_label.sum()), 971)
    chk("Section 8.3 mean rate (%)", ded_s.pct.mean(), 9.51, 2)
    g = ded_r.groupby("Config")[["F1_full", "F1_dedup"]].mean()
    s6d = [c for c in g.index if c.startswith("R-HGT_Stage_6")][0]
    chk_sig("Section 8.3 S6 R-HGT ΔF1 after removal", float(g.loc[s6d, "F1_dedup"] - g.loc[s6d, "F1_full"]), -4e-6, 1)
    DED = [("HIN_RGCN__shared_eval_", "Multimodal___HIN_RGCN"), ("HIN_MAGNN__shared_eval_", "Multimodal___HIN_MAGNN"),
           ("R-HGT_Stage_1__Vanilla_HGT_", "Multimodal___HIN_HGT"),
           ("R-HGT_Stage_2____Residual___LayerNorm_", "R-HGT_Stage_2__strict_eval_"),
           ("R-HGT_Stage_3____Node_Reliability_", "R-HGT_Stage_3__strict_eval_"),
           ("HIN_GAT__shared_eval_", "Multimodal___HIN_GAT"),
           ("R-HGT_Stage_5____Robust_Training_", "R-HGT_Strict_Control__No_Fine-Tuning_"),
           ("R-HGT_Stage_4____Relation_Reliability___Gating_", "R-HGT_Stage_4__strict_eval_")]
    full = float(np.mean([g.loc[i, "F1_full"] - g.loc[s, "F1_full"] for s, i in DED]))
    dedu = float(np.mean([g.loc[i, "F1_dedup"] - g.loc[s, "F1_dedup"] for s, i in DED]))
    chk("Section 8.3 mean protocol gap, eight pairs, full test set", full, -0.1281)
    chk("Section 8.3 mean protocol gap, eight pairs, after removal", dedu, -0.0951)
    chk("Section 8.3 share of the gap removed (%)", 100 * (1 - dedu / full), 26.0, 0)

    # ------------------------------------------------------------------ isolation, reproducibility
    head("Sections 3.3 / 5.3 — isolation check (results/06-isolation)")
    txt = open(path("06-isolation", "verify_slim_ego_output.txt"), encoding="utf-8").read()
    pat = r"^\s+(\S.*?)\s+([0-9.]+e[-+][0-9]+)\s+(?:OK|FAIL)"
    equiv = dict(re.findall(pat, txt.split("EQUIVALENCE CHECK")[1].split("ISOLATION CHECK")[0], re.M))
    iso = dict(re.findall(pat, txt.split("ISOLATION CHECK")[1].split("CONCLUSION")[0], re.M))
    others = {k: float(x) for k, x in iso.items() if not k.startswith("HIN HAN")}
    chk_int("architectures passing batch-invariance (bs 16 vs 128)", sum(x < 5e-7 for x in others.values()), 6)
    chk_true("largest difference among those six < 5e-7", max(others.values()) < 5e-7,
             f"{max(others.values()):.2e}", "< 5e-7")
    chk_sig("HAN batch-dependence", float(iso["HIN HAN"]), 3.2e-3, 2)
    chk_true("full vs reduced ego-graph: max difference < 5e-7",
             max(float(x) for x in equiv.values()) < 5e-7, f"{max(float(x) for x in equiv.values()):.2e}", "< 5e-7")
    red = re.search(r"→\s*([0-9.]+)×", txt)
    chk("reduction in nodes per batch (×)", float(red.group(1)), 12.5, 1)

    head("Section 9.3 — reproducibility across two independent runs")
    md = open(path("05-audit", "crosscheck_l3_l5.md"), encoding="utf-8").read()
    m = re.search(r"## overall — (\d+) matched keys.*?\| F1\s*\|\s*\d+\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)", md, re.S)
    chk_int("matched evaluation keys", int(m.group(1)), 1930)
    chk("median absolute F1 difference", float(m.group(3)), 0.00333, 5)
    chk("95th percentile absolute F1 difference", float(m.group(4)), 0.1062, 4)

    # ------------------------------------------------------------------ summary
    print(f"\n{'=' * 100}")
    print("Not re-derivable from this release:")
    for s in NOT_RELEASED:
        print(f"  - {s}")
    print("=" * 100)
    if FAILS:
        print(f"❌ {len(FAILS)}/{NCHECK} checks FAILED:")
        for f in FAILS:
            print(f"   • {f}")
        return 1
    print(f"✅ {NCHECK}/{NCHECK} reported numbers reproduced from the released results.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
