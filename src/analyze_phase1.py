#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_phase1.py — analysis of session-1 results (clean, extra=AC, 10 seeds).

Produces the three core tables of the paper, all using PAIRED tests by seed:

  T-A  Protocol gap per architecture (shared vs strict, same checkpoint)
  T-B  2x2 ablation: {fine-tune, no} x {class weight, no}
  T-C  Threshold vs representation: F1@0.5, F1@tau*, AP, and the percentage due to the threshold

scipy is not used (it may not be installed): the paired t-test, the exact Wilcoxon
test and Holm-Bonferroni are all hand-written in numpy.

    python3 analyze_phase1.py --results-dir . --out-dir ./analysis
"""
from __future__ import annotations

import argparse
import math
import os
import re
from itertools import combinations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Pure-numpy statistics
# --------------------------------------------------------------------------- #
def _betacf(a, b, x, itmax=200, eps=3e-16):
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < 1e-300: d = 1e-300
    d = 1.0 / d; h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d; c = 1.0 + aa / c
        if abs(d) < 1e-300: d = 1e-300
        if abs(c) < 1e-300: c = 1e-300
        d = 1.0 / d; h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d; c = 1.0 + aa / c
        if abs(d) < 1e-300: d = 1e-300
        if abs(c) < 1e-300: c = 1e-300
        d = 1.0 / d
        de = d * c; h *= de
        if abs(de - 1.0) < eps: break
    return h


def _betainc(a, b, x):
    if x <= 0: return 0.0
    if x >= 1: return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_sf(t, df):
    """Two-sided P(T > |t|) for the Student distribution."""
    t = abs(float(t))
    if df <= 0 or not np.isfinite(t): return float('nan')
    return _betainc(df / 2.0, 0.5, df / (df + t * t))


def paired_t(a, b):
    """Returns (mean_diff, t, p_two_sided, ci_low, ci_high, n)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 2: return (float(np.mean(d)) if n else float('nan'),
                      float('nan'), float('nan'), float('nan'), float('nan'), n)
    md, sd = d.mean(), d.std(ddof=1)
    if sd == 0:
        return md, float('inf') if md else 0.0, 0.0 if md else 1.0, md, md, n
    se = sd / math.sqrt(n)
    t = md / se
    p = t_sf(t, n - 1)
    # t 0.975 for small df — looked up by binary search on t_sf
    lo, hi = 0.0, 100.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if t_sf(mid, n - 1) > 0.05: lo = mid
        else: hi = mid
    tcrit = (lo + hi) / 2
    return md, t, p, md - tcrit * se, md + tcrit * se, n


def wilcoxon_exact(a, b):
    """Exact Wilcoxon signed-rank test (n <= 20). Returns (W, p_two_sided)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[np.isfinite(d) & (d != 0)]
    n = len(d)
    if n == 0: return float('nan'), 1.0
    order = np.argsort(np.abs(d))
    ranks = np.empty(n); ranks[order] = np.arange(1, n + 1)
    # handle tied ranks
    absd = np.abs(d)
    for v in np.unique(absd):
        m = absd == v
        if m.sum() > 1: ranks[m] = ranks[m].mean()
    W = ranks[d > 0].sum()
    if n > 20:
        mu = n * (n + 1) / 4; sig = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
        z = (W - mu) / sig
        return W, math.erfc(abs(z) / math.sqrt(2))
    # enumerate all 2^n sign patterns
    cnt = 0; tot = 0
    base = np.arange(1, n + 1)
    for mask in range(1 << n):
        s = base[[(mask >> i) & 1 == 1 for i in range(n)]].sum()
        tot += 1
        if abs(s - n * (n + 1) / 4) >= abs(W - n * (n + 1) / 4) - 1e-9: cnt += 1
    return W, cnt / tot


def holm(pvals):
    """Holm-Bonferroni correction. Returns the array of adjusted p-values."""
    p = np.asarray(pvals, float)
    n = len(p); order = np.argsort(p)
    adj = np.empty(n); run = 0.0
    for i, idx in enumerate(order):
        run = max(run, (n - i) * p[idx])
        adj[idx] = min(run, 1.0)
    return adj


def roc_auc(y, s):
    y = np.asarray(y).astype(int); s = np.asarray(s, float)
    pos, neg = (y == 1).sum(), (y == 0).sum()
    if pos == 0 or neg == 0: return float('nan')
    order = np.argsort(s, kind='mergesort')
    ranks = np.empty(len(s), float); ranks[order] = np.arange(1, len(s) + 1)
    for v in np.unique(s):
        m = s == v
        if m.sum() > 1: ranks[m] = ranks[m].mean()
    return (ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg)


def average_precision(y, s):
    y = np.asarray(y).astype(int); s = np.asarray(s, float)
    order = np.argsort(-s, kind='mergesort')
    y = y[order]
    tp = np.cumsum(y); fp = np.cumsum(1 - y)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / max(y.sum(), 1)
    return float(np.sum(np.diff(np.concatenate([[0.0], rec])) * prec))


def f1_at(y, s, tau):
    pred = (s >= tau).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)


def best_f1(y, s):
    """Optimal F1 over ALL possible thresholds, computed in O(n log n).

    Sort the scores in descending order; if the top k samples are predicted
    positive then tp = cumsum(y), fp = k - tp, fn = P - tp, so
        F1 = 2*tp / (2*tp + fp + fn) = 2*tp / (k + P).
    Take the max over k. No threshold grid sweep is needed, and no threshold is missed.
    """
    y = np.asarray(y).astype(int)
    s = np.asarray(s, float)
    P = int(y.sum())
    if P == 0:
        return 0.0, 0.5
    order = np.argsort(-s, kind='mergesort')
    ys, ss = y[order], s[order]
    tp = np.cumsum(ys)
    k = np.arange(1, len(ys) + 1)
    f1 = 2.0 * tp / (k + P)
    # keep only positions that are real boundaries (the next score differs)
    valid = np.ones(len(ys), bool)
    valid[:-1] = ss[:-1] != ss[1:]
    f1 = np.where(valid, f1, -1.0)
    i = int(np.argmax(f1))
    return float(f1[i]), float(ss[i])


# --------------------------------------------------------------------------- #
def load_raw(results_dir):
    p = os.path.join(results_dir, 'robustness_global', 'overall_robustness_raw_10seed.csv')
    d = pd.read_csv(p)
    if 'Noise_Type' in d.columns:
        d = d[(d['Noise_Type'].isna()) | (d['Noise_Type'].astype(str).str.lower().isin(['', 'nan', 'clean', 'none']))]
    return d


def pivot(d, metric):
    """Seed x config table for one metric."""
    return d.pivot_table(index='Run', columns='Config', values=metric, aggfunc='mean')


# --------------------------------------------------------------------------- #
# T-A: protocol gap
# --------------------------------------------------------------------------- #
PROTOCOL_PAIRS = [
    # (label, configuration under Protocol S, configuration under Protocol I)
    ("RGCN",                    "HIN RGCN [shared eval]",                          "Multimodal + HIN RGCN"),
    ("R2-RGCN (R-HGT, RGCN core)", "HIN R2-RGCN [shared eval]",                    "Multimodal + HIN R2-RGCN"),
    ("MAGNN*",                  "HIN MAGNN [shared eval]",                         "Multimodal + HIN MAGNN"),
    ("HAN",                     "HIN HAN [shared eval]",                           "Multimodal + HIN HAN"),
    ("GAT",                     "HIN GAT [shared eval]",                           "Multimodal + HIN GAT"),
    ("HGT (vanilla)",           "R-HGT Stage 1 (Vanilla HGT)",                     "Multimodal + HIN HGT"),
    ("R-HGT S2",                "R-HGT Stage 2 (+ Residual + LayerNorm)",          "R-HGT Stage 2 [strict eval]"),
    ("R-HGT S3",                "R-HGT Stage 3 (+ Node Reliability)",              "R-HGT Stage 3 [strict eval]"),
    ("R-HGT S4",                "R-HGT Stage 4 (+ Relation Reliability / Gating)", "R-HGT Stage 4 [strict eval]"),
    ("R-HGT S5",                "R-HGT Stage 5 (+ Robust Training)",               "R-HGT Strict Control (No Fine-Tuning)"),
]


def table_A(d, out_dir):
    f1 = pivot(d, 'F1'); ap = pivot(d, 'PR-AUC')
    pr = pivot(d, 'Precision'); rc = pivot(d, 'Recall')
    rows, praw = [], []
    for label, cs, ci in PROTOCOL_PAIRS:
        if cs not in f1.columns or ci not in f1.columns:
            print(f"  [!] skipping {label}: missing configuration"); continue
        md, t, p, lo, hi, n = paired_t(f1[ci], f1[cs])
        _, pw = wilcoxon_exact(f1[ci], f1[cs])
        mda, *_ = paired_t(ap[ci], ap[cs])
        rows.append({
            'Architecture': label, 'n_seed': n,
            'F1_shared': f1[cs].mean(), 'F1_shared_sd': f1[cs].std(ddof=1),
            'F1_strict': f1[ci].mean(), 'F1_strict_sd': f1[ci].std(ddof=1),
            'Delta_F1': md, 'CI95_low': lo, 'CI95_high': hi,
            't': t, 'p_paired_t': p, 'p_wilcoxon': pw,
            'Delta_AP': mda,
            'Prec_strict': pr[ci].mean(), 'Rec_strict': rc[ci].mean(),
        })
        praw.append(p)
    T = pd.DataFrame(rows)
    if len(T):
        T['p_holm'] = holm(praw)
        T = T.sort_values('Delta_F1')
    T.round(6).to_csv(os.path.join(out_dir, 'TA_protocol_gap.csv'), index=False)
    return T


# --------------------------------------------------------------------------- #
# T-B: ablation 2x2
# --------------------------------------------------------------------------- #
CELLS_2x2 = {
    ('no_FT', 'no_CW'): "R-HGT Strict Control (No Fine-Tuning)",
    ('no_FT', 'CW'):    "R-HGT Stage 5b (class-weighted, no FT)",
    ('FT',    'no_CW'): "R-HGT Stage 6a (FT, no class weight)",
    ('FT',    'CW'):    "R-HGT Stage 6 (+ Strict-Inductive Fine-Tuning)",
}


def table_B(d, out_dir):
    f1 = pivot(d, 'F1'); ap = pivot(d, 'PR-AUC')
    pr = pivot(d, 'Precision'); rc = pivot(d, 'Recall')
    rows = []
    for (ft, cw), cfg in CELLS_2x2.items():
        if cfg not in f1.columns: continue
        rows.append({'fine_tune': ft, 'class_weight': cw, 'Config': cfg,
                     'F1': f1[cfg].mean(), 'F1_sd': f1[cfg].std(ddof=1),
                     'Precision': pr[cfg].mean(), 'Recall': rc[cfg].mean(),
                     'AP': ap[cfg].mean(), 'AP_sd': ap[cfg].std(ddof=1)})
    T = pd.DataFrame(rows)
    T.round(6).to_csv(os.path.join(out_dir, 'TB_ablation_2x2.csv'), index=False)

    base = CELLS_2x2[('no_FT', 'no_CW')]
    contrasts, praw = [], []
    for name, cfg in [("class-weight effect (no FT)", CELLS_2x2[('no_FT', 'CW')]),
                      ("fine-tune effect (no CW)",       CELLS_2x2[('FT', 'no_CW')]),
                      ("both",                          CELLS_2x2[('FT', 'CW')])]:
        if cfg not in f1.columns: continue
        md, t, p, lo, hi, n = paired_t(f1[cfg], f1[base])
        _, pw = wilcoxon_exact(f1[cfg], f1[base])
        mda, *_ = paired_t(ap[cfg], ap[base])
        contrasts.append({'contrast': name, 'vs': base, 'Delta_F1': md,
                          'CI95_low': lo, 'CI95_high': hi, 't': t,
                          'p_paired_t': p, 'p_wilcoxon': pw, 'Delta_AP': mda, 'n': n})
        praw.append(p)
    C = pd.DataFrame(contrasts)
    if len(C): C['p_holm'] = holm(praw)
    C.round(6).to_csv(os.path.join(out_dir, 'TB_contrasts.csv'), index=False)

    # interaction: (FT+CW - FT) - (CW - base)
    if all(c in f1.columns for c in CELLS_2x2.values()):
        inter = ((f1[CELLS_2x2[('FT', 'CW')]] - f1[CELLS_2x2[('FT', 'no_CW')]])
                 - (f1[CELLS_2x2[('no_FT', 'CW')]] - f1[base]))
        md, t, p, lo, hi, n = paired_t(inter, np.zeros(len(inter)))
        pd.DataFrame([{'interaction_F1': md, 'CI95_low': lo, 'CI95_high': hi,
                       't': t, 'p': p, 'n': n}]).round(6).to_csv(
            os.path.join(out_dir, 'TB_interaction.csv'), index=False)
    return T, C


# --------------------------------------------------------------------------- #
# T-C: threshold vs representation, from prob_dumps
# --------------------------------------------------------------------------- #
SWEEP_CONFIGS = sorted({
    "R-HGT Strict Control (No Fine-Tuning)",
    "R-HGT Stage 5b (class-weighted, no FT)",
    "R-HGT Stage 6a (FT, no class weight)",
    "R-HGT Stage 6 (+ Strict-Inductive Fine-Tuning)",
    *[c for _l, cs, ci in PROTOCOL_PAIRS for c in (cs, ci)],
})


def slug(name):
    """File-naming rule of the pipeline's prob_dumps.

    Verified backwards from real file names:
      "Multimodal + HIN R-HGT"                      -> "Multimodal___HIN_R-HGT"
      "R-HGT Strict Control (No Fine-Tuning)"       -> "R-HGT_Strict_Control__No_Fine-Tuning_"
      "R-HGT Stage 5b (class-weighted, no FT)"      -> "R-HGT_Stage_5b__class-weighted__no_FT_"
    i.e. REPLACE EACH character not in [0-9A-Za-z_-] with '_', WITHOUT collapsing consecutive ones.
    """
    return re.sub(r'[^0-9A-Za-z_\-]', '_', name)


def table_C(results_dir, out_dir, seeds):
    pdir = os.path.join(results_dir, 'prob_dumps')
    if not os.path.isdir(pdir):
        print("  [!] no prob_dumps/, skipping T-C"); return None
    have = set(os.listdir(pdir))
    rows = []
    for cfg in SWEEP_CONFIGS:
        s = slug(cfg)
        for seed in seeds:
            fn = f"clean_seed{seed}_{s}.npz"
            if fn not in have:
                cand = [f for f in have if f.startswith(f"clean_seed{seed}_") and slug(f[len(f'clean_seed{seed}_'):-4]) == s]
                if not cand: continue
                fn = cand[0]
            z = np.load(os.path.join(pdir, fn), allow_pickle=True)
            y = z['y_true'].astype(int); p = z['y_prob'][:, 1].astype(float)
            f05 = f1_at(y, p, 0.5)
            fb, tb = best_f1(y, p)
            rows.append({'Config': cfg, 'seed': seed, 'n': len(y),
                         'prevalence': float(y.mean()),
                         'F1@0.5': f05, 'F1@tau*': fb, 'tau*': tb,
                         'AP': average_precision(y, p), 'ROC_AUC': roc_auc(y, p),
                         'regime': str(z['regime'][0]) if 'regime' in z.files else ''})
    if not rows:
        print("  [!] could not match any prob_dumps file"); return None
    R = pd.DataFrame(rows)
    R.round(6).to_csv(os.path.join(out_dir, 'TC_threshold_sweep_per_seed.csv'), index=False)

    S = (R.groupby('Config')
           .agg(n_seed=('seed', 'nunique'),
                F1_05=('F1@0.5', 'mean'), F1_05_sd=('F1@0.5', 'std'),
                F1_best=('F1@tau*', 'mean'), F1_best_sd=('F1@tau*', 'std'),
                tau_best=('tau*', 'median'),
                AP=('AP', 'mean'), ROC=('ROC_AUC', 'mean'))
           .reset_index())
    S['headroom_threshold'] = S['F1_best'] - S['F1_05']
    base = "R-HGT Strict Control (No Fine-Tuning)"
    if base in set(S['Config']):
        b = S[S['Config'] == base].iloc[0]
        S['dF1_vs_control'] = S['F1_05'] - b['F1_05']
        S['dF1best_vs_control'] = S['F1_best'] - b['F1_best']
        S['pct_gain_from_threshold'] = np.where(
            S['dF1_vs_control'].abs() > 1e-9,
            100 * (1 - S['dF1best_vs_control'] / S['dF1_vs_control']), np.nan)
    S.round(6).to_csv(os.path.join(out_dir, 'TC_threshold_summary.csv'), index=False)
    return S, R


# --------------------------------------------------------------------------- #
# T-D: protocol gap MEASURED AT THE OPTIMAL THRESHOLD
#
# This is the most important table. If the gap disappears when each protocol is
# allowed its own best threshold, the phenomenon is an OPERATING-POINT SHIFT,
# not a loss of discriminative ability — and the paper must say exactly that.
# Note: tau* here is chosen ON THE TEST SET, so it is an oracle-style upper bound
# that cannot be deployed; it is used to decompose the cause, not to be reported
# as performance.
# --------------------------------------------------------------------------- #
def table_D(R, out_dir):
    if R is None or not len(R):
        return None
    w05 = R.pivot_table(index='seed', columns='Config', values='F1@0.5')
    wbe = R.pivot_table(index='seed', columns='Config', values='F1@tau*')
    wap = R.pivot_table(index='seed', columns='Config', values='AP')
    rows, p05, pbe = [], [], []
    for label, cs, ci in PROTOCOL_PAIRS:
        if cs not in w05.columns or ci not in w05.columns:
            continue
        d05, _t0, q0, l0, h0, n = paired_t(w05[ci], w05[cs])
        dbe, _t1, q1, l1, h1, _ = paired_t(wbe[ci], wbe[cs])
        dap, *_ = paired_t(wap[ci], wap[cs])
        rows.append({
            'Architecture': label, 'n_seed': n,
            'gap_F1@0.5': d05, 'CI_low@0.5': l0, 'CI_high@0.5': h0, 'p@0.5': q0,
            'gap_F1@tau*': dbe, 'CI_low@tau*': l1, 'CI_high@tau*': h1, 'p@tau*': q1,
            'gap_AP': dap,
            'pct_gap_from_threshold': (100 * (1 - dbe / d05)) if abs(d05) > 1e-9 else np.nan,
            'tau*_shared': wbe[cs].notna().sum() and R[R['Config'] == cs]['tau*'].median(),
            'tau*_strict': R[R['Config'] == ci]['tau*'].median(),
        })
        p05.append(q0); pbe.append(q1)
    T = pd.DataFrame(rows)
    if len(T):
        T['p_holm@0.5'] = holm(p05)
        T['p_holm@tau*'] = holm(pbe)
        T = T.sort_values('gap_F1@0.5')
    T.round(6).to_csv(os.path.join(out_dir, 'TD_gap_at_optimal_threshold.csv'), index=False)
    return T


# --------------------------------------------------------------------------- #
def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument('--results-dir', default='.')
    ap_.add_argument('--out-dir', default='./analysis')
    a = ap_.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    d = load_raw(a.results_dir)
    seeds = sorted(d['Run'].unique().tolist())
    print(f"Loaded {len(d)} rows, {d['Config'].nunique()} configurations, {len(seeds)} seed: {seeds}\n")

    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)

    print("=" * 100); print("T-A  PROTOCOL GAP (same checkpoint, paired by seed)"); print("=" * 100)
    TA = table_A(d, a.out_dir)
    if len(TA):
        print(TA[['Architecture', 'F1_shared', 'F1_strict', 'Delta_F1', 'CI95_low',
                  'CI95_high', 't', 'p_holm', 'Delta_AP', 'Prec_strict', 'Rec_strict']]
              .round(4).to_string(index=False))

    print("\n" + "=" * 100); print("T-B  2x2 ABLATION  (all under Protocol I)"); print("=" * 100)
    TB, TBC = table_B(d, a.out_dir)
    print(TB.round(4).to_string(index=False))
    print("\nPaired contrasts against the base cell (no FT, no CW):")
    print(TBC.round(6).to_string(index=False))

    print("\n" + "=" * 100); print("T-C  THRESHOLD vs REPRESENTATION (threshold sweep on prob_dumps)"); print("=" * 100)
    res = table_C(a.results_dir, a.out_dir, seeds)
    TC, R = res if res else (None, None)
    if TC is not None:
        print(TC.round(4).to_string(index=False))

    print("\n" + "=" * 100)
    print("T-D  PROTOCOL GAP MEASURED AT THE OPTIMAL THRESHOLD (oracle) — cause decomposition")
    print("=" * 100)
    TD = table_D(R, a.out_dir)
    if TD is not None and len(TD):
        print(TD[['Architecture', 'gap_F1@0.5', 'gap_F1@tau*', 'gap_AP',
                  'pct_gap_from_threshold', 'tau*_shared', 'tau*_strict',
                  'p_holm@0.5', 'p_holm@tau*']].round(4).to_string(index=False))

    print(f"\n✅ Wrote CSV files to {os.path.abspath(a.out_dir)}")


if __name__ == '__main__':
    main()
