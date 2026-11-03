#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dedup_reeval.py — remove near-duplicate samples across train/test and RE-MEASURE, no GPU needed.

Why this is needed. The P3 probe showed that 7.05% of test samples have cosine > 0.999
to some train sample even though the split is group-safe by Hash — because Hash is
UNIQUE for all 17,023 samples, so splitting by Hash cannot prevent near-duplicate
samples that carry different hashes. The question is: is the
protocol gap merely a consequence of that cross-contamination?

Why no GPU is needed. The pipeline stores y_prob for every test sample in
prob_dumps/*.npz. Dropping a subset of test samples and recomputing the metrics
is an array computation, not a training run.

Requirement: the prob dumps must contain a `hash` field (patch K1). Dumps produced
BEFORE that patch cannot be used, and the script says so explicitly instead of
guessing. Why guessing is impossible: the test-set order CANNOT be reproduced
outside the original run — the split uses StratifiedGroupKFold(shuffle=True), the
result depends on the row order of the DataFrame, and that order is decided by
DuckDB during GROUP BY and changes between DuckDB versions. Verified: a
reconstruction with DuckDB 1.5.5 differs by 1,424/3,404 labels from the Kaggle
run (DuckDB 1.3.2), even though the test-set size and the stratum distribution
match exactly.

Usage:

    python3 dedup_reeval.py --data-dir <raw parquet directory> \
        --results <results-l3 directory> --out-dir results/dedup

    # different threshold, or drop the twin train samples instead of the test samples:
    python3 dedup_reeval.py ... --threshold 0.9999 --side test

Requirements: duckdb, pandas, numpy, scikit-learn. torch is not needed.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

PCAP_NAMES = ['Header_Length', 'Protocol Type', 'Time_To_Live', 'Rate', 'Tot sum',
              'Min', 'Max', 'AVG', 'Std', 'Tot size', 'IAT', 'Number', 'Variance']
PROTO_NAMES = ['TCP', 'UDP', 'ICMP', 'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SSH',
               'IRC', 'DHCP', 'ARP', 'IGMP', 'IPv', 'LLC']


# --------------------------------------------------------------------------- #
# 1. Load the data and reproduce the PIPELINE's split
# --------------------------------------------------------------------------- #
ARCHS = ('arm', 'mips', 'mipsel', 'x86')
_NUMERIC_SQL = ['DOUBLE', 'FLOAT', 'REAL', 'DECIMAL', 'TINYINT', 'SMALLINT',
                'INTEGER', 'BIGINT', 'HUGEINT', 'UTINYINT', 'USMALLINT',
                'UINTEGER', 'UBIGINT']


def _read_one(con, path: str) -> pd.DataFrame:
    """Verbatim copy of the pipeline's read_parquet_with_duckdb().

    Every detail must be copied, including FIRST() and the column order in the
    statement, because StratifiedGroupKFold(shuffle=True) depends on ROW ORDER,
    and the row order DuckDB returns depends on the GROUP BY statement itself.
    A "semantically equivalent" loader is not enough: the first attempt differed
    by 1,396/3,404 labels even though the test-set size matched exactly.
    """
    if not os.path.exists(path):
        return pd.DataFrame()
    desc = con.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{path}', union_by_name=True) "
        f"LIMIT 0").df()
    num, cat = [], []
    for _, row in desc.iterrows():
        c, t = row['column_name'], str(row['column_type']).upper()
        if c == 'Hash':
            continue
        if any(k in t for k in _NUMERIC_SQL):
            num.append(c)
        elif c in ['MalwareFamily', 'Arch', 'arch', 'label']:
            cat.append(c)
    if 'Hash' not in desc['column_name'].values or not num:
        return con.execute(
            f"SELECT * FROM read_parquet('{path}', union_by_name=True)").df()
    parts = [f'AVG(TRY_CAST("{c}" AS FLOAT)) as "{c}"' for c in num]
    parts += [f'FIRST("{c}") as "{c}"' for c in cat]
    q = (f'SELECT "Hash", {", ".join(parts)} FROM '
         f"read_parquet('{path}', union_by_name=True) GROUP BY \"Hash\"")
    return con.execute(q).df()


def load_merged(data_dir: str) -> pd.DataFrame:
    """Copy of the pipeline's merge, preserving row order."""
    import duckdb
    con = duckdb.connect(database=':memory:')
    frames = []
    for arch in ARCHS:
        adir = os.path.join(data_dir, arch)
        p_pcap = os.path.join(adir, 'pcap.parquet')
        if not os.path.exists(p_pcap):
            continue
        d_pcap = _read_one(con, p_pcap)
        d_sar = _read_one(con, os.path.join(adir, 'sar.parquet'))
        d_str = _read_one(con, os.path.join(adir, 'strace.parquet'))
        if not d_sar.empty and not d_str.empty:
            c_sar = [c for c in d_sar.columns
                     if c not in d_pcap.columns or c == 'Hash']
            c_str = [c for c in d_str.columns
                     if (c not in d_pcap.columns and c not in d_sar.columns)
                     or c == 'Hash']
            m = d_pcap.merge(d_sar[c_sar], on='Hash', how='inner')
            m = m.merge(d_str[c_str], on='Hash', how='inner')
        else:
            m = d_pcap
        m['arch'] = arch
        frames.append(m)
        print(f"  {arch:7s} → {len(m):,d} samples")
    con.close()
    if not frames:
        sys.exit(f"❌ No parquet found in {data_dir}")
    return pd.concat(frames, ignore_index=True).copy()


def feature_groups(all_cols):
    """Exactly reproduces how the pipeline groups columns (code-rhgt-v2.py, section 2270)."""
    syscall = [c for c in all_cols
               if c.startswith('Call_') or 'syscall' in c.lower()]
    proto = [c for c in PROTO_NAMES if c in all_cols]
    exclude = set(['label', 'arch', 'Arch', 'Hash', 'MalwareFamily',
                   'stratify_col'] + syscall + proto)
    pcap = [c for c in all_cols if c in PCAP_NAMES
            or c.endswith('_flag_number') or c.endswith('_count')]
    sar = [c for c in all_cols if c not in exclude and c not in pcap]
    return pcap, sar, syscall, proto


def prepare(df: pd.DataFrame):
    """Reconstructs df_valid exactly as the pipeline does before splitting."""
    if 'label' not in df.columns:
        df['label'] = (df['MalwareFamily'].astype(str).str.lower()
                       != 'benign').astype(int)
    if 'Hash' in df.columns and 'arch' in df.columns:
        df = df.drop_duplicates(subset=['Hash', 'arch']).reset_index(drop=True)
    d = df.copy()
    d['stratify_col'] = d['label'].astype(str) + "_" + d['arch'].astype(str)
    keys = d['stratify_col'].value_counts()[lambda x: x > 1].index
    return d[d['stratify_col'].isin(keys)].copy().reset_index(drop=True)


def split_like_pipeline(df, seed, group_col='Hash', test_size=0.2):
    """Verbatim copy of the pipeline's group_safe_stratified_split()."""
    from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
    strat = df['stratify_col'].astype(str)
    groups = df[group_col].astype(str).values
    n_splits = max(2, int(round(1.0 / float(test_size))))
    n_splits = min(n_splits, int(df[group_col].nunique()))

    best, best_score = None, float('inf')
    try:
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                    random_state=seed)
        gdist = strat.value_counts(normalize=True)
        for tr, te in sgkf.split(df, strat.values, groups=groups):
            td = strat.iloc[te].value_counts(normalize=True)
            al = gdist.index.union(td.index)
            derr = (gdist.reindex(al, fill_value=0.0)
                    - td.reindex(al, fill_value=0.0)).abs().sum()
            serr = abs(len(te) / len(df) - float(test_size))
            sc = float(derr + 2.0 * serr)
            if sc < best_score:
                best_score, best = sc, (tr, te)
    except ValueError:
        best = None
    if best is None:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size,
                                random_state=seed)
        best = next(gss.split(df, strat.values, groups=groups))
    tr, te = best
    return df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 2. Maximum cosine from each test sample to the train set
# --------------------------------------------------------------------------- #
def max_cos_to_train(Xtr: np.ndarray, Xte: np.ndarray, chunk: int = 512):
    """Returns (max_cos, argmax_index) for each row of Xte."""
    a = Xtr / np.maximum(np.linalg.norm(Xtr, axis=1, keepdims=True), 1e-12)
    mx = np.zeros(len(Xte), dtype=np.float64)
    am = np.zeros(len(Xte), dtype=np.int64)
    for i in range(0, len(Xte), chunk):
        b = Xte[i:i + chunk]
        b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
        s = b @ a.T
        mx[i:i + chunk] = s.max(axis=1)
        am[i:i + chunk] = s.argmax(axis=1)
    return mx, am


# --------------------------------------------------------------------------- #
# 3. Recompute the metrics on the filtered test set
# --------------------------------------------------------------------------- #
def metrics(y, p1, keep):
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, roc_auc_score, average_precision_score,
                                 matthews_corrcoef, confusion_matrix)
    y, p1 = y[keep], p1[keep]
    pred = (p1 >= 0.5).astype(int)
    out = dict(n=int(keep.sum()),
               Accuracy=accuracy_score(y, pred),
               Precision=precision_score(y, pred, zero_division=0),
               Recall=recall_score(y, pred, zero_division=0),
               F1=f1_score(y, pred, zero_division=0),
               MCC=matthews_corrcoef(y, pred) if len(set(y)) > 1 else 0.0)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp = int(cm[0, 0]), int(cm[0, 1])
    out['Specificity'] = tn / max(tn + fp, 1)
    if len(set(y)) > 1:
        out['ROC-AUC'] = roc_auc_score(y, p1)
        out['PR-AUC'] = average_precision_score(y, p1)
    else:
        out['ROC-AUC'] = out['PR-AUC'] = float('nan')
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--results", required=True,
                    help="Results directory containing prob_dumps/ (e.g. results-l3).")
    ap.add_argument("--out-dir", default="results/dedup")
    ap.add_argument("--threshold", type=float, default=0.999)
    ap.add_argument("--side", choices=["test", "train"], default="test",
                    help="Drop the contaminated TEST samples (default), or the twin "
                         "TRAIN samples. The test side can be measured from "
                         "prob_dumps; the train side would require retraining, "
                         "so only the counts are reported.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    dumps = os.path.join(args.results, "prob_dumps")
    if not os.path.isdir(dumps):
        sys.exit(f"❌ Not found: {dumps}")

    print("Loading data ...")
    df = prepare(load_merged(args.data_dir))
    pcap, sar, sysc, proto = feature_groups(list(df.columns))
    feat = [c for c in pcap + sar + sysc if df[c].dtype.kind in "ifub"]
    print(f"  {len(df):,d} samples · {len(feat):,d} features "
          f"(pcap {len(pcap)} + sar {len(sar)} + syscall {len(sysc)})")

    seeds = sorted({int(m.group(1)) for f in glob.glob(os.path.join(dumps, "clean_seed*.npz"))
                    for m in [re.search(r"clean_seed(\d+)_", os.path.basename(f))] if m})
    print(f"  seeds found: {seeds}")

    from sklearn.preprocessing import StandardScaler
    rows, summary = [], []
    df = df.set_index(df["Hash"].astype(str), drop=False)
    for seed in seeds:
        probe = sorted(glob.glob(os.path.join(dumps, f"clean_seed{seed}_*.npz")))
        if not probe:
            print(f"  [!] seed {seed}: no dump — skipped")
            continue
        d0 = np.load(probe[0], allow_pickle=True)
        if "hash" not in d0.files:
            sys.exit(
                f"❌ The prob dump for seed {seed} has NO 'hash' field.\n"
                f"   File: {probe[0]}\n"
                "   This dump was produced before patch K1. The test-set order cannot be\n"
                "   reproduced outside the original run (see the top of this file), so there\n"
                "   is no way to tell which samples are contaminated. Re-run the 'dedup'\n"
                "   session with a code-rhgt-v2.py that includes K1 — about 45 GPU-minutes for 3 seeds.")
        te_hash = d0["hash"].astype(str)
        miss = [h for h in te_hash[:50] if h not in df.index]
        if miss:
            sys.exit(f"❌ seed {seed}: {len(miss)}/50 of the first Hashes in the dump "
                     f"are not in the data. Wrong data directory?")
        te = df.loc[te_hash].reset_index(drop=True)
        tr = df[~df["Hash"].astype(str).isin(set(te_hash))].reset_index(drop=True)
        y_ref = d0["y_true"].astype(int)
        y_rep = te["label"].to_numpy().astype(int)
        if not np.array_equal(y_ref, y_rep):
            sys.exit(f"❌ seed {seed}: labels rebuilt from Hash differ from y_true "
                     f"at {int((y_ref != y_rep).sum())} positions — the data does not match the dump.")
        print(f"  seed {seed}: Hash match ✓ ({len(te):,d} test / {len(tr):,d} train)")

        # Clean BEFORE fitting: the raw data has NaN and infinite values, and
        # StandardScaler rejects them at fit time, not at transform time.
        def clean(x):
            x = np.asarray(x, dtype=np.float64)
            x[~np.isfinite(x)] = 0.0
            return np.clip(x, -1e9, 1e9)

        A = clean(tr[feat].to_numpy())
        B = clean(te[feat].to_numpy())
        sc = StandardScaler().fit(A)
        Xtr = clean(sc.transform(A)).astype(np.float32)
        Xte = clean(sc.transform(B)).astype(np.float32)
        mx, am = max_cos_to_train(Xtr, Xte)
        keep = mx <= args.threshold
        n_drop = int((~keep).sum())
        same_label = int((tr["label"].to_numpy()[am[~keep]]
                          == te["label"].to_numpy()[~keep]).sum()) if n_drop else 0
        print(f"  seed {seed:>4d}: test {len(te):,d} · contaminated {n_drop:,d} "
              f"({n_drop / len(te) * 100:.2f}%) · kept {int(keep.sum()):,d} "
              f"· of the contaminated, {same_label}/{n_drop} share the label of their twin")
        summary.append(dict(seed=seed, n_test=len(te), n_contaminated=n_drop,
                            pct=n_drop / len(te) * 100, n_kept=int(keep.sum()),
                            twin_same_label=same_label,
                            median_max_cos=float(np.median(mx))))

        for f in sorted(glob.glob(os.path.join(dumps, f"clean_seed{seed}_*.npz"))):
            d = np.load(f, allow_pickle=True)
            y, p = d["y_true"].astype(int), d["y_prob"]
            p1 = p[:, 1] if p.ndim == 2 else p
            cfg = re.sub(r"^clean_seed\d+_", "", os.path.basename(f))[:-4]
            base = metrics(y, p1, np.ones(len(y), bool))
            ded = metrics(y, p1, keep)
            rows.append(dict(Config=cfg, Run=seed,
                             regime=str(d["regime"][0]) if "regime" in d else "",
                             **{f"{k}_full": v for k, v in base.items()},
                             **{f"{k}_dedup": v for k, v in ded.items()},
                             dF1=ded["F1"] - base["F1"]))

    if not rows:
        sys.exit("❌ Nothing could be computed.")
    R = pd.DataFrame(rows)
    S = pd.DataFrame(summary)
    R.to_csv(os.path.join(args.out_dir, "dedup_per_config_per_seed.csv"), index=False)
    S.to_csv(os.path.join(args.out_dir, "dedup_contamination_summary.csv"), index=False)

    print(f"\n{'=' * 92}\nCROSS-CONTAMINATION (cosine threshold > {args.threshold})\n{'=' * 92}")
    print(S.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n  Mean: {S.pct.mean():.2f}% of test samples contaminated")

    agg = R.groupby("Config").agg(F1_full=("F1_full", "mean"),
                                  F1_dedup=("F1_dedup", "mean"),
                                  dF1=("dF1", "mean"),
                                  sd=("dF1", "std")).sort_values("dF1")
    print(f"\n{'=' * 92}\nF1 BEFORE and AFTER removing contaminated samples (mean over {len(S)} seeds)\n{'=' * 92}")
    print(agg.to_string(float_format=lambda x: f"{x:+.5f}"))
    print(f"\n  max |ΔF1| over all configurations: {agg.dF1.abs().max():.5f}")
    print(f"  median |ΔF1|: {agg.dF1.abs().median():.5f}")
    if agg.dF1.abs().max() < 0.01:
        print("\n  ✅ Removing near-duplicates changes NO conclusion — report this number "
              "in the paper so the cross-contamination question is answered up front.")
    else:
        print("\n  ⚠️  Some configuration changes by > 0.01 F1 — it must be stated in the paper and discussed.")
    print(f"\n✅ {args.out_dir}/dedup_per_config_per_seed.csv")
    print(f"✅ {args.out_dir}/dedup_contamination_summary.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
