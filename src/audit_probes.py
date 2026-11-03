#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
audit_probes.py — four CPU-only sanity probes on the data.

Four benchmark-audit probes, none of which needs a GPU:

  P1  Is the binary task subject to a provenance shortcut?
      → logistic regression + random forest on RAW FEATURES, group-safe split by Hash.
      If ~0.99, most of the "malware detection" here is distinguishing LLM-generated benign files.

  P2  Does the feature vector leak the CPU architecture identity?
      → classify `arch` from that same vector. If ~1.00, the zero-filled missing columns
      encode the architecture, and since the malware family composition differs by
      architecture, this is a detour to the label.

  P3  Are there near-duplicates across train/test?
      → largest cosine from each test sample to the train set; percentiles are reported.

  P4  What did preprocessing remove?
      → inner-join drop rate by family and by architecture; trace the Botnet family;
      percentiles of rows per sample.

How to run (from the src/ directory of the cleaned-up repo):

    python3 audit_probes.py                      # auto-discover data via rhgt_paths
    python3 audit_probes.py --data-dir <path>
    python3 audit_probes.py --probes 1,2         # run only P1 and P2

Results are written to results/audit/*.csv and the tables are printed to the screen.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")

ARCHS = ("arm", "mips", "mipsel", "x86")
MODALITIES = ("pcap", "sar", "strace")


# --------------------------------------------------------------------------- #
# Data loading — reuses the pipeline logic exactly so that results are comparable
# --------------------------------------------------------------------------- #
def load_merged(data_dir: str):
    """Returns (df_all, df_raw_counts). Prefers the merged parquet if present."""
    import pandas as pd

    cache = os.path.join(data_dir, "cached_merged_multimodal.parquet")
    if os.path.isfile(cache):
        print(f"⚡ Using merged parquet: {cache}")
        return pd.read_parquet(cache), None

    try:
        import duckdb
    except ImportError:
        print("❌ duckdb is required: pip install duckdb")
        sys.exit(1)

    con = duckdb.connect()
    frames, counts = [], []
    for arch in ARCHS:
        adir = os.path.join(data_dir, arch)
        if not os.path.isdir(adir):
            continue
        per_mod = {}
        for mod in MODALITIES:
            p = os.path.join(adir, f"{mod}.parquet")
            if not os.path.isfile(p):
                continue
            desc = con.execute(
                f"DESCRIBE SELECT * FROM read_parquet('{p}', union_by_name=True) LIMIT 0"
            ).df()
            cols = list(desc["column_name"])
            types = dict(zip(desc["column_name"], desc["column_type"].astype(str)))

            # [FIX 28/09] String columns must be aggregated with MIN, NOT AVG(TRY_CAST).
            #
            # The previous version applied AVG(TRY_CAST(... AS DOUBLE)) to EVERY column other than Hash.
            # For string columns TRY_CAST returns NULL and AVG(NULL) returns NULL, so
            # MalwareFamily and Arch survived as all-NaN columns. Then
            # labels_of() read "nan" != "benign" and labelled ALL 17,023 samples as mal-
            # ware — prevalence 1.0000 — and LogisticRegression then broke because the data
            # had only one class. The note "keep categorical columns with FIRST if present"
            # described the intent correctly but was never implemented.
            #
            # Use MIN instead of FIRST/ANY_VALUE so the result is deterministic: a Hash can in
            # principle carry several values, and a non-deterministic function
            # here would give two runs two different labels.
            _NUMERIC = ("INT", "DOUBLE", "FLOAT", "DECIMAL", "REAL",
                        "HUGEINT", "BIGINT", "SMALLINT", "TINYINT", "BOOLEAN")
            parts = []
            n_str = 0
            for c in cols:
                if c == "Hash":
                    continue
                t = types.get(c, "").upper()
                if any(k in t for k in _NUMERIC):
                    parts.append(f'AVG(TRY_CAST("{c}" AS DOUBLE)) AS "{c}"')
                else:
                    parts.append(f'MIN("{c}") AS "{c}"')
                    n_str += 1
            agg = ", ".join(parts)
            df = con.execute(
                f'SELECT "Hash", {agg} FROM read_parquet(\'{p}\', union_by_name=True) '
                f'GROUP BY "Hash"'
            ).df()
            if n_str:
                _sc = [c for c in cols
                       if c != "Hash"
                       and not any(k in types.get(c, "").upper() for k in _NUMERIC)]
                print(f"  {arch:7s} {mod:7s} kept {n_str} categorical columns as-is: {_sc}")
            n_rows = con.execute(
                f"SELECT COUNT(*) FROM read_parquet('{p}', union_by_name=True)"
            ).fetchone()[0]
            counts.append({"arch": arch, "modality": mod,
                           "rows": n_rows, "hashes": len(df),
                           "rows_per_sample": n_rows / max(len(df), 1)})
            per_mod[mod] = df
            print(f"  {arch:7s} {mod:7s} {n_rows:>12,d} rows → {len(df):>7,d} hash")

        if "pcap" not in per_mod:
            continue
        merged = per_mod["pcap"]
        sizes = {"pcap": len(merged)}
        for mod in ("sar", "strace"):
            if mod in per_mod:
                sizes[mod] = len(per_mod[mod])
                merged = merged.merge(per_mod[mod], on="Hash", how="inner",
                                      suffixes=("", f"_{mod}"))
        merged["arch"] = arch
        print(f"  {arch:7s} inner join → {len(merged):,d} samples "
              f"(from the smallest, {min(sizes.values()):,d}, "
              f"lost {min(sizes.values()) - len(merged):,d})")
        frames.append(merged)

    import pandas as pd
    df_all = pd.concat(frames, ignore_index=True)
    return df_all, pd.DataFrame(counts)


def feature_matrix(df):
    """Returns (cleaned float32 X, column names).

    Columns are dropped by BASE NAME, not full name: merging the three modalities adds
    the suffixes _sar and _strace, so MalwareFamily_sar and Arch_strace must also be
    dropped. Otherwise P2 (architecture-leakage probe) would be handed the Arch column as a
    feature and report 1.00 — a meaningless result that looks like a major finding.
    """
    import numpy as np
    drop = {"hash", "arch", "malwarefamily", "malware_family", "family",
            "label", "y"}

    def base(c):
        b = str(c)
        for suf in ("_pcap", "_sar", "_strace"):
            if b.endswith(suf):
                b = b[: -len(suf)]
        return b.lower()

    cols = [c for c in df.columns
            if base(c) not in drop and df[c].dtype.kind in "ifub"]
    X = df[cols].to_numpy(dtype="float64", copy=True)
    X[~np.isfinite(X)] = 0.0
    X = np.clip(X, -1e9, 1e9).astype("float32")
    keep = X.std(axis=0) > 0            # constant columns carry no information
    n_drop = int((~keep).sum())
    if n_drop:
        print(f"  (dropped {n_drop} constant columns, kept {int(keep.sum())})")
        X = X[:, keep]
        cols = [c for c, k in zip(cols, keep) if k]
    return X, cols


def labels_of(df):
    """Binary labels from the malware-family column, with a sanity check.

    This check exists because of a real bug: the family column was once turned into
    NaN by the merge over Hash, so every sample was labelled malware, and the symptom surfaced
    deep inside sklearn as "data contains only one class" — a message
    that does not point to the cause. A broken label must stop the run RIGHT where it breaks.
    """
    import numpy as np
    for c in ("MalwareFamily", "malware_family", "Family"):
        if c not in df.columns:
            continue
        raw = df[c]
        n_null = int(raw.isna().sum())
        if n_null == len(raw):
            raise SystemExit(
                f"❌ Column '{c}' is entirely empty ({len(raw):,d} NaN values).\n"
                f"   Almost certainly because the merge over Hash coerced the string column "
                f"through AVG(TRY_CAST(...)). See the [FIX 28/09] note in "
                f"load_merged().")
        fam = raw.astype(str).str.strip().str.lower()
        y = (fam != "benign").astype(int).to_numpy()
        prev = float(y.mean())
        vc = fam.value_counts()
        if prev in (0.0, 1.0):
            raise SystemExit(
                f"❌ Labels have only ONE class (prevalence = {prev:.4f}).\n"
                f"   Values in column '{c}': {vc.head(10).to_dict()}\n"
                f"   A 'benign' value is required (case-insensitive). "
                f"Stopping — every later probe would be meaningless.")
        if n_null:
            print(f"  [!] {n_null:,d}/{len(raw):,d} NaN values in '{c}' "
                  f"→ counted as malware. Recheck before citing.")
        print(f"  Labels from '{c}': {int((y==0).sum()):,d} benign / "
              f"{int(y.sum()):,d} malware (prevalence {prev:.4f})")
        return y, raw.astype(str)
    raise SystemExit("❌ Malware-family column (MalwareFamily) not found.")


def group_split(groups, y, test_size=0.2, seed=42):
    """Group-safe split by Hash, stratified by label at the group level."""
    import numpy as np
    rng = np.random.default_rng(seed)
    uniq, first_idx = np.unique(groups, return_index=True)
    gy = y[first_idx]
    te_groups = []
    for cls in np.unique(gy):
        g = uniq[gy == cls]
        rng.shuffle(g)
        te_groups.append(g[: int(round(len(g) * test_size))])
    te_groups = set(np.concatenate(te_groups).tolist())
    mask_te = np.array([g in te_groups for g in groups])
    return ~mask_te, mask_te


# --------------------------------------------------------------------------- #
# P1 — provenance shortcut
# --------------------------------------------------------------------------- #
def probe1(df, out_dir, seeds=(10, 42, 100)):
    import numpy as np, pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

    print("\n" + "=" * 70)
    print("P1 — PROVENANCE SHORTCUT PROBE (LLM-generated benign)")
    print("=" * 70)
    X, cols = feature_matrix(df)
    y, _ = labels_of(df)
    groups = df["Hash"].to_numpy()
    print(f"{X.shape[0]:,d} samples × {X.shape[1]:,d} features · "
          f"malware prevalence = {y.mean():.4f}")

    rows = []
    for seed in seeds:
        tr, te = group_split(groups, y, seed=seed)
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        for name, clf in (
            ("LogisticRegression", LogisticRegression(max_iter=2000, n_jobs=-1)),
            ("RandomForest", RandomForestClassifier(
                n_estimators=200, n_jobs=-1, random_state=seed)),
        ):
            clf.fit(Xtr, y[tr])
            p = clf.predict_proba(Xte)[:, 1]
            rows.append({
                "seed": seed, "model": name,
                "accuracy": accuracy_score(y[te], p >= 0.5),
                "f1": f1_score(y[te], p >= 0.5),
                "roc_auc": roc_auc_score(y[te], p),
                "n_test": int(te.sum()),
            })
            print(f"  seed {seed:4d}  {name:20s} "
                  f"Acc {rows[-1]['accuracy']:.4f}  "
                  f"F1 {rows[-1]['f1']:.4f}  "
                  f"ROC-AUC {rows[-1]['roc_auc']:.4f}")

    # ---- P1b: how thin is the shortcut? --------------------------------
    #
    # "A random forest reaches F1 1.000" could still be defended as a sufficiently strong model.
    # A single TREE STUMP — exactly one feature, exactly one threshold — cannot be defended
    # that way. This is the most quotable number of the whole probe, because it shows
    # how easy the task is and WHAT makes it easy.
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier as _RF
    tr, te = group_split(groups, y, seed=seeds[0])
    _rf = _RF(n_estimators=200, n_jobs=-1, random_state=0).fit(X[tr], y[tr])
    imp = pd.Series(_rf.feature_importances_, index=cols).sort_values(ascending=False)
    stump_rows = []
    for f in imp.head(10).index:
        j = cols.index(f)
        d = DecisionTreeClassifier(max_depth=1, random_state=0).fit(X[tr][:, [j]], y[tr])
        stump_rows.append({"feature": f, "importance": float(imp[f]),
                           "f1_stump": float(f1_score(y[te], d.predict(X[te][:, [j]])))})
    d3 = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X[tr], y[tr])
    f1_d3 = float(f1_score(y[te], d3.predict(X[te])))
    sdf = pd.DataFrame(stump_rows).sort_values("f1_stump", ascending=False)
    sdf.to_csv(os.path.join(out_dir, "P1b_single_feature_shortcut.csv"), index=False)
    print("\nP1b — what one feature and one threshold achieve:")
    print(sdf.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\n  Depth-3 decision tree on all features: F1 = {f1_d3:.4f}")
    _best = sdf.iloc[0]
    if _best["f1_stump"] >= 0.90:
        print(f"  🔴 A SINGLE feature ('{_best['feature']}') reaches "
              f"F1 = {_best['f1_stump']:.4f}. This feature must be named explicitly "
              f"trong Limitations.")

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(out_dir, "P1_provenance_shortcut.csv"), index=False)
    m = res.groupby("model")[["accuracy", "f1", "roc_auc"]].agg(["mean", "std"])
    print("\nMean over seeds:\n", m.round(4).to_string())
    best = res["f1"].mean()
    print("\nINTERPRETATION: " + (
        "🔴 F1 ≈ %.3f from a LINEAR model on raw features ⇒ the binary task "
        "on this dataset is mostly provenance discrimination, not malware detection. "
        "State this in Limitations and position the contribution around the PROTOCOL GAP." % best
        if best > 0.97 else
        "🟢 F1 ≈ %.3f — a simple model does NOT solve the task; the difference from the "
        "GNN is a real representational contribution." % best))
    return res


# --------------------------------------------------------------------------- #
# P2 — architecture identity leakage
# --------------------------------------------------------------------------- #
def probe2(df, out_dir, seeds=(10, 42, 100)):
    import numpy as np, pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score

    print("\n" + "=" * 70)
    print("P2 — ARCHITECTURE IDENTITY LEAKAGE PROBE")
    print("=" * 70)
    if "arch" not in df.columns:
        print("⚠️  No 'arch' column — skipped.")
        return None
    X, _ = feature_matrix(df)
    y, _ = labels_of(df)
    a = pd.Categorical(df["arch"]).codes
    groups = df["Hash"].to_numpy()

    rows = []
    for seed in seeds:
        tr, te = group_split(groups, y, seed=seed)
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, n_jobs=-1)
        clf.fit(sc.transform(X[tr]), a[tr])
        acc = accuracy_score(a[te], clf.predict(sc.transform(X[te])))
        rows.append({"seed": seed, "arch_accuracy": acc, "n_classes": len(set(a))})
        print(f"  seed {seed:4d}  arch prediction accuracy = {acc:.4f} "
              f"(chance ≈ {1/len(set(a)):.4f})")

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(out_dir, "P2_architecture_leakage.csv"), index=False)
    m = res["arch_accuracy"].mean()
    print("\nINTERPRETATION: " + (
        "🔴 %.4f ⇒ the feature vector determines the architecture almost perfectly. Since the family "
        "composition of malware differs by architecture, this is a detour to the label. Recommendation: take the INTERSECTION of "
        "the columns common to all 4 architectures, or add explicit missing-indicator columns." % m
        if m > 0.95 else
        "🟢 %.4f — no significant architecture leakage." % m))

    # family distribution by architecture — this table must appear in the paper
    _, fam = labels_of(df)
    ct = pd.crosstab(df["arch"], fam)
    ct.to_csv(os.path.join(out_dir, "P2_family_by_arch.csv"))
    print("\nFamily distribution by architecture:\n", ct.to_string())
    return res


# --------------------------------------------------------------------------- #
# P3 — near-duplicates across train/test
# --------------------------------------------------------------------------- #
def probe3(df, out_dir, seed=42, block=2048):
    import numpy as np, pandas as pd
    from sklearn.preprocessing import StandardScaler

    print("\n" + "=" * 70)
    print("P3 — NEAR-DUPLICATES ACROSS TRAIN/TEST")
    print("=" * 70)
    X, _ = feature_matrix(df)
    y, _ = labels_of(df)
    groups = df["Hash"].to_numpy()
    tr, te = group_split(groups, y, seed=seed)

    sc = StandardScaler().fit(X[tr])
    A = sc.transform(X[te]); B = sc.transform(X[tr])
    A /= (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B /= (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)

    best = np.empty(len(A), dtype="float32")
    for i in range(0, len(A), block):
        best[i:i + block] = (A[i:i + block] @ B.T).max(axis=1)

    q = np.percentile(best, [50, 75, 90, 95, 99, 100])
    res = pd.DataFrame({"percentile": [50, 75, 90, 95, 99, 100],
                        "max_cosine_to_train": q})
    res.to_csv(os.path.join(out_dir, "P3_near_duplicate.csv"), index=False)
    print(res.round(4).to_string(index=False))
    frac = float((best > 0.999).mean())
    print(f"\nFraction of test samples with cosine > 0.999 to a train sample: {frac:.4%}")
    print("INTERPRETATION: " + (
        "🔴 near-duplicates across the split despite the group-safe split by Hash — "
        "must be reported; consider deduplication by a cosine threshold."
        if frac > 0.01 else
        "🟢 no evidence of significant near-duplication."))
    return res


# --------------------------------------------------------------------------- #
# P4 — preprocessing audit
# --------------------------------------------------------------------------- #
def probe4(df, counts, out_dir):
    import pandas as pd
    print("\n" + "=" * 70)
    print("P4 — PREPROCESSING AUDIT")
    print("=" * 70)
    if counts is not None and len(counts):
        counts.to_csv(os.path.join(out_dir, "P4_rows_per_sample.csv"), index=False)
        print("Raw row count / sample count after merging by Hash:\n",
              counts.round(1).to_string(index=False))
    else:
        print("⚠️  Using the merged parquet, so raw row counts are unavailable.")
        print("    Rerun with --data-dir pointing to the RAW data to get this table.")

    _, fam = labels_of(df)
    tab = (fam.value_counts().rename_axis("family")
           .reset_index(name="n_samples"))
    tab["pct"] = 100 * tab["n_samples"] / len(df)
    tab.to_csv(os.path.join(out_dir, "P4_family_counts.csv"), index=False)
    print("\nSamples per family after preprocessing:\n", tab.to_string(index=False))

    names = {s.lower() for s in fam.unique()}
    missing = [f for f in ("botnet", "agent", "rudedevil", "tsunami")
               if f not in names]
    if missing:
        print(f"\n🔴 Families in the original dataset paper that are ABSENT here: {missing}")
        print("   Must be explained in the paper — losing a whole family signals filtering bias.")
    else:
        print("\n🟢 All expected families are present.")
    return tab


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out-dir", default="../results/audit")
    ap.add_argument("--probes", default="1,2,3,4")
    ap.add_argument("--seeds", default="10,42,100")
    args = ap.parse_args()

    data_dir = args.data_dir
    if data_dir is None:
        try:
            import rhgt_paths
            data_dir, _kind, cands = rhgt_paths.find_data_dir(verbose=False)
            rhgt_paths.report(cands)
        except ImportError:
            pass
    if not data_dir or not os.path.isdir(data_dir):
        print("❌ Data not found. Pass --data-dir.")
        return 1

    os.makedirs(args.out_dir, exist_ok=True)
    seeds = tuple(int(s) for s in args.seeds.split(","))
    want = {p.strip() for p in args.probes.split(",")}

    print(f"\nLoading data from {data_dir} …")
    df, counts = load_merged(data_dir)
    print(f"→ {len(df):,d} samples, {df.shape[1]:,d} columns")

    if "1" in want: probe1(df, args.out_dir, seeds)
    if "2" in want: probe2(df, args.out_dir, seeds)
    if "3" in want: probe3(df, args.out_dir, seeds[0])
    if "4" in want: probe4(df, counts, args.out_dir)

    print(f"\n✅ Done. Results in {os.path.abspath(args.out_dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
