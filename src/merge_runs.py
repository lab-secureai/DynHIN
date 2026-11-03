#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_runs.py — Merges results from several parallel runs (multiple Kaggle accounts,
                multiple sessions, or both) into a single set of raw CSVs.

    python3 merge_runs.py --inputs acc1/experiments acc2/experiments acc3/experiments \\
                          --out-dir merged --run-tables

Why merging is safe
-----------------------
Each seed in the pipeline is an INDEPENDENT run: its own data split, its own
initialisation, its own training. No state is shared between seeds. Hence
running seed 10 on machine A and seed 42 on machine B gives exactly the same results as
running both on one machine. The only condition: each unit of work must be run in ONE
place, or if it is duplicated the duplicate must be detectable — that is this script's job.

The script checks
---------------
  * Duplicate work: the same key appears in several sources.
  * CONFLICTS: the same key but different numbers (a sign of a duplicate run with a
    different configuration, or of a source that was edited by hand).
  * Gaps: any seed or (seed, family) pair that is missing from every source.
  * Heterogeneous hardware: if the sources ran on different GPUs, the timing
    and memory columns are NOT comparable with each other (see --env-file).

Dependencies: pandas, numpy.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Locations the pipeline writes to, in order of priority when probing.
LAYOUTS = {
    "overall": [
        "raw/overall_robustness_raw_10seed.csv",
        "robustness_global/overall_robustness_raw_10seed.csv",
        "overall_robustness_raw_10seed.csv",
    ],
    "perarch": [
        "raw/per_architecture_robustness_raw_10seed.csv",
        "robustness_per_architecture/per_architecture_robustness_raw_10seed.csv",
        "per_architecture_robustness_raw_10seed.csv",
    ],
    "lofo": [
        "raw/lofo_zeroday_generalization_raw_10seed.csv",
        "lofo_zeroday_generalization/lofo_zeroday_generalization_raw_10seed.csv",
        "lofo_zeroday_generalization_raw_10seed.csv",
    ],
}

# Key that uniquely identifies a unit of work in each file type.
KEYS = {
    "overall": ["Config", "Run", "Architecture", "Noise_Type", "Noise_Level"],
    "perarch": ["Config", "Run", "Architecture", "Noise_Type", "Noise_Level"],
    "lofo": ["Config", "Seed", "Held_Out_Family", "Architecture"],
}

# Columns used to detect conflicts (different numbers for the same key).
CHECK_COLS = ["F1", "Accuracy", "ROC-AUC"]

REPORT: List[str] = []


def say(s: str = "") -> None:
    print(s)
    REPORT.append(s)


def find_file(root: str, kind: str) -> Optional[str]:
    for rel in LAYOUTS[kind]:
        p = os.path.join(root, rel)
        if os.path.exists(p):
            return p
    return None


def load_source(root: str, kind: str, tag: str) -> Optional[pd.DataFrame]:
    p = find_file(root, kind)
    if p is None:
        return None
    try:
        d = pd.read_csv(p)
    except Exception as e:                                   # noqa: BLE001
        say(f"    [!] Could not read {p}: {e}")
        return None
    if d.empty:
        return None
    d["__source"] = tag
    d["__path"] = p
    return d


def merge_kind(frames: List[pd.DataFrame], kind: str,
               prefer: Optional[List[str]] = None) -> Tuple[pd.DataFrame, dict]:
    keys = [k for k in KEYS[kind] if all(k in f.columns for f in frames)]
    if not keys:
        say(f"    [!] No usable key columns for '{kind}' — skipping merge.")
        return pd.concat(frames, ignore_index=True), {}

    allrows = pd.concat(frames, ignore_index=True)
    allrows["__key"] = allrows[keys].astype(str).agg("|".join, axis=1)

    stats = {"total rows loaded": len(allrows), "distinct keys": allrows["__key"].nunique()}

    # --- duplicated work ---
    per_key_src = allrows.groupby("__key")["__source"].nunique()
    dup_keys = per_key_src[per_key_src > 1].index
    stats["keys run in several sources"] = int(len(dup_keys))

    # --- conflicting numbers ---
    conflicts = []
    if len(dup_keys):
        sub = allrows[allrows["__key"].isin(dup_keys)]
        for col in [c for c in CHECK_COLS if c in sub.columns]:
            g = sub.groupby("__key")[col].agg(
                lambda s: float(np.nanmax(s) - np.nanmin(s)) if s.notna().any() else 0.0)
            bad = g[g > 1e-6]
            for k, spread in bad.items():
                conflicts.append({"key": k, "metric": col, "spread": float(spread)})
    stats["conflicting numbers"] = len(conflicts)

    # Who wins when a key exists in several sources.
    #
    # The previous version used keep="first", meaning whichever source was typed first on the
    # command line won — exactly what this script itself warns against ("do not let the
    # script choose"). The order of command-line arguments is not a scientific decision. Now the
    # priority order must be stated explicitly with --prefer, and it is PRINTED so that it
    # appears in the experiment log.
    if prefer:
        rank = {t: i for i, t in enumerate(prefer)}
        fallback = len(rank)
        allrows["__rank"] = allrows["__source"].map(lambda t: rank.get(t, fallback))
        allrows = allrows.sort_values("__rank", kind="mergesort")
        stats["priority order"] = " > ".join(prefer)
    else:
        stats["priority order"] = "(--prefer not set: following the --inputs order)"

    merged = allrows.drop_duplicates(subset="__key", keep="first").copy()
    merged = merged.drop(columns=[c for c in ("__key", "__rank") if c in merged.columns])
    return merged, {"stats": stats, "conflicts": conflicts, "keys": keys,
                    "per_source": allrows.groupby("__source").size().to_dict()}


def coverage_report(df: pd.DataFrame, kind: str) -> None:
    if kind in ("overall", "perarch"):
        if "Run" not in df.columns:
            return
        seeds = sorted(int(v) for v in
                       pd.to_numeric(df["Run"], errors="coerce").dropna().unique())
        say(f"    Seeds present ({len(seeds)}): {seeds}")
        if "Noise_Type" in df.columns:
            t = df.pivot_table(index="Run", columns="Noise_Type",
                               values="F1", aggfunc="count").fillna(0).astype(int)
            say("    Record counts by seed × noise type:")
            for line in t.to_string().splitlines():
                say("      " + line)
        if "Config" in df.columns:
            per_seed_cfg = df.groupby("Run")["Config"].nunique()
            if per_seed_cfg.nunique() > 1:
                say(f"    [!] Uneven number of configurations across seeds: "
                    f"{per_seed_cfg.to_dict()} — meaning the sources ran with "
                    f"different RHGT_EXTRA. The comparison tables will have empty cells.")
    else:
        if not {"Seed", "Held_Out_Family"}.issubset(df.columns):
            return
        t = df.pivot_table(index="Seed", columns="Held_Out_Family",
                           values="F1", aggfunc="count").fillna(0).astype(int)
        say("    Record counts by seed × held-out family:")
        for line in t.to_string().splitlines():
            say("      " + line)


def _load_env(root: str):
    """Read the environment description of one run, normalised to flat keys.

    The pipeline writes run_config.json with the library versions NESTED in
    cfg['versions']; the first version of merge_runs looked for run_env.json with flat
    keys, so the hardware check NEVER ran — exactly when it matters most, since
    three Kaggle accounts may be assigned three different GPU types. This function accepts
    both file names and both ways of nesting the keys.
    """
    for name in ("run_config.json", "run_env.json"):
        p = os.path.join(root, name)
        if not os.path.exists(p):
            continue
        try:
            cfg = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        v = cfg.get("versions", {}) or {}
        return {
            "gpu": cfg.get("gpu", "?"),
            "torch": cfg.get("torch", v.get("torch", "?")),
            "torch_geometric": cfg.get("torch_geometric",
                                       v.get("torch_geometric", "?")),
            "platform": cfg.get("platform", "?"),
            "env": cfg.get("env", {}) or {},
            "_file": name,
        }
    return None


def check_environments(inputs: List[str], tags: List[str]) -> None:
    """Warn when hardware or configuration is not uniform across sources."""
    envs = {}
    for root, tag in zip(inputs, tags):
        e = _load_env(root)
        if e is not None:
            envs[tag] = e
    if not envs:
        say("\n[!] No source has run_config.json. Hardware CANNOT be checked.")
        say("    If the accounts ran on different GPUs (P100 vs T4), the columns")
        say("    'Training Time (s)', 'Inference (ms/sample)' and 'Peak GPU Memory'")
        say("    are NOT comparable with each other and must not go into the same table.")
        return

    say("\nEnvironment per source:")
    for tag, e in envs.items():
        say(f"    {tag:10s} GPU={e.get('gpu', '?')}  torch={e.get('torch', '?')}  "
            f"PyG={e.get('torch_geometric', '?')}")

    # Different RHGT_* variables are a legitimate reason for two sources to give different numbers.
    keys = sorted({k for e in envs.values() for k in e.get("env", {})})
    diff = [k for k in keys
            if len({str(e.get("env", {}).get(k, "")) for e in envs.values()}) > 1]
    if diff:
        say("\n    Environment variables DIFFER between sources:")
        for k in diff:
            say(f"      {k}")
            for tag, e in envs.items():
                say(f"        {tag:10s} = {e.get('env', {}).get(k, '(not set)')}")

    gpus = {e.get("gpu") for e in envs.values() if e.get("gpu")}
    if len(gpus) > 1:
        say(f"\n    ⚠️  NON-UNIFORM HARDWARE: {sorted(gpus)}")
        say("        The compute-performance table will mix different GPUs. Either rerun")
        say("        the performance measurement on a SINGLE machine, or state the GPU")
        say("        of each seed in the paper.")
    else:
        say(f"    ✅ Same GPU type: {gpus.pop() if gpus else '?'}")

    vers = {(e.get("torch"), e.get("torch_geometric")) for e in envs.values()}
    if len(vers) > 1:
        say(f"    ⚠️  Library versions differ between sources: {sorted(vers)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge results from several parallel runs.")
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="The experiments/ directories to merge.")
    ap.add_argument("--tags", nargs="*", default=None,
                    help="Label for each source (default: directory name).")
    ap.add_argument("--out-dir", default="merged")
    ap.add_argument("--run-tables", action="store_true",
                    help="Run make_tables.py on the merged results right away.")
    ap.add_argument("--prefer", nargs="*", default=None,
                    help="Priority order when a key exists in several sources, given "
                         "by source LABEL, highest first. Example: "
                         "--prefer m1 m3. If not set, the --inputs order is used, "
                         "and the order of command-line arguments is not a scientific decision.")
    args = ap.parse_args()

    missing = [p for p in args.inputs if not os.path.isdir(p)]
    if missing:
        sys.exit(f"Directory not found: {missing}")

    tags = args.tags or [os.path.basename(os.path.normpath(p)) or p for p in args.inputs]
    if len(tags) != len(args.inputs):
        sys.exit("The number of --tags must equal the number of --inputs")
    # make sure the labels are unique
    seen = {}
    for i, t in enumerate(tags):
        if t in seen:
            tags[i] = f"{t}#{i}"
        seen[t] = True

    os.makedirs(os.path.join(args.out_dir, "raw"), exist_ok=True)

    say("=" * 74)
    say("MERGING RESULTS FROM MULTIPLE RUNS")
    say("=" * 74)
    for p, t in zip(args.inputs, tags):
        say(f"  {t:12s} <- {p}")

    check_environments(args.inputs, tags)

    out_names = {
        "overall": "overall_robustness_raw_10seed.csv",
        "perarch": "per_architecture_robustness_raw_10seed.csv",
        "lofo": "lofo_zeroday_generalization_raw_10seed.csv",
    }

    total_conflicts = 0
    for kind in ["overall", "perarch", "lofo"]:
        say(f"\n--- {kind} ---")
        frames = []
        for root, tag in zip(args.inputs, tags):
            d = load_source(root, kind, tag)
            if d is None:
                say(f"    {tag:12s}: no data")
            else:
                say(f"    {tag:12s}: {len(d):6,d} rows  ({os.path.relpath(d['__path'].iloc[0], root)})")
                frames.append(d)
        if not frames:
            say("    → skipped, no source has this file.")
            continue

        merged, info = merge_kind(frames, kind, args.prefer)
        if info:
            say(f"    keys: {info['keys']}")
            for k, v in info["stats"].items():
                say(f"    {k}: {v:,}")
            if info["conflicts"]:
                total_conflicts += len(info["conflicts"])
                say(f"    ⚠️  {len(info['conflicts'])} CONFLICTS — same key, different numbers:")
                for c in info["conflicts"][:10]:
                    say(f"        {c['metric']} differs by {c['spread']:.6f}  ::  {c['key'][:80]}")
                if len(info["conflicts"]) > 10:
                    say(f"        ... and {len(info['conflicts']) - 10} more")
                say("        Common cause: two accounts ran the same seed")
                say("        but with different RHGT_ATTACK_RESTARTS or EXTRA groups.")
                say("        Decide which source to keep and drop the other; do not let")
                say("        the script choose.")

        merged = merged.drop(columns=[c for c in ["__source", "__path"] if c in merged.columns])
        dst = os.path.join(args.out_dir, "raw", out_names[kind])
        merged.to_csv(dst, index=False)
        say(f"    → {dst}  ({len(merged):,} rows)")
        coverage_report(merged, kind)

    p = os.path.join(args.out_dir, "MERGE_REPORT.md")
    with open(p, "w", encoding="utf-8") as f:
        f.write("# Merge report\n\n```\n" + "\n".join(REPORT) + "\n```\n")
    print(f"\n✅ Report: {p}")

    if total_conflicts:
        print(f"⚠️  {total_conflicts} conflicts — read the report BEFORE generating the tables.")

    if args.run_tables:
        import subprocess
        here = os.path.dirname(os.path.abspath(__file__))
        mt = os.path.join(here, "make_tables.py")
        if not os.path.exists(mt):
            mt = "make_tables.py"
        print("\nRunning make_tables.py ...")
        subprocess.run([sys.executable, mt,
                        "--raw-dir", os.path.join(args.out_dir, "raw"),
                        "--out-dir", os.path.join(args.out_dir, "tables_generated")])


if __name__ == "__main__":
    main()
