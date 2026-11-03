#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
measure_params.py — MEASURES (does not estimate) parameter counts, model size and
                    FLOPs for every architecture in the HIN-Dynamic pipeline.

Why this exists: the raw CSV files have NO 'Parameters' column, and the pipeline
never calls sum(p.numel()). Every parameter count in the current draft is a
hand estimate, and the estimates contradict each other (Late Fusion was listed
with fewer parameters than the sum of the three branches it contains). This
script measures them.

Usage
-----
    # Take feature dimensions directly from the cached parquet (recommended):
    python3 measure_params.py \
        --code code-10seed-strict-inductive-rhgt-sota-FINETUNED.py \
        --parquet cached_merged_multimodal.parquet

    # Or pass them by hand if there is no parquet:
    python3 measure_params.py \
        --code code-10seed-strict-inductive-rhgt-sota-FINETUNED.py \
        --pcap-dim 40 --sar-dim 300 --strace-dim 160 \
        --n-syscall 160 --n-proto 8

Output: params_measured.csv / .md / .tex

Requirements: torch, torch_geometric, pandas. FLOPs are optional (thop or fvcore).

WARNING: importing the pipeline file runs its module-level code (creates
directories, prints logs, probes the GPU). Its main() is guarded by __name__, so
the benchmark does NOT run. To avoid side effects, use --dry-run, which only
prints the formulas.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from typing import Dict, List, Optional

import pandas as pd


# =========================================================================
# Analytic formulas — used to cross-check the measured counts, and to run when
# torch is unavailable. Derived directly from the layer definitions in the
# pipeline (hidden = 64).
# =========================================================================

def analytic_counts(d_pcap: int, d_sar: int, d_strace: int, d_all: int) -> List[dict]:
    pcap = 64 * d_pcap + 2_338          # Linear+BN+Linear+classifier
    sar = 128 * d_sar + 8_770           # Linear+BN+Linear+classifier
    strace = 50_626                     # Conv1d+BN+BiLSTM+fc — INDEPENDENT of d_strace
    fusion_head = 31_493                # 3 proj + attn_net + classifier
    fusion = pcap + sar + strace + fusion_head
    sage = 192 * d_all + 8_770          # 2×SAGEConv + lin_res + 2×LayerNorm + clf
    return [
        {"Model": "PCAP MLP", "Analytic": pcap,
         "Formula": "64·d_pcap + 2338"},
        {"Model": "SAR Encoder", "Analytic": sar,
         "Formula": "128·d_sar + 8770"},
        {"Model": "STRACE CNN-LSTM", "Analytic": strace,
         "Formula": "50626 (constant, independent of d_strace)"},
        {"Model": "Multimodal Late Fusion", "Analytic": fusion,
         "Formula": "PCAP + SAR + STRACE + 31493"},
        {"Model": "Res-GraphSAGE (non-HIN)", "Analytic": sage,
         "Formula": "192·d_all + 8770"},
    ]


# =========================================================================
# Infer feature dimensions with the pipeline's exact heuristic
# =========================================================================

PCAP_NAMES = ['Header_Length', 'Protocol Type', 'Time_To_Live', 'Rate', 'Tot sum',
              'Min', 'Max', 'AVG', 'Std', 'Tot size', 'IAT', 'Number', 'Variance']
PROTO_NAMES = ['TCP', 'UDP', 'ICMP', 'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SSH',
               'IRC', 'DHCP', 'ARP', 'IGMP', 'IPv', 'LLC']


def dims_from_parquet(path: str) -> dict:
    """Exactly reproduces how the pipeline groups columns (lines 2038–2043)."""
    cols = list(pd.read_parquet(path, engine="pyarrow").columns)
    syscall = [c for c in cols if c.startswith('Call_') or 'syscall' in c.lower()]
    proto = [c for c in PROTO_NAMES if c in cols]
    exclude = set(['label', 'arch', 'Arch', 'Hash', 'MalwareFamily',
                   'stratify_col'] + syscall + proto)
    pcap = [c for c in cols if c in PCAP_NAMES
            or c.endswith('_flag_number') or c.endswith('_count')]
    sar = [c for c in cols if c not in exclude and c not in pcap]
    d = dict(d_pcap=len(pcap), d_sar=len(sar), d_strace=len(syscall),
             n_syscall=max(len(syscall), 1), n_proto=len(proto), n_arch=4,
             d_all=len(pcap) + len(sar) + len(syscall))
    print("Feature dimensions inferred from the parquet:")
    for k, v in d.items():
        print(f"    {k:12s} = {v}")
    print("\nSAR column list (check by hand whether any identifier column slipped in):")
    print("   ", sar[:40], "..." if len(sar) > 40 else "")
    for name, lst in (("pcap", pcap), ("sar", sar), ("syscall", syscall)):
        if not lst:
            print(f"  [!] WARNING: feature group '{name}' is EMPTY — the heuristic "
                  f"does not match the parquet schema.")
    return d


# =========================================================================
# Auto-detect the pipeline file and infer dimensions from the RAW data directory
# =========================================================================

PIPELINE_NAMES = (
    "code-rhgt-v2.py",                                    # current canonical version
    "code-10seed-strict-inductive-rhgt-sota-FINETUNED.py",  # original unpatched version
)


def find_pipeline(explicit: Optional[str]) -> Optional[str]:
    """Find the pipeline file. Returns None if not found.

    An earlier version defaulted to exactly ONE name — that of the original
    unpatched file — so when run on Kaggle with the hin-code bundle (which
    contains code-rhgt-v2.py) it reported "pipeline file not found" and then
    silently printed only the analytic formulas. The analytic formulas contain
    NO HIN model at all, i.e. exactly the part the paper needs is missing.
    """
    if explicit and os.path.isfile(explicit):
        return explicit
    roots = [os.getcwd(), os.path.dirname(os.path.abspath(__file__))]
    for r in roots:
        for n in PIPELINE_NAMES:
            p = os.path.join(r, n)
            if os.path.isfile(p):
                return p
    inp = "/kaggle/input"
    if os.path.isdir(inp):
        for root, dirs, files in os.walk(inp):
            if root[len(inp):].count(os.sep) >= 4:
                dirs[:] = []
                continue
            for n in PIPELINE_NAMES:
                if n in files:
                    return os.path.join(root, n)
    return None


_NUMERIC_SQL = ("INT", "DOUBLE", "FLOAT", "DECIMAL", "REAL",
                "HUGEINT", "BIGINT", "SMALLINT", "TINYINT", "BOOLEAN")


def dims_from_data_dir(data_dir: str) -> Optional[dict]:
    """Infer dimensions from the RAW parquet directory WITHOUT loading data.

    Reproduces the pipeline's code path: read_parquet_with_duckdb() keeps only
    NUMERIC columns (plus MalwareFamily/Arch/label), then the three modalities
    are merged on Hash with duplicate columns dropped explicitly. So the final
    column set = union of the numeric columns over all architectures and all
    modalities. Only DESCRIBE is needed; no rows are read.
    """
    try:
        import duckdb
    except ImportError:
        print("[!] duckdb is not installed — cannot infer dimensions from raw data.")
        return None
    archs = ("arm", "mips", "mipsel", "x86")
    mods = ("pcap", "sar", "strace")
    con = duckdb.connect()
    per_mod_cols = {m: set() for m in mods}
    n_files = 0
    for a in archs:
        for m in mods:
            p = os.path.join(data_dir, a, f"{m}.parquet")
            if not os.path.isfile(p):
                continue
            n_files += 1
            d = con.execute(
                f"DESCRIBE SELECT * FROM read_parquet('{p}', union_by_name=True) "
                f"LIMIT 0").df()
            for c, t in zip(d["column_name"], d["column_type"].astype(str)):
                if c == "Hash":
                    continue
                if any(k in t.upper() for k in _NUMERIC_SQL):
                    per_mod_cols[m].add(c)
    if not n_files:
        print(f"[!] No parquet found in {data_dir}")
        return None

    # The pipeline drops duplicate columns in the order pcap → sar → strace.
    cols_pcap = per_mod_cols["pcap"]
    cols_sar = per_mod_cols["sar"] - cols_pcap
    cols_strace = per_mod_cols["strace"] - cols_pcap - cols_sar
    all_cols = sorted(cols_pcap | cols_sar | cols_strace)

    syscall = [c for c in all_cols if c.startswith("Call_") or "syscall" in c.lower()]
    proto = [c for c in PROTO_NAMES if c in all_cols]
    exclude = set(["label", "arch", "Arch", "Hash", "MalwareFamily",
                   "stratify_col"] + syscall + proto)
    pcap = [c for c in all_cols if c in PCAP_NAMES
            or c.endswith("_flag_number") or c.endswith("_count")]
    sar = [c for c in all_cols if c not in exclude and c not in pcap]
    d = dict(d_pcap=len(pcap), d_sar=len(sar), d_strace=len(syscall),
             n_syscall=max(len(syscall), 1), n_proto=len(proto), n_arch=4,
             d_all=len(pcap) + len(sar) + len(syscall))
    print(f"Feature dimensions inferred from {n_files} raw parquet files in {data_dir}:")
    for k, v in d.items():
        print(f"    {k:12s} = {v}")
    print(f"\n  Protocols recognised ({len(proto)}): {proto}")
    print(f"  PCAP ({len(pcap)}): {pcap}")
    print(f"\n  SAR takes ALL remaining columns — check by hand whether any identifier column slipped in:")
    for i in range(0, min(len(sar), 60), 6):
        print("    " + ", ".join(sar[i:i + 6]))
    if len(sar) > 60:
        print(f"    ... {len(sar) - 60} more columns")
    for name, lst in (("pcap", pcap), ("sar", sar), ("syscall", syscall)):
        if not lst:
            print(f"  [!] WARNING: group '{name}' is EMPTY — the heuristic does not match the schema.")
    return d


# =========================================================================
# Real measurement with torch
# =========================================================================

def load_pipeline(code_path: str):
    spec = importlib.util.spec_from_file_location("hin_pipeline", code_path)
    if spec is None or spec.loader is None:
        sys.exit(f"Could not load {code_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["hin_pipeline"] = mod
    spec.loader.exec_module(mod)          # main() is guarded by __name__
    return mod


def build_dummy_graph(M, n_files: int, d_file: int,
                      n_syscall: int, n_proto: int, n_arch: int):
    """Dummy HIN graph with the exact schema of build_hetero_graph_hin + ToUndirected."""
    import torch
    import torch_geometric.transforms as T
    from torch_geometric.data import HeteroData

    g = HeteroData()
    g['file'].x = torch.randn(n_files, d_file)
    g['file'].y = torch.randint(0, 2, (n_files,))
    g['syscall'].x = torch.eye(n_syscall)
    g['arch'].x = torch.eye(n_arch)
    if n_proto > 0:
        g['protocol'].x = torch.eye(n_proto)

    ar = torch.arange(n_files)
    g['file', 'similar', 'file'].edge_index = torch.stack(
        [ar, torch.roll(ar, 1)], dim=0)
    g['file', 'executes', 'syscall'].edge_index = torch.stack(
        [ar % n_files, ar % n_syscall], dim=0)
    g['file', 'runs_on', 'arch'].edge_index = torch.stack(
        [ar % n_files, ar % n_arch], dim=0)
    if n_proto > 0:
        g['file', 'uses_proto', 'protocol'].edge_index = torch.stack(
            [ar % n_files, ar % n_proto], dim=0)
    return T.ToUndirected()(g)


def count(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    buffers = sum(b.numel() for b in model.buffers())
    bytes_ = sum(p.numel() * p.element_size() for p in model.parameters())
    bytes_ += sum(b.numel() * b.element_size() for b in model.buffers())
    return {"Parameters": total, "Trainable": trainable, "Buffers": buffers,
            "Size (MB, fp32)": round(bytes_ / (1024 ** 2), 3)}


def try_flops(fn, inputs) -> Optional[float]:
    try:
        from fvcore.nn import FlopCountAnalysis
        return float(FlopCountAnalysis(fn, inputs).total())
    except Exception:
        pass
    try:
        from thop import profile
        macs, _ = profile(fn, inputs=inputs, verbose=False)
        return float(macs) * 2.0
    except Exception:
        return None


def measure(mod, d: dict, hidden: int, heads: int, layers: int) -> pd.DataFrame:
    import torch

    rows: List[dict] = []
    dev = torch.device("cpu")
    B = 8

    # ---------- tabular baselines ----------
    tabular = [
        ("PCAP MLP", mod.PCAP_MLP(in_dim=d["d_pcap"]), d["d_pcap"]),
        ("SAR Encoder", mod.SAR_Autoencoder(in_dim=d["d_sar"]), d["d_sar"]),
        ("STRACE CNN-LSTM", mod.STRACE_CNN_LSTM(in_dim=d["d_strace"]), d["d_strace"]),
    ]
    for name, m, dim in tabular:
        m.eval().to(dev)
        x = torch.randn(B, dim)
        with torch.no_grad():
            m(x)
        r = {"Model": name, **count(m)}
        r["FLOPs (fwd, 1 sample)"] = try_flops(m, (torch.randn(1, dim),))
        rows.append(r)

    # ---------- late fusion ----------
    fus = mod.MultimodalLateFusionModel(pcap_dim=d["d_pcap"], sar_dim=d["d_sar"],
                                        strace_dim=d["d_strace"]).eval().to(dev)
    with torch.no_grad():
        fus(torch.randn(B, d["d_pcap"]), torch.randn(B, d["d_sar"]),
            torch.randn(B, d["d_strace"]))
    rows.append({"Model": "Multimodal Late Fusion", **count(fus),
                 "FLOPs (fwd, 1 sample)": None})

    # ---------- homogeneous GNN ----------
    sage = mod.Multimodal_NonHIN_GNN(in_dim=d["d_all"], hidden_dim=hidden,
                                     out_dim=2).eval().to(dev)
    ei = torch.stack([torch.arange(B), torch.roll(torch.arange(B), 1)], dim=0)
    with torch.no_grad():
        sage(torch.randn(B, d["d_all"]), ei)
    rows.append({"Model": "Res-GraphSAGE (non-HIN)", **count(sage),
                 "FLOPs (fwd, 1 sample)": None})

    # ---------- HIN models ----------
    g = build_dummy_graph(mod, n_files=B, d_file=d["d_pcap"] + d["d_sar"],
                          n_syscall=d["n_syscall"], n_proto=d["n_proto"],
                          n_arch=d["n_arch"])
    meta = g.metadata()

    hin_specs = [
        ("HIN-RGCN", mod.HIN_Vanilla_RGCN, dict(hidden=hidden, out=2, metadata=meta)),
        ("HIN-R2-RGCN", mod.HIN_R2_RGCN, dict(hidden=hidden, out=2, metadata=meta)),
        ("HIN-MAGNN", mod.HIN_MAGNN, dict(hidden=hidden, out=2, metadata=meta)),
        ("HIN-HAN", mod.HIN_HAN, dict(hidden=hidden, out=2, metadata=meta, heads=heads)),
        ("HIN-GAT", mod.HIN_GAT, dict(hidden=hidden, out=2, metadata=meta, heads=heads)),
        ("HIN-HGT", mod.HIN_Vanilla_HGT,
         dict(hidden=hidden, out=2, metadata=meta, heads=heads, layers=layers)),
        ("R-HGT S1 (vanilla HGT)", mod.HIN_HGT_Stage1_Vanilla,
         dict(hidden=hidden, out=2, metadata=meta, heads=heads, layers=layers)),
        ("R-HGT S2 (+res+LN)", mod.HIN_HGT_Stage2_ResLN,
         dict(hidden=hidden, out=2, metadata=meta, heads=heads, layers=layers)),
        ("R-HGT S3 (+node rel.)", mod.HIN_HGT_Stage3_NodeReliability,
         dict(hidden=hidden, out=2, metadata=meta, heads=heads, layers=layers)),
        ("R-HGT S4 (+rel. gating)", mod.HIN_HGT_Stage4_RelationGating,
         dict(hidden=hidden, out=2, metadata=meta, heads=heads, layers=layers)),
        ("Proposed R-HGT (= S4 arch.)", mod.HIN_Full_R_HGT,
         dict(hidden=hidden, out=2, metadata=meta, heads=heads, layers=layers)),
    ]
    for name, cls, kw in hin_specs:
        try:
            m = cls(**kw).eval().to(dev)
            with torch.no_grad():
                m(g.x_dict, g.edge_index_dict)   # materialize Linear(-1, ...)
            rows.append({"Model": name, **count(m), "FLOPs (fwd, 1 sample)": None})
        except Exception as e:                    # noqa: BLE001
            rows.append({"Model": name, "Parameters": None, "Trainable": None,
                         "Buffers": None, "Size (MB, fp32)": None,
                         "FLOPs (fwd, 1 sample)": None, "Error": repr(e)[:160]})

    return pd.DataFrame(rows)


# =========================================================================
# MAIN
# =========================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="Measure parameters / size / FLOPs.")
    ap.add_argument("--code", default=None,
                    help="Pipeline file. If omitted it is auto-detected: code-rhgt-v2.py "
                         "first, then the original unpatched version, in cwd / next to the script / "
                         "/kaggle/input.")
    ap.add_argument("--data-dir", default=None,
                    help="RAW parquet directory (<dir>/{arm,mips,...}/{pcap,sar,"
                         "strace}.parquet). Dimensions are inferred with DESCRIBE; no "
                         "data is loaded. Use this when there is no merged parquet.")
    ap.add_argument("--parquet", default=None,
                    help="cached_merged_multimodal.parquet — used to infer feature dimensions.")
    ap.add_argument("--pcap-dim", type=int, default=40)
    ap.add_argument("--sar-dim", type=int, default=300)
    ap.add_argument("--strace-dim", type=int, default=160)
    ap.add_argument("--n-syscall", type=int, default=None)
    ap.add_argument("--n-proto", type=int, default=8)
    ap.add_argument("--n-arch", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--out", default="params_measured")
    ap.add_argument("--dry-run", action="store_true",
                    help="Only print the analytic formulas; do not import torch.")
    args = ap.parse_args()

    d = None
    if args.parquet and os.path.exists(args.parquet):
        d = dims_from_parquet(args.parquet)
    elif args.data_dir:
        d = dims_from_data_dir(args.data_dir)
    elif args.parquet:
        print(f"[!] Not found: {args.parquet}")

    if d is None:
        d = dict(d_pcap=args.pcap_dim, d_sar=args.sar_dim, d_strace=args.strace_dim,
                 n_syscall=args.n_syscall or max(args.strace_dim, 1),
                 n_proto=args.n_proto, n_arch=args.n_arch,
                 d_all=args.pcap_dim + args.sar_dim + args.strace_dim)
        print("Feature dimensions (ASSUMED):", d)
        print("\n" + "!" * 70)
        print("❌ These dimensions are ASSUMED, not measured. Every parameter count")
        print("   produced from them is wrong and must not go into the paper.")
        print("   Pass --data-dir <raw parquet directory> or --parquet <merged")
        print("   file>. Add --dry-run if you only want to see the formulas.")
        print("!" * 70)
        if not args.dry_run:
            sys.exit(2)

    ana = pd.DataFrame(analytic_counts(d["d_pcap"], d["d_sar"],
                                       d["d_strace"], d["d_all"]))
    print("\n=== Analytic formulas (cross-check) ===")
    print(ana.to_string(index=False))

    # Invariant: Late Fusion CONTAINS all three branches, so it must exceed their sum.
    a = dict(zip(ana["Model"], ana["Analytic"]))
    branches = a["PCAP MLP"] + a["SAR Encoder"] + a["STRACE CNN-LSTM"]
    print(f"\nInvariant check: params(Late Fusion) must be > sum of the three branches")
    print(f"    sum of branches = {branches:,}")
    print(f"    Late Fusion   = {a['Multimodal Late Fusion']:,}  "
          f"{'✓ valid' if a['Multimodal Late Fusion'] > branches else '✗ INVALID'}")
    print("    (The table in the current draft lists Late Fusion = 89,200 < 233,800 "
          "= sum of the three branches → impossible.)")

    if args.dry_run:
        print("\n--dry-run: stopping here, torch is not imported.")
        return

    code = find_pipeline(args.code)
    if code is None:
        print("\n" + "!" * 70)
        print("❌ Pipeline file not found. The formula table above contains ONLY")
        print("   the non-graph branches — NO HIN model at all,")
        print("   i.e. exactly the part the paper needs is missing. Pass --code <path>.")
        print(f"   Names tried: {list(PIPELINE_NAMES)}")
        print("!" * 70)
        sys.exit(2)

    print(f"\nLoading pipeline from {code} ...")
    mod = load_pipeline(code)

    print("Instantiating and measuring each model ...")
    df = measure(mod, d, args.hidden, args.heads, args.layers)

    df = df.merge(ana[["Model", "Analytic", "Formula"]], on="Model", how="left")
    df["Formula match"] = [
        ("—" if pd.isna(an) or pd.isna(pa) else ("✓" if int(an) == int(pa) else "✗"))
        for an, pa in zip(df["Analytic"], df["Parameters"])]

    cols = ["Model", "Parameters", "Trainable", "Buffers", "Size (MB, fp32)",
            "FLOPs (fwd, 1 sample)", "Analytic", "Formula match", "Formula"]
    df = df[[c for c in cols if c in df.columns]]

    df.to_csv(f"{args.out}.csv", index=False)
    with open(f"{args.out}.md", "w", encoding="utf-8") as f:
        f.write("### MEASURED parameter counts\n\n")
        f.write(df.fillna("—").astype(str).to_string(index=False))
        f.write("\n")
    print("\n=== RESULTS ===")
    print(df.fillna("—").to_string(index=False))
    print(f"\nWrote {args.out}.csv and {args.out}.md")
    print("\nPaste the 'Parameters' and 'Size (MB, fp32)' columns into the computational "
          "performance table, REPLACING the old estimates.")


if __name__ == "__main__":
    main()
