#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_slim_ego.py — Verifies that the ego-graph optimisation does NOT change the results.

Background
----------
The original version loads the ENTIRE anchor set (syscall + arch + protocol) into
EACH ego-graph, so a batch of 256 samples has 256·(K_sys + 4 + K_proto) anchor
nodes instead of K_sys + 4 + K_proto. Version v3 keeps only the anchor ROWS that
are actually connected, but keeps the one-hot feature DIMENSION unchanged so that
it matches the Linear(-1, hidden) layer already materialised from the train graph.

Equivalence argument: a 'file' node only receives messages from the anchors
connected to it; unconnected anchors form isolated components and so contribute
nothing to its representation. This script VERIFIES that argument numerically,
on every HIN architecture.

Usage
-----
    python3 verify_slim_ego.py --code code-rhgt-v2.py

    # use real data instead of synthetic data:
    python3 verify_slim_ego.py --code code-rhgt-v2.py \\
                               --parquet experiments/cached_merged_multimodal.parquet

Pass criterion: maximum absolute difference of the 'file' node logits < 1e-5.

Requirements: torch, torch_geometric, pandas, numpy (same as the pipeline).
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from typing import List, Tuple

import numpy as np
import pandas as pd

TOL = 1e-5


def load_pipeline(path: str):
    spec = importlib.util.spec_from_file_location("hin_pipeline", path)
    if spec is None or spec.loader is None:
        sys.exit(f"Could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["hin_pipeline"] = mod
    spec.loader.exec_module(mod)          # main() is guarded by __name__
    return mod


def synthetic_frame(n: int = 240, d_pcap: int = 12, d_sar: int = 40,
                    n_sys: int = 180, n_proto: int = 8, seed: int = 0) -> pd.DataFrame:
    """Synthetic dataframe following the pipeline's exact column-naming conventions."""
    rng = np.random.RandomState(seed)
    cols = {}
    # PCAP: match the suffix that the pipeline's heuristic catches
    for i in range(d_pcap):
        cols[f"f{i}_count"] = rng.rand(n).astype(np.float32) * 10
    # SAR: all remaining columns
    for i in range(d_sar):
        cols[f"sar_metric_{i}"] = rng.randn(n).astype(np.float32)
    # STRACE: sparse, like real syscall frequencies
    for i in range(n_sys):
        v = rng.rand(n)
        cols[f"Call_{i}"] = np.where(v < 0.06, rng.rand(n) * 5, 0.0).astype(np.float32)
    # Protocol: sparse binary
    for p in ["TCP", "UDP", "ICMP", "HTTP", "HTTPS", "DNS", "SSH", "ARP"][:n_proto]:
        cols[p] = (rng.rand(n) < 0.25).astype(np.float32)

    df = pd.DataFrame(cols)
    df["arch"] = rng.choice(["arm", "mips", "mipsel", "x86"], size=n)
    df["label"] = (rng.rand(n) < 0.45).astype(int)
    df["Hash"] = [f"h{i:06d}" for i in range(n)]
    df["MalwareFamily"] = np.where(df["label"] == 1, "Mirai", "benign")
    return df


PCAP_NAMES = ['Header_Length', 'Protocol Type', 'Time_To_Live', 'Rate', 'Tot sum',
              'Min', 'Max', 'AVG', 'Std', 'Tot size', 'IAT', 'Number', 'Variance']
PROTO_NAMES = ['TCP', 'UDP', 'ICMP', 'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SSH',
               'IRC', 'DHCP', 'ARP', 'IGMP', 'IPv', 'LLC']


def feature_groups(df: pd.DataFrame):
    cols = df.columns.tolist()
    syscall = [c for c in cols if c.startswith('Call_') or 'syscall' in c.lower()]
    proto = [c for c in PROTO_NAMES if c in cols]
    exclude = set(['label', 'arch', 'Arch', 'Hash', 'MalwareFamily',
                   'stratify_col'] + syscall + proto)
    pcap = [c for c in cols if c in PCAP_NAMES
            or c.endswith('_flag_number') or c.endswith('_count')]
    sar = [c for c in cols if c not in exclude and c not in pcap]
    return pcap + sar, syscall, proto


def build_egos(mod, slim: bool, df_te, num_cols, sys_cols, proto_list,
               scaler, arch_le, anchors):
    os.environ['RHGT_SLIM_EGO'] = '1' if slim else '0'
    t0 = time.perf_counter()
    egos = mod.RectifiedStrictInductiveEngine.build_isolated_ego_graphs_batch(
        df_te, num_cols, sys_cols, proto_list, scaler, arch_le, anchors)
    return egos, time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify equivalence of the slim ego-graph.")
    ap.add_argument("--code", default="code-rhgt-v2.py")
    ap.add_argument("--parquet", default="")
    ap.add_argument("--n-test", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    if not os.path.exists(args.code):
        sys.exit(f"Not found: {args.code}")
    print(f"Loading pipeline: {args.code}")
    mod = load_pipeline(args.code)

    import torch
    from torch_geometric.data import Batch

    torch.manual_seed(0)
    np.random.seed(0)

    # ---------- data ----------
    if args.parquet and os.path.exists(args.parquet):
        df = pd.read_parquet(args.parquet)
        if 'label' not in df.columns:
            df['label'] = (df['MalwareFamily'].astype(str).str.lower() != 'benign').astype(int)
        df = df.sample(n=min(len(df), 4 * args.n_test), random_state=0).reset_index(drop=True)
        print(f"Using real data: {df.shape}")
    else:
        df = synthetic_frame(n=4 * args.n_test)
        print(f"Using synthetic data: {df.shape}")

    num_cols, sys_cols, proto_list = feature_groups(df)
    print(f"  num_cols={len(num_cols)}  syscall={len(sys_cols)}  proto={len(proto_list)}")

    df_tr = df.iloc[args.n_test:].reset_index(drop=True)
    df_te = df.iloc[:args.n_test].reset_index(drop=True)

    # ---------- train graph + anchors ----------
    g_tr, scaler, arch_le = mod.build_hetero_graph_hin(
        df_tr, num_cols, sys_cols, proto_list, mode='train', k_sim=3)
    metadata = g_tr.metadata()
    anchors = mod.RectifiedStrictInductiveEngine.extract_canonical_anchors(g_tr)
    print(f"  neo: " + ", ".join(f"{k}={v.shape[0]}" for k, v in anchors.items()))

    # ---------- build ego-graphs both ways ----------
    egos_full, t_full = build_egos(mod, False, df_te, num_cols, sys_cols,
                                   proto_list, scaler, arch_le, anchors)
    egos_slim, t_slim = build_egos(mod, True, df_te, num_cols, sys_cols,
                                   proto_list, scaler, arch_le, anchors)

    def count_nodes(egos) -> int:
        b = Batch.from_data_list(egos[:args.batch_size])
        return int(sum(b[nt].x.size(0) for nt in b.node_types))

    n_full, n_slim = count_nodes(egos_full), count_nodes(egos_slim)
    print("\n" + "=" * 70)
    print("SIZE COMPARISON")
    print("=" * 70)
    print(f"  Number of nodes in a batch of {args.batch_size} ego-graph:")
    print(f"      full   : {n_full:,}")
    print(f"      slim   : {n_slim:,}   → reduced by {n_full / max(n_slim, 1):.1f}×")
    print(f"  Time to build {len(df_te)} ego-graph:")
    print(f"      full   : {t_full:.2f} s")
    print(f"      slim   : {t_slim:.2f} s")

    # ---------- verification on every architecture ----------
    specs = [
        ("HIN RGCN",    mod.HIN_Vanilla_RGCN, dict(hidden=64, out=2, metadata=metadata)),
        ("HIN R2-RGCN", mod.HIN_R2_RGCN,      dict(hidden=64, out=2, metadata=metadata)),
        ("HIN MAGNN",   mod.HIN_MAGNN,        dict(hidden=64, out=2, metadata=metadata)),
        ("HIN HAN",     mod.HIN_HAN,          dict(hidden=64, out=2, metadata=metadata, heads=4)),
        ("HIN GAT",     mod.HIN_GAT,          dict(hidden=64, out=2, metadata=metadata, heads=4)),
        ("HIN HGT",     mod.HIN_Vanilla_HGT,  dict(hidden=64, out=2, metadata=metadata, heads=4, layers=2)),
        ("R-HGT (S4/5/6)", mod.HIN_Full_R_HGT, dict(hidden=64, out=2, metadata=metadata, heads=4, layers=2)),
    ]

    print("\n" + "=" * 70)
    print("EQUIVALENCE CHECK  (pass threshold: difference < 1e-5)")
    print("=" * 70)

    dev = mod.device
    results: List[Tuple[str, float, bool]] = []
    for name, cls, kw in specs:
        try:
            torch.manual_seed(1234)
            m = cls(**kw).to(dev)
            # materialise Linear(-1, hidden), then lock in eval mode
            with torch.no_grad():
                m(g_tr.to(dev).x_dict, g_tr.to(dev).edge_index_dict)
            m.eval()

            outs = []
            for egos in (egos_full, egos_slim):
                chunks = []
                with torch.no_grad():
                    for i in range(0, len(egos), args.batch_size):
                        b = Batch.from_data_list(egos[i:i + args.batch_size]).to(dev)
                        chunks.append(m(b.x_dict, b.edge_index_dict).cpu())
                outs.append(torch.cat(chunks, dim=0))

            diff = float((outs[0] - outs[1]).abs().max())
            ok = diff < TOL
            results.append((name, diff, ok))
            print(f"  {'✅' if ok else '❌'} {name:16s} max difference = {diff:.3e}")
        except Exception as e:                                   # noqa: BLE001
            results.append((name, float('nan'), False))
            print(f"  ⚠️  {name:16s} ERROR: {repr(e)[:110]}")

    # ---------- batch-isolation check ----------
    print("\n" + "=" * 70)
    print("ISOLATION CHECK  (changing batch_size must NOT change the results)")
    print("=" * 70)
    for name, cls, kw in specs:
        try:
            torch.manual_seed(1234)
            m = cls(**kw).to(dev)
            with torch.no_grad():
                m(g_tr.to(dev).x_dict, g_tr.to(dev).edge_index_dict)
            m.eval()
            vals = []
            for bs in (16, 128):
                chunks = []
                with torch.no_grad():
                    for i in range(0, len(egos_slim), bs):
                        b = Batch.from_data_list(egos_slim[i:i + bs]).to(dev)
                        chunks.append(m(b.x_dict, b.edge_index_dict).cpu())
                vals.append(torch.cat(chunks, dim=0))
            d = float((vals[0] - vals[1]).abs().max())
            print(f"  {'✅' if d < TOL else '❌'} {name:16s} bs=16 vs bs=128 : {d:.3e}")
            if d >= TOL:
                print(f"       ↳ this model DEPENDS ON THE BATCH — it is not truly "
                      f"strict-inductive; fix it or declare it in the paper.")
        except Exception as e:                                   # noqa: BLE001
            print(f"  ⚠️  {name:16s} ERROR: {repr(e)[:110]}")

    n_ok = sum(1 for _, _, ok in results if ok)
    print("\n" + "=" * 70)
    if n_ok == len(results):
        print(f"✅ PASS — all {len(results)} architectures give matching results.")
        print(f"   The ego-graph optimisation is safe to use; {n_full/max(n_slim,1):.1f}× fewer nodes.")
        sys.exit(0)
    else:
        print(f"❌ FAIL — {len(results)-n_ok}/{len(results)} architectures differ.")
        print("   Do NOT use RHGT_SLIM_EGO=1 until the cause is found;")
        print("   for now set RHGT_SLIM_EGO=0 to return to the old behaviour.")
        sys.exit(1)


if __name__ == "__main__":
    main()
