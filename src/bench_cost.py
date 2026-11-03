#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bench_cost.py — RE-MEASURES peak memory and inference latency correctly, on ONE GPU.

Why this is needed. The 'Peak GPU Memory' column in the pipeline uses
torch.cuda.max_memory_allocated() but does NOT call reset_peak_memory_stats()
between models, so it measures the peak of the WHOLE PROCESS since start-up.
Consequence measured in results-l3: 16 configuration groups give identical
numbers, and R-HGT Stage 5 / Stage 6 / Stage 6a / Strict Control are all equal to
5,512.617 MB even though Stage 6 includes an extra fine-tuning round. That column
cannot be used in the paper.

Inference latency also needs re-measuring: the pipeline measures exactly once on
the test set, with no warm-up loop, so the first model to run bears the full cost
of CUDA context initialisation and kernel compilation.

This script measures, for each model independently:
  * peak memory for inference and for training, with a reset before each;
  * inference latency under BOTH protocols, with warm-up and many repetitions,
    reporting the median rather than the mean so that one outlier run does not
    skew the result.

Run on Kaggle, one seed, about 20-30 minutes:

    python3 bench_cost.py --data-dir <data directory> --out bench_cost

Parameter counts are already covered by measure_params.py; this script does NOT
repeat that.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import sys
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def _mp():
    spec = importlib.util.spec_from_file_location(
        "measure_params_mod", os.path.join(HERE, "measure_params.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def reset(torch):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def peak_mb(torch):
    return (torch.cuda.max_memory_allocated() / 1024 ** 2
            if torch.cuda.is_available() else float('nan'))


def time_forward(torch, fn, warmup: int, reps: int):
    """Median time of one forward pass, after warm-up."""
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts), min(ts), max(ts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--parquet", default=None)
    ap.add_argument("--code", default=None)
    ap.add_argument("--n-test", type=int, default=3404,
                    help="Number of simulated test samples (default equals the real test set).")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--out", default="bench_cost")
    args = ap.parse_args()

    mp = _mp()
    d = None
    if args.parquet and os.path.exists(args.parquet):
        d = mp.dims_from_parquet(args.parquet)
    elif args.data_dir:
        d = mp.dims_from_data_dir(args.data_dir)
    if d is None:
        sys.exit("❌ --data-dir or --parquet is required to get the REAL feature dimensions.")

    code = mp.find_pipeline(args.code)
    if code is None:
        sys.exit("❌ Pipeline file not found. Pass --code.")
    print(f"\nLoading pipeline from {code} ...")
    M = mp.load_pipeline(code)

    import torch
    if not torch.cuda.is_available():
        print("[!] NO GPU — memory will be NaN and latency is not comparable "
              "with the numbers in the paper. Use only to test the pipeline.")
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {gpu}")

    B = int(args.n_test)
    g = mp.build_dummy_graph(M, n_files=B, d_file=d["d_pcap"] + d["d_sar"],
                             n_syscall=d["n_syscall"], n_proto=d["n_proto"],
                             n_arch=d["n_arch"]).to(dev)
    g1 = mp.build_dummy_graph(M, n_files=1, d_file=d["d_pcap"] + d["d_sar"],
                              n_syscall=d["n_syscall"], n_proto=d["n_proto"],
                              n_arch=d["n_arch"]).to(dev)

    # Constructor signatures are taken VERBATIM from measure_params.py, not guessed again:
    # metadata comes LAST in the parameter list, and heads/layers are positional
    # parameters in the class definition. Pass everything by KEYWORD to avoid
    # depending on order.
    meta = g.metadata()
    H, HD, LY = args.hidden, args.heads, args.layers
    SPECS = [
        ("HIN-RGCN", M.HIN_Vanilla_RGCN, dict(hidden=H, out=2, metadata=meta)),
        ("HIN-R2-RGCN", M.HIN_R2_RGCN, dict(hidden=H, out=2, metadata=meta)),
        ("HIN-MAGNN", M.HIN_MAGNN, dict(hidden=H, out=2, metadata=meta)),
        ("HIN-HAN", M.HIN_HAN, dict(hidden=H, out=2, metadata=meta, heads=HD)),
        ("HIN-GAT", M.HIN_GAT, dict(hidden=H, out=2, metadata=meta, heads=HD)),
        ("HIN-HGT", M.HIN_Vanilla_HGT,
         dict(hidden=H, out=2, metadata=meta, heads=HD, layers=LY)),
        ("R-HGT S2 (+res+LN)", M.HIN_HGT_Stage2_ResLN,
         dict(hidden=H, out=2, metadata=meta, heads=HD, layers=LY)),
        ("R-HGT S3 (+node rel.)", M.HIN_HGT_Stage3_NodeReliability,
         dict(hidden=H, out=2, metadata=meta, heads=HD, layers=LY)),
        ("R-HGT S4 / Proposed", M.HIN_Full_R_HGT,
         dict(hidden=H, out=2, metadata=meta, heads=HD, layers=LY)),
    ]

    rows = []
    for name, cls, kw in SPECS:
        try:
            model = cls(**kw).to(dev)
        except Exception as e:
            print(f"  [!] {name}: could not instantiate ({type(e).__name__}: {e})")
            continue

        # Layers using Linear(-1, hidden) initialise LAZILY: before the first forward
        # pass they have no parameters, so counting right now would give 0. Run one
        # forward pass to materialise the parameters, then count.
        model.eval()
        with torch.no_grad():
            model(g.x_dict, g.edge_index_dict)
        nparam = sum(p.numel() for p in model.parameters())

        # ---- Protocol S: the whole test set in ONE entity-sharing graph ----
        reset(torch)
        with torch.no_grad():
            med_s, lo_s, hi_s = time_forward(
                torch, lambda: model(g.x_dict, g.edge_index_dict),
                args.warmup, args.reps)
        mem_infer_s = peak_mb(torch)

        # ---- Protocol I: one ego-graph at a time ----
        reset(torch)
        with torch.no_grad():
            med_i, lo_i, hi_i = time_forward(
                torch, lambda: model(g1.x_dict, g1.edge_index_dict),
                args.warmup, args.reps)
        mem_infer_i = peak_mb(torch)

        # ---- Training memory: one forward + backward step ----
        model.train()
        reset(torch)
        try:
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
            out = model(g.x_dict, g.edge_index_dict)
            logits = out['file'] if isinstance(out, dict) else out
            loss = torch.nn.functional.cross_entropy(
                logits, torch.zeros(logits.size(0), dtype=torch.long, device=dev))
            loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            mem_train = peak_mb(torch)
        except Exception as e:
            print(f"  [!] {name}: could not measure training memory ({e})")
            mem_train = float('nan')

        rows.append({
            "Model": name, "Parameters": nparam,
            "Mem infer S (MB)": mem_infer_s,
            "Mem infer I (MB)": mem_infer_i,
            "Mem train (MB)": mem_train,
            "Latency S (ms/sample)": med_s / B * 1000.0,
            "Latency I (ms/sample)": med_i * 1000.0,
            "I/S slowdown": (med_i * B) / max(med_s, 1e-12),
        })
        print(f"  {name:24s} params {nparam:>9,d} | inference memory S "
              f"{mem_infer_s:8.1f} MB | train memory {mem_train:8.1f} MB | "
              f"S {med_s / B * 1000:.4f} ms/sample | I {med_i * 1000:.4f} ms/sample")
        del model
        reset(torch)

    if not rows:
        sys.exit("❌ No model could be measured.")
    df = pd.DataFrame(rows)
    df.to_csv(args.out + ".csv", index=False)
    with open(args.out + ".md", "w", encoding="utf-8", newline="\n") as f:
        f.write(f"### Re-measured computational cost — {gpu}\n\n")
        f.write(f"n_test = {B}, warmup = {args.warmup}, reps = {args.reps}, "
                f"median over repetitions\n\n")
        f.write(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        f.write("\n")
    with open(args.out + "_env.json", "w", encoding="utf-8") as f:
        json.dump({"gpu": gpu, "torch": torch.__version__,
                   "n_test": B, "warmup": args.warmup, "reps": args.reps,
                   "dims": d}, f, indent=2)

    print(f"\n{'=' * 96}")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\n✅ {args.out}.csv / .md / _env.json")
    print("\nNOTE when putting this in the paper: every number here comes from ONE GPU in ONE")
    print("process, measured on a simulated graph of exactly the real test-set size.")
    print("Do not mix with the time/memory columns in the old raw files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
