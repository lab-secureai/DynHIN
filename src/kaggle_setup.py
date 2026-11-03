#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kaggle_setup.py — Setup cell for a Kaggle notebook: checks the data, sets the
                  environment variables, and restores results from the previous session.

Paste into the FIRST cell of the notebook:

    !cp /kaggle/input/<dataset-code>/kaggle_setup.py .
    %run kaggle_setup.py

Or call it with arguments:

    %run kaggle_setup.py --data-dir /kaggle/working/dong --phase 1

The script prints diagnostics and then sets os.environ for `code-rhgt-v2.py`.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

WORKING = "/kaggle/working"
DEFAULT_DATA = "auto"   # "auto" = let rhgt_paths.py detect it
DEFAULT_SAVE = "/kaggle/working/experiments"

ARCHS = ["arm", "mips", "mipsel", "x86"]
MODALITIES = ["pcap", "sar", "strace"]

# Per-session configuration; see README.md.
# The six configurations that appear in the robustness-curve figure. The names must
# match hin_model_specs in code-rhgt-v2.py EXACTLY: a single wrong character
# yields an empty set, and the run takes 11 hours without attacking any configuration.
ATTACK_SET = ",".join([
    'R-HGT Stage 6 (+ Strict-Inductive Fine-Tuning)',
    'R-HGT Strict Control (No Fine-Tuning)',
    'Multimodal + HIN R2-RGCN',
    'Multimodal + HIN HAN',
    'Multimodal + HIN RGCN',
    'Multimodal + HIN HGT',
])

PHASES = {
    "1": {  # ~5 h: clean + groups A/C, all 10 seeds
        "RHGT_EXTRA": "AC", "RHGT_SCENARIOS": "clean",
        "RHGT_SKIP_LOFO": "1", "RHGT_SAVE_PROBS": "1",
    },
    "2": {  # ~5 h: LOFO, 2 families with enough samples, first 5 seeds
        "RHGT_EXTRA": "AC", "RHGT_SKIP_MAIN": "1",
        "RHGT_LOFO_FAMILIES": "Mirai,DarkNexus",
        "RHGT_SEEDS": "10,42,100,314,555",
    },
    "3": {  # ~5 h: LOFO, remaining 5 seeds
        "RHGT_EXTRA": "AC", "RHGT_SKIP_MAIN": "1",
        "RHGT_LOFO_FAMILIES": "Mirai,DarkNexus",
        "RHGT_SEEDS": "777,999,1234,2024,2026",
    },
    "23": {  # merges sessions 2+3 into ONE 12-hour session, with a time budget
        "RHGT_EXTRA": "AC", "RHGT_SKIP_MAIN": "1",
        "RHGT_LOFO_FAMILIES": "Mirai,DarkNexus",
        # all 10 seeds; if it does not finish, the budget stops cleanly and the next session resumes
        "RHGT_TIME_BUDGET_S": "39600",   # 11 h, leaving a 1 h safety margin
    },
    "4": {  # adversarial attacks + noise; 11 h budget, stops cleanly on its own
        "RHGT_EXTRA": "AC", "RHGT_SCENARIOS": "attack",
        "RHGT_SKIP_LOFO": "1", "RHGT_ATTACK_RESTARTS": "3",
        "RHGT_SEEDS": "10,42,100,314,555",
        "RHGT_TIME_BUDGET_S": "39600",
    },
    "5": {  # ~7 h: group B (baselines with the same procedure)
        "RHGT_EXTRA": "ABC", "RHGT_SCENARIOS": "clean",
        "RHGT_SKIP_LOFO": "1",
    },
    # ==================================================================
    # FULL PROFILES: split by SEED, not by scenario.
    # ==================================================================
    # Splitting by scenario (m1/m2/m3 below) has a defect that merging cannot
    # fix: two machines both produce Clean rows for the configurations they
    # share, and because scatter-add aggregation on GPU is non-deterministic
    # (and the two accounts may also be assigned different GPU types), those
    # two rows are NEVER equal. At merge time one must be chosen, and if the
    # wrong one is chosen, the epsilon=0 point of the robustness curve comes
    # from a different training run than the attacked points.
    #
    # Splitting by SEED removes that defect:
    #   - Each key (configuration, seed, scenario) is computed EXACTLY ONCE in
    #     exactly one process. Merging becomes a concatenation, with no
    #     conflicts and no decision about which source to keep.
    #   - Every test in the paper is a PAIRED test WITHIN THE SAME SEED
    #     (protocol gap, 2x2, R-HGT vs. baselines). Both sides of each pair
    #     always come from the same process, GPU and RNG stream. Hardware
    #     differences between machines become noise BETWEEN seeds, which the
    #     paired design already cancels.
    #   - Clean and attacked evaluations share one checkpoint, so the
    #     robustness curve is internally consistent.
    #   - The base is not retrained on a second machine: saves ~2.4 h.
    #
    # Hence "one machine" and "three machines" yield the SAME merged table;
    # the number of machines only affects wall-clock time, not validity.
    #
    # Measured cost (T4, groups ABC), per seed:
    #     main loop + Clean                3,117 s   (m1: 31,166.8 s / 10 seeds)
    #   + noise and attacks, 6 configs     1,217 s   (derived from m3, training subtracted)
    #   + LOFO, two families               5,458 s   (m2: 38,203.3 s / 7 seeds)
    #   = 9,792 s = 2.72 h/seed  =>  10 seeds = 27.2 h
    #
    # Single machine: run "full" in three consecutive sessions; RHGT_RESUME
    # handles the continuation.
    # Several machines: run k1/k2/k3 in parallel, each on a disjoint seed subset.
    # ------------------------------------------------------------------
    # GENERATING HASHED PROB DUMPS for the deduplication experiment
    # ------------------------------------------------------------------
    # A probe found that 7.05% of test samples have cosine > 0.999 to some
    # training sample. To answer "does cross-contamination explain the
    # protocol gap?" it is enough to recompute the metrics on the filtered
    # test set: an array computation, not a training run. However, the old
    # prob dumps did NOT store the Hash, and the test-set ordering cannot be
    # reproduced outside the original run: the split uses
    # StratifiedGroupKFold(shuffle=True), so it depends on the DataFrame row
    # order, which DuckDB decides during GROUP BY. Reconstructing with DuckDB
    # 1.5.5 differs by 1,424/3,404 labels from the Kaggle run (DuckDB 1.3.2),
    # even though the test-set size and stratum distribution match exactly.
    #
    # Patch K1 stores the Hash in the dump. This phase regenerates hashed
    # dumps for exactly the protocol-comparison pairs (group A); 3 seeds are
    # enough to answer the question. dedup_reeval.py is then run on CPU only.
    "dedup": {
        "RHGT_EXTRA": "A", "RHGT_SCENARIOS": "clean",
        "RHGT_SKIP_LOFO": "1", "RHGT_SAVE_PROBS": "1",
        "RHGT_SEEDS": "10,42,100",
        "RHGT_TIME_BUDGET_S": "39600",
    },

    # ------------------------------------------------------------------
    # CLOSING THE ATTACK-SET GAP
    # ------------------------------------------------------------------
    # The earlier run attacked only 6 configurations, four of which are
    # baselines WITHOUT the full procedure. The robustness table therefore
    # compared R-HGT *with* the procedure against baselines *without* it,
    # which is exactly the confound that group B exists to remove.
    #
    # This phase attacks 4 configurations that pose the right question: the
    # proposed model, two baselines that DID go through the full procedure
    # (GAT and HAN, the two competitors on par with it in the clean and LOFO
    # tables), and the unaligned control.
    #
    # It CANNOT resume from the earlier run: the pipeline saves no checkpoints
    # (there is no torch.save call), so attacking new configurations requires
    # retraining from scratch. Cost: 3,117 s/seed training + ~811 s/seed
    # attacks = 3,928 s/seed => ~10.9 h for 10 seeds, not one hour.
    "atk": {
        "RHGT_EXTRA": "ABC", "RHGT_SCENARIOS": "all",
        "RHGT_SKIP_LOFO": "1", "RHGT_ATTACK_RESTARTS": "3",
        "RHGT_ATTACK_CONFIGS": ",".join([
            "R-HGT Stage 6 (+ Strict-Inductive Fine-Tuning)",
            "R-HGT Strict Control (No Fine-Tuning)",
            "HIN GAT (robust + strict FT)",
            "HIN HAN (robust + strict FT)",
        ]),
        "RHGT_TIME_BUDGET_S": "39600",
    },
    "full": {  # all scenarios, all seeds; stops on its own at the budget, then resume
        "RHGT_EXTRA": "ABC", "RHGT_SCENARIOS": "all",
        "RHGT_SAVE_PROBS": "1",
        "RHGT_LOFO_FAMILIES": "Mirai,DarkNexus",
        "RHGT_ATTACK_RESTARTS": "3",
        "RHGT_ATTACK_CONFIGS": ATTACK_SET,
        "RHGT_TIME_BUDGET_S": "39600",
    },
    "k1": {  # like "full" but only the first 4 seeds   (~10.9 h)
        "RHGT_EXTRA": "ABC", "RHGT_SCENARIOS": "all",
        "RHGT_SAVE_PROBS": "1",
        "RHGT_LOFO_FAMILIES": "Mirai,DarkNexus",
        "RHGT_ATTACK_RESTARTS": "3",
        "RHGT_ATTACK_CONFIGS": ATTACK_SET,
        "RHGT_SEEDS": "10,42,100,314",
        "RHGT_TIME_BUDGET_S": "39600",
    },
    "k2": {  # middle 3 seeds                          (~8.2 h)
        "RHGT_EXTRA": "ABC", "RHGT_SCENARIOS": "all",
        "RHGT_SAVE_PROBS": "1",
        "RHGT_LOFO_FAMILIES": "Mirai,DarkNexus",
        "RHGT_ATTACK_RESTARTS": "3",
        "RHGT_ATTACK_CONFIGS": ATTACK_SET,
        "RHGT_SEEDS": "555,777,999",
        "RHGT_TIME_BUDGET_S": "39600",
    },
    "k3": {  # last 3 seeds                            (~8.2 h)
        "RHGT_EXTRA": "ABC", "RHGT_SCENARIOS": "all",
        "RHGT_SAVE_PROBS": "1",
        "RHGT_LOFO_FAMILIES": "Mirai,DarkNexus",
        "RHGT_ATTACK_RESTARTS": "3",
        "RHGT_ATTACK_CONFIGS": ATTACK_SET,
        "RHGT_SEEDS": "1234,2024,2026",
        "RHGT_TIME_BUDGET_S": "39600",
    },

    # ------------------------------------------------------------------
    # THREE MACHINE PROFILES (OLD, split by scenario): run IN PARALLEL on three
    # machines, each at most 12 h.
    # All three MUST use the SAME code-rhgt-v2.py and the SAME dataset; only the
    # environment variables differ. Compare md5 checksums before running.
    # ------------------------------------------------------------------
    "m1": {  # MACHINE 1: scenarios 1 & 2: clean benchmark, groups A+B+C
        "RHGT_EXTRA": "ABC", "RHGT_SCENARIOS": "clean",
        "RHGT_SKIP_LOFO": "1", "RHGT_SAVE_PROBS": "1",
        "RHGT_TIME_BUDGET_S": "39600",
    },
    "m2": {  # MACHINE 2: scenario 3: LOFO zero-day, groups A+B+C
        "RHGT_EXTRA": "ABC", "RHGT_SKIP_MAIN": "1",
        "RHGT_LOFO_FAMILIES": "Mirai,DarkNexus",
        "RHGT_TIME_BUDGET_S": "39600",
    },
    # MACHINE 3 uses group "A", NOT "AC". Reason: seed_everything() is called
    # ONCE per seed, so the random-number stream runs sequentially through the
    # models in the order of hin_model_specs. Group B (12 models) comes BEFORE
    # group C, so the two group-C models trained in an ABC session (machine 1)
    # and in an AC session (machine 3) would get different RNG states and
    # produce different results for the SAME seed, giving two conflicting rows
    # at merge time. The 14 base models plus the 8 group-A models are
    # identical on both machines because they are trained before A/B/C. All 6
    # attacked configurations are in the base, so machine 3 does not need
    # group C; dropping it also saves one strict fine-tuning run
    # (~11 min/seed) that machine 1 already performs.
    "m3": {  # MACHINE 3: noise + adversarial attacks, restricted configuration set
        "RHGT_EXTRA": "A", "RHGT_SCENARIOS": "all",
        "RHGT_SKIP_LOFO": "1", "RHGT_ATTACK_RESTARTS": "3",
        "RHGT_ATTACK_CONFIGS": "R-HGT Stage 6 (+ Strict-Inductive Fine-Tuning),R-HGT Strict Control (No Fine-Tuning),Multimodal + HIN R2-RGCN,Multimodal + HIN HAN,Multimodal + HIN RGCN,Multimodal + HIN HGT",
        "RHGT_TIME_BUDGET_S": "39600",
    },
    "smoke": {  # a few minutes: sanity check before burning 9 hours
        "RHGT_QUICK_TEST": "1", "RHGT_EXTRA": "AC",
        "RHGT_SCENARIOS": "clean", "RHGT_LOFO_FAMILIES": "Tsunami",
    },
}


# [v3] Spell out what each phase DOES and DOES NOT do. Motivation: two
# consecutive sessions ran --phase 1 and were surprised to find no LOFO
# results, although phase 1 sets RHGT_SKIP_LOFO=1 by design.
_FULL_DOES = [
    "ALL scenarios in ONE process: Clean + noise + FGSM/BIM/PGD + LOFO",
    "36 configurations (groups A+B+C); 6 configurations get noise/attacks",
    "LOFO zero-day, two families: Mirai, DarkNexus",
    "Saves y_prob for threshold sweeps",
    "Writes results after EACH seed (main loop) and EACH seed-family pair (LOFO)",
]
_FULL_NOT = [
    "Does NOT split work by scenario, which causes conflicts at merge time",
    "Time and memory columns must still be measured separately on a SINGLE GPU",
]

PHASE_DOC = {
    "full": {
        "does": _FULL_DOES + ["Runs all 10 seeds; stops cleanly when time runs out, the next session resumes"],
        "does_not": _FULL_NOT + ["Does NOT finish in one session: 27.2 h needs 3 sessions"],
        "next": "full (repeat until all 10 seeds are done)",
        "hours": "2.72 h/seed measured => 27.2 h for 10 seeds => 3 sessions",
    },
    "dedup": {
        "does": ["Regenerates prob_dumps WITH THE HASH FIELD (patch K1)",
                 "22 group-A configurations, covering all 10 protocol-comparison pairs",
                 "ONLY 3 seeds: 10, 42, 100",
                 "Clean scenario only",
                 "AFTERWARD: run dedup_reeval.py on CPU, with --results pointing at this output"],
        "does_not": ["Does NOT run LOFO, noise or attacks",
                     "Does NOT replace the main results; this is an auxiliary set for "
                     "one experiment, not the main results table",
                     "Its rows are NOT comparable with the main results: the EXTRA "
                     "group differs, hence a different RNG stream"],
        "next": None,
        "hours": "887 s/seed × 3 seeds ≈ 45 min",
    },
    "atk": {
        "does": ["Closes the attack-set gap: 4 configurations with noise + FGSM/BIM/PGD",
                 "  • R-HGT Stage 6 (proposed)",
                 "  • HIN GAT (robust + strict FT)  — competitor on par",
                 "  • HIN HAN (robust + strict FT)  — competitor on par",
                 "  • R-HGT Strict Control          — unaligned control",
                 "Clean runs for all 36 configurations of groups ABC (its own epsilon=0 reference)"],
        "does_not": ["Does NOT run LOFO",
                     "Cannot resume from earlier results: the pipeline saves no "
                     "checkpoints, so it must retrain",
                     "Do NOT merge this phase's Clean rows with earlier results; "
                     "the robustness table uses THIS PHASE'S OWN Clean as its reference"],
        "next": None,
        "hours": "3,928 s/seed (3,117 training + 811 attacks) => ~10.9 h",
    },
    "k1": {
        "does": _FULL_DOES + ["ONLY seeds 10, 42, 100, 314, each run in full"],
        "does_not": _FULL_NOT + ["Does NOT touch the 6 seeds of k2 and k3"],
        "next": None,
        "hours": "~10.9 h (4 seeds × 2.72), close to the limit, a short resume may be needed",
    },
    "k2": {
        "does": _FULL_DOES + ["ONLY seeds 555, 777, 999, each run in full"],
        "does_not": _FULL_NOT + ["Does NOT touch the seeds of k1 and k3"],
        "next": None,
        "hours": "~8.2 h (3 seeds × 2.72)",
    },
    "k3": {
        "does": _FULL_DOES + ["ONLY seeds 1234, 2024, 2026, each run in full"],
        "does_not": _FULL_NOT + ["Does NOT touch the seeds of k1 and k2"],
        "next": None,
        "hours": "~8.2 h (3 seeds × 2.72)",
    },
    "1": {
        "does": ["Scenarios 1&2: clean benchmark, 10 seeds",
                 "24 configurations (extra groups A + C)",
                 "Saves y_prob for later threshold sweeps"],
        "does_not": ["Does NOT run LOFO / zero-day  (RHGT_SKIP_LOFO=1)",
                     "Does NOT run noise or attacks (RHGT_SCENARIOS=clean)"],
        "next": "2",
        "hours": "~4.3 h (measured on a T4)",
    },
    "2": {
        "does": ["Scenario 3: LOFO zero-day, families Mirai + DarkNexus",
                 "First 5 seeds: 10, 42, 100, 314, 555"],
        "does_not": ["Does NOT rerun scenarios 1&2 (RHGT_SKIP_MAIN=1)",
                     "Does NOT touch the main result files; they are protected"],
        "next": "3",
        "hours": "~4-5 h (estimated from session 1)",
    },
    "3": {
        "does": ["Scenario 3: LOFO zero-day, families Mirai + DarkNexus",
                 "Remaining 5 seeds: 777, 999, 1234, 2024, 2026"],
        "does_not": ["Does NOT rerun scenarios 1&2 (RHGT_SKIP_MAIN=1)"],
        "next": "4",
        "hours": "~4-5 h",
    },
    "23": {
        "does": ["Scenario 3: LOFO zero-day, families Mirai + DarkNexus",
                 "ALL 10 seeds in one session",
                 "11 h budget: stops cleanly before Kaggle's 12 h limit",
                 "Writes results after EACH (seed, family) pair, so completed work is never lost"],
        "does_not": ["Does NOT rerun scenarios 1&2 (RHGT_SKIP_MAIN=1)",
                     "Does NOT touch the main result files",
                     "If 10 seeds do not fit, the next session resumes; nothing is lost"],
        "next": "4",
        "hours": "estimated ~7.5 h; hard limit 11 h",
    },
    "4": {
        "does": ["Noise + FGSM/BIM/PGD adversarial attacks, 4 epsilon levels",
                 "Requires 5 seeds; PGD uses 3 restarts instead of 5 to save time",
                 "11 h budget: stops cleanly after the last seed that fits",
                 "Writes results after EACH seed, so completed work is never lost"],
        "does_not": ["Does NOT run LOFO",
                     "May NOT finish all 5 seeds; the next session resumes",
                     "Does NOT touch existing clean and LOFO results"],
        "next": "5",
        "hours": "not measured; hard budget 11 h",
    },
    "5": {
        "does": ["Group B: applies Stage 5 + Stage 6 to EVERY HIN baseline",
                 "36 configurations (A + B + C), clean, 10 seeds"],
        "does_not": ["Does NOT run LOFO", "Does NOT run attacks"],
        "next": None,
        "hours": "~7 h",
    },
    "m1": {
        "does": ["MACHINE 1: scenarios 1 & 2: clean benchmark",
                 "36 configurations: groups A + B + C (group B = HIN baselines with the full S5+S6 procedure)",
                 "Requires 10 seeds; saves y_prob for threshold sweeps",
                 "11 h budget, writes results after EACH seed"],
        "does_not": ["Does NOT run LOFO", "Does NOT run noise or attacks",
                     "May not finish 10 seeds; the next session resumes"],
        "next": None,
        "hours": "estimated ~1.4 h/seed with groups ABC; hard limit 11 h",
    },
    "m2": {
        "does": ["MACHINE 2: scenario 3: LOFO zero-day, families Mirai + DarkNexus",
                 "36 configurations: groups A + B + C",
                 "Requires 10 seeds (20 seed-family pairs)",
                 "11 h budget, writes results after EACH (seed, family) pair"],
        "does_not": ["Does NOT rerun scenarios 1 & 2 (RHGT_SKIP_MAIN=1)",
                     "Does NOT touch the main result files"],
        "next": None,
        "hours": "MEASURED with groups ABC: 7 seeds in one ~11 h session => ~1.5 h/seed",
    },
    "m3": {
        "does": ["MACHINE 3: noise (Gaussian, masking, edge drop) + FGSM/BIM/PGD attacks",
                 "Only 6 configurations get noise/attacks (RHGT_ATTACK_CONFIGS)",
                 "Clean STILL runs for all 22 group-A configurations",
                 "These 22 Clean rows MATCH machine 1 bit for bit, which allows cross-validation",
                 "PGD uses 3 restarts instead of 5"],
        "does_not": ["Does NOT run LOFO",
                     "Does NOT attack the remaining 16 configurations: too expensive, and they appear in no figure",
                     "Does NOT run groups B and C; machine 1 covers those"],
        "next": None,
        "hours": "estimated ~1.8 h/seed; hard limit 11 h",
    },
    "smoke": {
        "does": ["Runs a tiny job to test the pipeline before burning 9 hours"],
        "does_not": ["Results are NOT usable for the paper"],
        "next": "1",
        "hours": "a few minutes",
    },
}


def print_phase_doc(phase: str) -> None:
    """Print what this phase does / does not do, BEFORE it runs, not after."""
    d = PHASE_DOC.get(phase)
    if not d:
        return
    print(f"\n{'=' * 70}\nPHASE {phase} — {d['hours']}\n{'=' * 70}")
    print("WILL DO:")
    for x in d["does"]:
        print(f"  ✅ {x}")
    print("WILL NOT DO:")
    for x in d["does_not"]:
        print(f"  ⛔ {x}")
    if d["next"]:
        print(f"\nAfter this phase: save the output, attach it as an input, then run")
        print(f"     %run kaggle_setup.py --phase {d['next']}")


def human(n: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def dir_size(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def check_data(data_dir: str) -> str:
    """Returns 'raw' | 'cache' | 'unknown'. Prints detailed diagnostics."""
    print(f"\n{'=' * 70}\nDATA CHECK: {data_dir}\n{'=' * 70}")
    if not os.path.isdir(data_dir):
        print(f"❌ Does not exist.")
        parent = os.path.dirname(data_dir) or "/"
        if os.path.isdir(parent):
            print(f"   Contents of {parent}: {sorted(os.listdir(parent))[:20]}")
        return "unknown"

    entries = sorted(os.listdir(data_dir))
    print(f"Contents: {entries[:20]}{' ...' if len(entries) > 20 else ''}")

    # (a) is there an already-merged parquet?
    cache = os.path.join(data_dir, "cached_merged_multimodal.parquet")
    if os.path.exists(cache):
        print(f"\n⚡ Found merged parquet: {human(os.path.getsize(cache))}")
        print("   The pipeline will use it directly and skip the DuckDB read step (saves a few minutes).")
        return "cache"

    # (b) raw layout per architecture
    print("\nLayout per architecture:")
    found, complete, missing, partial = 0, [], [], []
    for arch in ARCHS:
        adir = os.path.join(data_dir, arch)
        if not os.path.isdir(adir):
            print(f"  {arch:8s} ❌ directory missing")
            missing.append(arch)
            continue
        parts, have = [], 0
        for mod in MODALITIES:
            p = os.path.join(adir, f"{mod}.parquet")
            if os.path.exists(p):
                parts.append(f"{mod}={human(os.path.getsize(p))}")
                found += 1
                have += 1
            else:
                parts.append(f"{mod}=MISSING")
        print(f"  {arch:8s} {' | '.join(parts)}")
        (complete if have == len(MODALITIES) else partial).append(arch)

    if found == 0:
        print("\n❌ No parquet file found in the expected layout.")
        print("   The pipeline needs: <data_dir>/<arch>/{pcap,sar,strace}.parquet")
        print("   If your files are nested one level deeper, point --data-dir at")
        print("   the parent directory that directly contains arm/ mips/ mipsel/ x86/.")
        return "unknown"

    if not complete:
        print("\n❌ NO architecture has all three modalities.")
        print("   The pipeline merges the three modalities with merge(how='inner'); without strace")
        print("   syscall_cols is empty and STRACE_CNN_LSTM(in_dim=0) will crash.")
        return "unknown"

    if missing or partial:
        print()
        if missing:
            print(f"⚠️  Architectures missing entirely: {missing}")
        if partial:
            print(f"⚠️  Missing modalities in: {partial}. These architectures will be")
            print("    almost entirely dropped by merge(how='inner'), which skews Table I relative to")
            print("    the original dataset paper. State this explicitly in the paper if intentional.")
    print(f"\n✅ Ready to run: {len(complete)}/{len(ARCHS)} architectures complete "
          f"({found}/{len(ARCHS) * len(MODALITIES)} file).")
    return "raw"


def warn_working_dir(data_dir: str) -> None:
    """Important warning when the data lives under /kaggle/working."""
    if not os.path.abspath(data_dir).startswith(WORKING):
        return
    size = dir_size(data_dir)
    print(f"\n{'=' * 70}\n⚠️  DATA IS LOCATED IN /kaggle/working\n{'=' * 70}")
    print(f"Size: {human(size)}")
    print("""
Three consequences to be aware of:

1. /kaggle/working is WIPED when the session ends. In every new session you
   must copy the data back into it, which takes time and is easy to forget.

2. Everything in /kaggle/working becomes OUTPUT when you press "Save Version".
   The data is thus duplicated into the output, counts against the 20 GB
   quota, and the dataset used to resume in the next session grows needlessly.

3. The pipeline's resume mechanism relies on attaching the previous session's
   output as the next session's input. If the output also contains the raw
   data, this loop gets heavier with every session.

RECOMMENDATION: turn the data into a separate Kaggle Dataset (Data → New Dataset)
and attach it to the notebook. It then lives at /kaggle/input/<name>, is
read-only, does not count toward the output, and needs no copying each session.

If you still want to keep it in /kaggle/working, delete the data BEFORE Save Version:
    import shutil; shutil.rmtree('/kaggle/working/dong')
(results in experiments/ are kept).
""".strip())


def restore_previous(save_dir: str, prev_dir: str | None) -> None:
    print(f"\n{'=' * 70}\nRESTORING PREVIOUS SESSION RESULTS\n{'=' * 70}")
    os.makedirs(save_dir, exist_ok=True)

    candidates = []
    if prev_dir:
        candidates.append(prev_dir)

    # Look for experiments/ directories under /kaggle/input.
    #
    # An earlier version looked at exactly one level: /kaggle/input/<ds>/experiments.
    # But Kaggle does not always mount a dataset at that level: a dataset
    # created from another notebook's output often sits deeper, for example
    # /kaggle/input/datasets/<user>/<slug>/experiments. In that case the old
    # loop found nothing, printed "this is the first session", and the
    # pipeline restarted from the first seed, wasting the entire previous GPU
    # session WITHOUT any error. (The old loop also computed a variable `sub`
    # and never used it, so the deeper branch was never actually exercised.)
    # The walk now descends up to 4 levels.
    inp = "/kaggle/input"
    if os.path.isdir(inp):
        for root, dirs, _files in os.walk(inp):
            depth = root[len(inp):].count(os.sep)
            if depth >= 4:
                dirs[:] = []
                continue
            if "experiments" in dirs:
                candidates.append(os.path.join(root, "experiments"))
                dirs.remove("experiments")   # do not descend into it

    seen, uniq = set(), []
    for c in candidates:
        if c in seen or not os.path.isdir(c):
            continue
        seen.add(c)
        uniq.append(c)

    for i, c in enumerate(uniq):
        print(f"  {'→ PICK  ' if i == 0 else '  skip  '} {c}  ({human(dir_size(c))})")

    if not uniq:
        print("  No previous results found. This is the first session.")
        print("  If you INTEND to resume, this is an ERROR: pass")
        print("  --prev <path to experiments/> explicitly.")
        return

    if len(uniq) > 1:
        print(f"\n  ⚠️  Found {len(uniq)} experiments/ directories in the input. The script")
        print("     takes the FIRST one. If that is not the one you want, pass")
        print("     --prev <path> to specify it.")

    picked = uniq[0]

    shutil.copytree(picked, save_dir, dirs_exist_ok=True)
    print(f"  ♻️  Copied {picked} → {save_dir}")

    for name in ("robustness_global/overall_robustness_raw_10seed.csv",
                 "robustness_per_architecture/per_architecture_robustness_raw_10seed.csv",
                 "lofo_zeroday_generalization/lofo_zeroday_generalization_raw_10seed.csv"):
        p = os.path.join(save_dir, name)
        if os.path.exists(p):
            try:
                import pandas as pd
                d = pd.read_csv(p)
                extra = ""
                if "Run" in d.columns:
                    seeds = sorted({int(v) for v in d["Run"]
                                    if str(v).lstrip("-").isdigit()})
                    extra = f", seed: {seeds}"
                elif "Seed" in d.columns:
                    extra = f", seed: {sorted(d['Seed'].dropna().unique().tolist())}"
                print(f"     {os.path.basename(name):52s} {len(d):6,d} rows{extra}")
            except Exception as e:
                print(f"     {os.path.basename(name)}: could not be read ({e})")


def check_gpu() -> None:
    print(f"\n{'=' * 70}\nHARDWARE\n{'=' * 70}")
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20)
        print("GPU:", out.stdout.strip() or "not found")
    except Exception:
        print("GPU: could not run nvidia-smi")
    try:
        total, used, free = shutil.disk_usage(WORKING)
        print(f"/kaggle/working free / total: {human(free)} / {human(total)}")
    except Exception:
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description="Set up a Kaggle session for HIN-Dynamic.")
    ap.add_argument("--data-dir", default=DEFAULT_DATA,
                    help='Data path, or "auto" to detect it (default).')
    ap.add_argument("--save-dir", default=DEFAULT_SAVE)
    ap.add_argument("--prev", default=None,
                    help="experiments/ directory of the previous session (if it cannot be detected).")
    ap.add_argument("--phase", default="1", choices=sorted(PHASES),
                    help="Session configuration (see README.md).")
    ap.add_argument("--no-restore", action="store_true")
    ap.add_argument("--seeds", default=None,
                    help="Comma-separated list of seeds, overriding the phase's "
                         "seeds. Example: --seeds 10,42,100. Used to split "
                         "work by SEED across several machines: each seed is "
                         "computed in full in exactly one process, so merging "
                         "has no duplicate keys and no source to choose.")
    ap.add_argument("--budget-hours", type=float, default=None,
                    help="Wall-clock time budget (hours). Overrides the phase's "
                         "value. 0 = unlimited. Kaggle allows 12 h per GPU "
                         "session, and a 'Save & Run All' that runs past the limit saves NOTHING.")
    args, _unknown = ap.parse_known_args()

    check_gpu()

    # [v3] Auto-detect the data path instead of hardcoding it. See rhgt_paths.py.
    if args.data_dir == "auto":
        try:
            import rhgt_paths
        except ImportError:
            print("❌ Could not import rhgt_paths.py. Copy it next to this file,")
            print("   or pass --data-dir <path> explicitly.")
            if "IPython" not in sys.modules:
                sys.exit(1)
            return
        found, _kind, cands = rhgt_paths.find_data_dir(verbose=False)
        print(f"\n{'=' * 70}\nDETECTING DATA PATH\n{'=' * 70}")
        rhgt_paths.report(cands)
        if found is None:
            print("\n❌ STOPPING: could not detect the data. Pass --data-dir explicitly.")
            if "IPython" not in sys.modules:
                sys.exit(1)
            return
        args.data_dir = found

    kind = check_data(args.data_dir)
    warn_working_dir(args.data_dir)
    if not args.no_restore:
        restore_previous(args.save_dir, args.prev)

    env = {
        "RHGT_DATA_DIR": args.data_dir,
        "RHGT_SAVE_DIR": args.save_dir,
        "RHGT_RESUME": "1",
    }
    if kind == "cache":
        env["RHGT_CACHE_PARQUET"] = os.path.join(
            args.data_dir, "cached_merged_multimodal.parquet")
    env.update(PHASES[args.phase])
    if args.budget_hours is not None:
        env["RHGT_TIME_BUDGET_S"] = str(int(args.budget_hours * 3600))
    if args.seeds:
        seeds = [t.strip() for t in args.seeds.split(",") if t.strip()]
        bad = [t for t in seeds if not t.lstrip("-").isdigit()]
        if bad:
            sys.exit(f"--seeds contains a non-integer value: {bad}")
        env["RHGT_SEEDS"] = ",".join(seeds)
    os.environ.update(env)

    print(f"\n{'=' * 70}\nENVIRONMENT VARIABLES (phase {args.phase})\n{'=' * 70}")
    for k in sorted(env):
        print(f"  {k:22s} = {env[k]}")

    print_phase_doc(args.phase)

    if kind == "unknown":
        print("\n" + "!" * 70)
        print("❌ STOPPING: the data is not usable yet. Fix --data-dir and rerun this cell.")
        print("   Do NOT run code-rhgt-v2.py until the section above reports ✅.")
        print("!" * 70)
        # In a notebook, sys.exit raises an ugly SystemExit traceback; only
        # exit hard when running from the command line.
        if "IPython" not in sys.modules:
            sys.exit(1)
        return

    print(f"\n✅ Ready. Next cell:\n     !python3 code-rhgt-v2.py")
    if args.phase == "smoke":
        print("   (smoke-test mode: 1 seed, Clean only, LOFO on one family)")


if __name__ == "__main__":
    main()
