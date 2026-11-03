#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
separate_mechanisms.py — Mechanism-separation experiments for the protocol gap.

Four experiments that separate the mechanisms behind the protocol gap:

  E1  Protocol S   — all test files in ONE graph, sharing entities (as in the pipeline).
  E2  Protocol I   — isolated ego-graphs, batch 256 (as in the pipeline).
  E3  Protocol T   — EACH test file is attached to the TRAIN GRAPH. Entities aggregate the
                     population of train files; there is no test->test path. T keeps the
                     "aggregate statistics of the entity" but removes all information
                     coming from other test samples.
  E4  Batch-invariance on REAL DATA, REAL checkpoints: Protocol I with
                     batch = 1 (fully isolated), 16, 128, 256.
  E5  Threshold calibration: pick the F1-maximising threshold on the ego-graphs of the
                     TRAIN SET and apply it to Protocol I on test (no fine-tuning, no test data).
  E6  Label composition: score M target files in a Protocol S graph that has K accompanying
                     test files that are ALL benign / ALL malware / random. The accompanying
                     files are balanced by architecture (K/4 per ISA) so that labels are not
                     confounded with ISA.
                     Δp = p(malware companions) − p(benign companions) is a DIRECT measure
                     of "labels propagate between test files".
  (optional) E7   Robust + strict FT variants of each model, to obtain F1*_I in the
                     SAME run (comparing "S inflates" against "I drops").

How to read the results (printed by the `analyze` command):
  share_T = (F1_T − F1_I) / (F1_S − F1_I)
     ≈ 1  → the advantage of S comes from population statistics (topology bias), NOT leakage
     ≈ 0  → the advantage of S requires information from other test samples (leakage)
  Δp ≈ 0 and label-flip rate ≈ 0  → no evidence that labels propagate between test samples
  HAN: gap S − I(bs=1) ≈ 0      → HAN is genuinely immune; the argument in §3.7/§7.5 must be rewritten

----------------------------------------------------------------------------------
RUNNING ON KAGGLE (T4)
----------------------------------------------------------------------------------
  Cell 1:  !cp /kaggle/input/<dataset-code>/*.py .
           %run kaggle_setup.py              # sets RHGT_DATA_DIR / RHGT_CACHE_PARQUET as usual
  Cell 2:  !RHGT_SEP_QUICK=1 python separate_mechanisms.py run      # smoke test, ~5 minutes
  Cell 3:  !python separate_mechanisms.py run                       # full run
  Cell 4:  !python separate_mechanisms.py analyze                   # tables + interpretation

  Output: /kaggle/working/separation/   (change with RHGT_SEP_OUT)
  Resume: rerun the same command; finished (seed, model) units are skipped and saved
          weights are reloaded instead of retrained.

Environment variables (all have defaults):
  RHGT_CODE            path to code-rhgt-v2.py (default: next to this script / cwd)
  RHGT_SEP_OUT         output directory                      [/kaggle/working/separation]
  RHGT_SEP_SEEDS       "10,42,..."                            [the 10 seeds of the paper]
  RHGT_SEP_MODELS      "RGCN,R2-RGCN,MAGNN,HAN,GAT,HGT"       [+ S2,S3,S4 if desired]
  RHGT_SEP_WITH_FT     1 = add robust + strict FT variants   [0]
  RHGT_SEP_K           number of companion files in E6       [512]
  RHGT_SEP_M           number of target files per E6 graph   [16]
  RHGT_SEP_T_BATCH     number of test files inserted at once into the train graph in E3.
                       1 = EXACT (default). >1 is faster but files inserted in the same
                       round influence each other through entities with weight ~B/degree(entity).
  RHGT_SEP_SORT_HASH   1 = sort by Hash before splitting    [1]
                       (fixes the non-reproducible split caused by DuckDB's GROUP BY order, §10.4)
  RHGT_SEP_QUICK       1 = smoke test: 1 seed, 2 models, 5 epochs, 64 samples for E3/E6
  RHGT_TIME_BUDGET_S   stop cleanly before the deadline (e.g. 39600 = 11 hours)

Methodological notes:
  * The original pipeline does NOT save weights, so models must be retrained. The new
    models are not bit-identical to the runs in the paper. Therefore EVERY S / I / T / E4 /
    E5 / E6 comparison is made on the SAME new checkpoints, and the new S − I gap is
    reported again to cross-check against Table 14.
  * Hyperparameters are kept as in the pipeline: 100 epochs, Adam lr 5e-3, wd 5e-4, hidden 64,
    4 heads, 2 layers, k = 3. Nothing is tuned.
  * E5 selects the threshold on the ego-graphs of the train set — data the model saw during
    training (in-sample) — so it is a deployable control (it never touches test), not an
    unbiased estimate. The oracle threshold on test is recorded only for diagnosis.
"""

from __future__ import annotations

import copy
import glob
import importlib.util
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------------
# 0. Configuration
# ---------------------------------------------------------------------------------
_env = os.environ.get
QUICK = _env('RHGT_SEP_QUICK', '0') == '1'
OUT = _env('RHGT_SEP_OUT', '/kaggle/working/separation'
           if os.path.isdir('/kaggle/working') else './separation')
DEFAULT_SEEDS = [10, 42, 100, 314, 555, 777, 999, 1234, 2024, 2026]
SEEDS = ([int(s) for s in _env('RHGT_SEP_SEEDS').split(',') if s.strip()]
         if _env('RHGT_SEP_SEEDS') else ([10] if QUICK else DEFAULT_SEEDS))
MODELS = ([m.strip() for m in _env('RHGT_SEP_MODELS').split(',') if m.strip()]
          if _env('RHGT_SEP_MODELS') else
          (['RGCN', 'HAN'] if QUICK else ['RGCN', 'R2-RGCN', 'MAGNN', 'HAN', 'GAT', 'HGT']))
WITH_FT = _env('RHGT_SEP_WITH_FT', '0') == '1'
K_COMP = int(_env('RHGT_SEP_K', '512'))
M_TGT = int(_env('RHGT_SEP_M', '16'))
T_BATCH = max(1, int(_env('RHGT_SEP_T_BATCH', '1')))
SORT_HASH = _env('RHGT_SEP_SORT_HASH', '1') == '1'
EPOCHS = 5 if QUICK else 100
LIMIT = 64 if QUICK else None          # limit on the number of test samples for E3/E6 in the smoke test
TIME_BUDGET_S = float(_env('RHGT_TIME_BUDGET_S', '0') or 0)
BATCH_SIZES = [1, 16, 128, 256]
ARCHS = ['ARM', 'MIPS', 'MIPSEL', 'x86']
TEST_TRAIN_RATIO = 0.25                # 20/80, for the corrected resampled t
T0 = time.time()

RES_CSV = os.path.join(OUT, 'sep_results.csv')          # metrics per (seed, model, experiment)
DP_CSV = os.path.join(OUT, 'sep_label_composition.csv')  # Δp summary
BI_CSV = os.path.join(OUT, 'sep_batch_invariance.csv')   # differences by batch size
DONE_CSV = os.path.join(OUT, 'sep_done.csv')             # finished units (resume)
WDIR = os.path.join(OUT, 'weights')
PDIR = os.path.join(OUT, 'probs')
SDIR = os.path.join(OUT, 'splits')


# ---------------------------------------------------------------------------------
# 1. Load the original pipeline as a module (without running main)
# ---------------------------------------------------------------------------------
def _find_code():
    cands = [_env('RHGT_CODE'),
             os.path.join(os.path.dirname(os.path.abspath(__file__)), 'code-rhgt-v2.py'),
             os.path.join(os.getcwd(), 'code-rhgt-v2.py')]
    cands += sorted(glob.glob('/kaggle/input/**/code-rhgt-v2.py', recursive=True))
    for c in cands:
        if c and os.path.isfile(c):
            return c
    sys.exit('❌ code-rhgt-v2.py not found — set RHGT_CODE=<path>.')


def load_pipeline():
    # code-rhgt-v2.py writes run_config.json into BASE_SAVE_DIR AS SOON AS it is imported. Redirect
    # to this script's directory so the configuration of the main run is not overwritten.
    orig_save = _env('RHGT_SAVE_DIR')
    os.makedirs(OUT, exist_ok=True)
    os.environ['RHGT_SAVE_DIR'] = os.path.join(OUT, '_pipeline')
    path = _find_code()
    spec = importlib.util.spec_from_file_location('rhgt', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    print(f'📦 Pipeline: {path}')
    return mod, orig_save


# ---------------------------------------------------------------------------------
# 2. Data — the same column-selection logic as main() in code-rhgt-v2.py
# ---------------------------------------------------------------------------------
def load_dataframe(rhgt, orig_save):
    cands = [_env('RHGT_CACHE_PARQUET'),
             os.path.join(orig_save, 'cached_merged_multimodal.parquet') if orig_save else None,
             os.path.join(rhgt.LOCAL_DATA_DIR, 'cached_merged_multimodal.parquet')]
    for c in cands:
        if c and os.path.isfile(c):
            print(f'⚡ Cache parquet: {c}')
            return pd.read_parquet(c)
    print('⏳ No cache — reading the original parquet with DuckDB (a few minutes).')
    return rhgt.load_and_merge_multimodal_parquet(rhgt.BASE_DATA_DIR)


def feature_columns(df):
    all_cols = df.columns.tolist()
    syscall_cols = [c for c in all_cols if c.startswith('Call_') or 'syscall' in c.lower()]
    proto_list = [c for c in ['TCP', 'UDP', 'ICMP', 'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SSH',
                              'IRC', 'DHCP', 'ARP', 'IGMP', 'IPv', 'LLC'] if c in all_cols]
    exclude = set(['label', 'arch', 'Arch', 'Hash', 'MalwareFamily', 'stratify_col']
                  + syscall_cols + proto_list)
    pcap_cols = [c for c in all_cols if c in ['Header_Length', 'Protocol Type', 'Time_To_Live',
                 'Rate', 'Tot sum', 'Min', 'Max', 'AVG', 'Std', 'Tot size', 'IAT', 'Number',
                 'Variance'] or c.endswith('_flag_number') or c.endswith('_count')]
    sar_cols = [c for c in all_cols if c not in exclude and c not in pcap_cols]
    return pcap_cols + sar_cols, syscall_cols, proto_list


def prepare_df(df):
    d = df.copy()
    d['stratify_col'] = d['label'].astype(str) + '_' + d['arch'].astype(str)
    keep = d['stratify_col'].value_counts()[lambda x: x > 1].index
    d = d[d['stratify_col'].isin(keep)].copy()
    if SORT_HASH:
        d = d.sort_values(['Hash', 'arch'], kind='mergesort').reset_index(drop=True)
    return d


# ---------------------------------------------------------------------------------
# 3. Models
# ---------------------------------------------------------------------------------
def registry(rhgt):
    return {
        'RGCN':    (rhgt.HIN_Vanilla_RGCN, {}),
        'R2-RGCN': (rhgt.HIN_R2_RGCN, {}),
        'MAGNN':   (rhgt.HIN_MAGNN, {}),
        'HAN':     (rhgt.HIN_HAN, {'heads': 4}),
        'GAT':     (rhgt.HIN_GAT, {'heads': 4}),
        'HGT':     (rhgt.HIN_HGT_Stage1_Vanilla, {'heads': 4, 'layers': 2}),
        'S2':      (rhgt.HIN_HGT_Stage2_ResLN, {'heads': 4, 'layers': 2}),
        'S3':      (rhgt.HIN_HGT_Stage3_NodeReliability, {'heads': 4, 'layers': 2}),
        'S4':      (rhgt.HIN_HGT_Stage4_RelationGating, {'heads': 4, 'layers': 2}),
    }


def new_model(rhgt, name, metadata, g_tr):
    cls, kw = registry(rhgt)[name]
    m = cls(hidden=64, out=2, metadata=metadata, **kw).to(rhgt.device)
    with rhgt.torch.no_grad():               # materialise the lazy Linear(-1, ·)
        m.eval()
        m(g_tr.x_dict, g_tr.edge_index_dict)
    return m


def _unit_seed(seed, name, salt=0):
    return (seed * 1_000_003 + sum(ord(c) * (i + 1) for i, c in enumerate(name)) * 7919
            + salt) % (2 ** 31 - 1)


# ---------------------------------------------------------------------------------
# 4. Inference protocols
# ---------------------------------------------------------------------------------
def probs_from_logits(torch, logits):
    return torch.softmax(logits, dim=1).detach().cpu().numpy()


def run_protocol_S(rhgt, model, g_te):
    torch = rhgt.torch
    model.eval()
    g = g_te.to(rhgt.device)
    with torch.no_grad():
        return probs_from_logits(torch, model(g.x_dict, g.edge_index_dict))


def run_protocol_I(rhgt, model, egos, batch_size):
    """Protocol I, returns (probs, logits). batch_size = 1 is fully isolated."""
    torch = rhgt.torch
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(egos), batch_size):
            b = rhgt.Batch.from_data_list(egos[i:i + batch_size]).to(rhgt.device)
            outs.append(model(b.x_dict, b.edge_index_dict).detach().cpu())
    logits = torch.cat(outs, 0)
    return probs_from_logits(torch, logits), logits.numpy()


def active_entities(rhgt, df_te, syscall_cols, proto_list, arch_le):
    """Entities touched by each test file — same rule as build_isolated_ego_graphs_batch."""
    sys_mat = rhgt.clean_numeric_array(df_te, syscall_cols)
    proto_mat = rhgt.clean_numeric_array(df_te, proto_list)
    known = set(arch_le.classes_)
    arch_ids = [int(arch_le.transform([str(a)])[0]) if str(a) in known else -1
                for a in df_te['arch']]
    act = {'syscall': [], 'protocol': [], 'arch': []}
    for i in range(len(df_te)):
        act['syscall'].append(np.where(sys_mat[i] > 0)[0] if sys_mat.shape[1] else np.empty(0, int))
        act['protocol'].append(np.where(proto_mat[i] > 0)[0] if proto_mat.shape[1] else np.empty(0, int))
        act['arch'].append(np.array([arch_ids[i]]) if arch_ids[i] >= 0 else np.empty(0, int))
    return act


def run_protocol_T(rhgt, model, g_tr, x_te_file, act, idx_list, t_batch=1):
    """Protocol T: insert test files into the TRAIN GRAPH; connect only to their own entities,
    the 'similar' edge is a self-loop (as in Protocol S). With t_batch = 1 there is no
    path whatsoever between two test files."""
    torch = rhgt.torch
    dev = rhgt.device
    model.eval()
    x_tr = {k: v.to(dev) for k, v in g_tr.x_dict.items()}
    e_tr = {k: v.to(dev) for k, v in g_tr.edge_index_dict.items()}
    N = x_tr['file'].size(0)
    xf = torch.as_tensor(x_te_file, dtype=torch.float32, device=dev)
    out = np.zeros((len(idx_list), 2), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(idx_list), t_batch):
            chunk = idx_list[s:s + t_batch]
            xd = dict(x_tr)
            xd['file'] = torch.cat([x_tr['file'], xf[chunk]], 0)
            ed = {}
            for et, ei in e_tr.items():
                src, _, dst = et
                adds = []
                for j, i in enumerate(chunk):
                    node = N + j
                    if src == 'file' and dst == 'file':
                        adds.append(torch.tensor([[node], [node]], device=dev))
                    elif src == 'file' and dst in act:
                        ids = torch.as_tensor(act[dst][i], dtype=torch.long, device=dev)
                        if ids.numel():
                            adds.append(torch.stack([torch.full_like(ids, node), ids]))
                    elif dst == 'file' and src in act:
                        ids = torch.as_tensor(act[src][i], dtype=torch.long, device=dev)
                        if ids.numel():
                            adds.append(torch.stack([ids, torch.full_like(ids, node)]))
                ed[et] = torch.cat([ei] + adds, 1) if adds else ei
            logits = model(xd, ed)[N:N + len(chunk)]
            out[s:s + len(chunk)] = probs_from_logits(torch, logits)
    return out


# ---------------------------------------------------------------------------------
# 5. E6 — label-composition graphs (built ONCE per seed, shared by all models)
# ---------------------------------------------------------------------------------
def build_label_comp_graphs(rhgt, df_te, num_cols, syscall_cols, proto_list, scaler, arch_le,
                            seed, n_targets=None):
    rng = np.random.RandomState(_unit_seed(seed, 'labelcomp'))
    n = len(df_te) if n_targets is None else min(n_targets, len(df_te))
    order = rng.permutation(len(df_te))[:n]
    y = df_te['label'].values.astype(int)
    a = np.array([str(x).upper() for x in df_te['arch']])
    k_per = max(1, K_COMP // 4)
    graphs = {'benign': [], 'malware': [], 'natural': []}
    for s in range(0, n, M_TGT):
        block = order[s:s + M_TGT]
        mask = np.ones(len(df_te), bool)
        mask[block] = False
        for cond, cls in (('benign', 0), ('malware', 1)):
            comp = []
            for arch in ARCHS:
                pool = np.where(mask & (y == cls) & (a == arch.upper()))[0]
                if len(pool):
                    comp.append(rng.choice(pool, size=min(k_per, len(pool)), replace=False))
            comp = np.concatenate(comp) if comp else np.empty(0, int)
            g, _, _ = rhgt.build_hetero_graph_hin(df_te.iloc[np.concatenate([block, comp])],
                                                  num_cols, syscall_cols, proto_list,
                                                  scaler=scaler, arch_le=arch_le, mode='test')
            graphs[cond].append((block, g))
        pool = np.where(mask)[0]
        comp = rng.choice(pool, size=min(K_COMP, len(pool)), replace=False)
        g, _, _ = rhgt.build_hetero_graph_hin(df_te.iloc[np.concatenate([block, comp])],
                                              num_cols, syscall_cols, proto_list,
                                              scaler=scaler, arch_le=arch_le, mode='test')
        graphs['natural'].append((block, g))
    return order, graphs


def run_label_comp(rhgt, model, order, graphs):
    torch = rhgt.torch
    model.eval()
    res = {}
    pos = {int(i): k for k, i in enumerate(order)}
    with torch.no_grad():
        for cond, items in graphs.items():
            p = np.zeros((len(order), 2), np.float32)
            for block, g in items:
                g = g.to(rhgt.device)
                pr = probs_from_logits(torch, model(g.x_dict, g.edge_index_dict))[:len(block)]
                for j, i in enumerate(block):
                    p[pos[int(i)]] = pr[j]
            res[cond] = p
    return res


# ---------------------------------------------------------------------------------
# 6. Writing results
# ---------------------------------------------------------------------------------
def best_threshold(y, p1):
    from sklearn.metrics import precision_recall_curve
    pr, rc, th = precision_recall_curve(y, p1)
    f1 = 2 * pr[:-1] * rc[:-1] / np.clip(pr[:-1] + rc[:-1], 1e-12, None)
    k = int(np.nanargmax(f1))
    return float(th[k])


def metric_rows(rhgt, y, probs, arch, seed, model, exp, variant='', thr=0.5, extra=None):
    p1 = probs[:, 1]
    pred = (p1 >= thr).astype(int) if thr != 0.5 else np.argmax(probs, 1)
    rows = []
    for a in ['Overall'] + ARCHS:
        mask = None if a == 'Overall' else (_up(arch) == a.upper())
        if mask is not None and mask.sum() == 0:
            continue
        r = rhgt.evaluate_metrics(y, pred, probs, f'{model}|{exp}', seed,
                                  arch_mask=mask, arch_name=a)
        if not r:
            continue
        for k in ('Training Time (s)', 'Inference (ms/sample)', 'Peak GPU Memory (MB)'):
            r.pop(k, None)
        r.update({'Seed': seed, 'Model': model, 'Exp': exp, 'Variant': variant,
                  'Threshold': thr, 'N': int(len(y) if mask is None else mask.sum())})
        if extra:
            r.update(extra)
        rows.append(r)
    return rows


def _up(a):
    return np.array([str(x).upper() for x in a])


def append_csv(path, rows):
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(path, mode='a', header=not os.path.exists(path), index=False)


def save_probs(seed, model, exp, y, probs, arch, hashes):
    safe = ''.join(c if c.isalnum() or c in '._-' else '_' for c in f'{model}_{exp}')
    np.savez_compressed(os.path.join(PDIR, f'seed{seed}_{safe}.npz'),
                        y_true=y.astype(np.int8), y_prob=probs.astype(np.float32),
                        arch=arch.astype('U8'), hash=hashes.astype('U64'))


def done_units():
    if not os.path.exists(DONE_CSV):
        return set()
    d = pd.read_csv(DONE_CSV)
    return set(zip(d['Seed'].astype(int), d['Model'].astype(str)))


def budget_left(unit_times):
    if TIME_BUDGET_S <= 0:
        return True
    need = sorted(unit_times)[int(0.8 * (len(unit_times) - 1))] if unit_times else 0
    left = TIME_BUDGET_S - (time.time() - T0)
    if left <= need:
        print(f'⏳ Stopping on budget: {left/60:.0f} min left, the next unit needs ~{need/60:.0f} min.')
        return False
    return True


# ---------------------------------------------------------------------------------
# 7. One unit (seed, model)
# ---------------------------------------------------------------------------------
def run_unit(rhgt, ctx, seed, name, ft=False):
    torch = rhgt.torch
    label = f'{name} (robust + strict FT)' if ft else name
    wpath = os.path.join(WDIR, f'seed{seed}_{label.replace(" ", "_").replace("+", "p")}.pt')
    g_tr, g_te = ctx['g_tr'], ctx['g_te']
    y, arch, hashes = ctx['y_te'], ctx['arch_te'], ctx['hash_te']

    rhgt.seed_everything(_unit_seed(seed, label))
    model = new_model(rhgt, name, ctx['metadata'], g_tr)
    t_train = 0.0
    if os.path.exists(wpath):
        model.load_state_dict(torch.load(wpath, map_location=rhgt.device))
        print(f'      ♻️  loaded weights {os.path.basename(wpath)}')
    else:
        t0 = time.perf_counter()
        if ft:
            model, _, _ = rhgt.train_robust_hin_gnn(model, g_tr, epochs=EPOCHS, lr=0.005,
                                                    weight_decay=5e-4, adv_eps=0.05)
            model, _, _ = rhgt.fine_tune_strict_inductive_hin(
                model, ctx['egos_tr'], epochs=(1 if QUICK else rhgt.STRICT_FT_EPOCHS),
                lr=rhgt.STRICT_FT_LR, weight_decay=1e-4, adv_eps=rhgt.STRICT_FT_ADV_EPS,
                clean_weight=rhgt.STRICT_FT_CLEAN_WEIGHT,
                batch_size=rhgt.STRICT_FT_BATCH_SIZE, seed=seed)
        else:
            model, _, _ = rhgt.train_standard_hin_gnn(model, g_tr, epochs=EPOCHS,
                                                      lr=0.005, weight_decay=5e-4)
        t_train = time.perf_counter() - t0
        torch.save(model.state_dict(), wpath)
    model.eval()
    rows, bi_rows, dp_rows = [], [], []
    extra = {'TrainTime_s': round(t_train, 2), 'SortHash': int(SORT_HASH), 'Epochs': EPOCHS}

    # E1 — Protocol S
    pS = run_protocol_S(rhgt, model, g_te)
    rows += metric_rows(rhgt, y, pS, arch, seed, label, 'S', extra=extra)
    save_probs(seed, label, 'S', y, pS, arch, hashes)

    # E2 + E4 — Protocol I at several batch sizes
    pI, logits = {}, {}
    for bs in BATCH_SIZES:
        pI[bs], logits[bs] = run_protocol_I(rhgt, model, ctx['egos_te'], bs)
        rows += metric_rows(rhgt, y, pI[bs], arch, seed, label, 'I', variant=f'bs={bs}', extra=extra)
    save_probs(seed, label, 'I_bs1', y, pI[1], arch, hashes)
    save_probs(seed, label, 'I_bs256', y, pI[256], arch, hashes)
    for bs in BATCH_SIZES[1:]:
        bi_rows.append({'Seed': seed, 'Model': label, 'BatchSize': bs, 'RefBatch': 1,
                        'MaxAbsLogitDiff': float(np.abs(logits[bs] - logits[1]).max()),
                        'MaxAbsProbDiff': float(np.abs(pI[bs][:, 1] - pI[1][:, 1]).max()),
                        'PredFlips': int((pI[bs].argmax(1) != pI[1].argmax(1)).sum())})

    # E5 — threshold calibration on the ego-graphs of the train set
    pTr, _ = run_protocol_I(rhgt, model, ctx['egos_tr'], 256)
    thr_tr = best_threshold(ctx['y_tr'], pTr[:, 1])
    thr_or = best_threshold(y, pI[256][:, 1])
    rows += metric_rows(rhgt, y, pI[256], arch, seed, label, 'I_thr_train', variant='bs=256',
                        thr=thr_tr, extra=extra)
    rows += metric_rows(rhgt, y, pI[256], arch, seed, label, 'I_thr_oracle', variant='bs=256',
                        thr=thr_or, extra=extra)

    if not ft:
        # E3 — Protocol T
        idx = list(range(len(y))) if LIMIT is None else list(range(min(LIMIT, len(y))))
        t0 = time.perf_counter()
        pT = run_protocol_T(rhgt, model, g_tr, ctx['x_te_file'], ctx['act'], idx, T_BATCH)
        tT = time.perf_counter() - t0
        sel = np.array(idx)
        rows += metric_rows(rhgt, y[sel], pT, arch[sel], seed, label, 'T',
                            variant=f'tbatch={T_BATCH}', extra={**extra, 'Time_s': round(tT, 1)})
        if LIMIT is not None:      # when samples are limited, record S/I on the same subset for a fair comparison
            rows += metric_rows(rhgt, y[sel], pS[sel], arch[sel], seed, label, 'S_subset', extra=extra)
            rows += metric_rows(rhgt, y[sel], pI[256][sel], arch[sel], seed, label, 'I_subset',
                                variant='bs=256', extra=extra)
        save_probs(seed, label, 'T', y[sel], pT, arch[sel], hashes[sel])

        # E6 — label composition
        order, graphs = ctx['lc_order'], ctx['lc_graphs']
        lc = run_label_comp(rhgt, model, order, graphs)
        yo, ao = y[order], arch[order]
        for cond, p in lc.items():
            rows += metric_rows(rhgt, yo, p, ao, seed, label, f'LC_{cond}',
                                variant=f'K={K_COMP},M={M_TGT}', extra=extra)
        dp = lc['malware'][:, 1] - lc['benign'][:, 1]
        flips = lc['malware'].argmax(1) != lc['benign'].argmax(1)
        np.savez_compressed(os.path.join(PDIR, f'seed{seed}_{label}_labelcomp.npz'),
                            order=order, y=yo, arch=ao.astype('U8'),
                            p_benign=lc['benign'][:, 1], p_malware=lc['malware'][:, 1],
                            p_natural=lc['natural'][:, 1])
        for grp, m in [('Overall', np.ones(len(dp), bool)), ('y=0', yo == 0), ('y=1', yo == 1)] + \
                      [(a, _up(ao) == a.upper()) for a in ARCHS]:
            if m.sum() == 0:
                continue
            dp_rows.append({'Seed': seed, 'Model': label, 'Group': grp, 'N': int(m.sum()),
                            'MeanDp': float(dp[m].mean()), 'MedianDp': float(np.median(dp[m])),
                            'MeanAbsDp': float(np.abs(dp[m]).mean()),
                            'FracAbsDpGt0.1': float((np.abs(dp[m]) > 0.1).mean()),
                            'FlipRate': float(flips[m].mean()),
                            'K': K_COMP, 'M': M_TGT})

    append_csv(RES_CSV, rows)
    append_csv(BI_CSV, bi_rows)
    append_csv(DP_CSV, dp_rows)
    append_csv(DONE_CSV, [{'Seed': seed, 'Model': label,
                           'Finished_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}])
    f = {r['Exp'] + (r['Variant'] and f"[{r['Variant']}]"): r['F1'] for r in rows
         if r['Architecture'] == 'Overall'}
    print('      ' + ' | '.join(f'{k}={v:.4f}' for k, v in f.items()))
    del model
    rhgt.clean_gpu_memory()


# ---------------------------------------------------------------------------------
# 8. The `run` command
# ---------------------------------------------------------------------------------
def cmd_run():
    for d in (OUT, WDIR, PDIR, SDIR):
        os.makedirs(d, exist_ok=True)
    rhgt, orig_save = load_pipeline()
    reg = registry(rhgt)
    bad = [m for m in MODELS if m not in reg]
    if bad:
        sys.exit(f'❌ Invalid model: {bad}. Valid: {list(reg)}')

    cfg = {'seeds': SEEDS, 'models': MODELS, 'with_ft': WITH_FT, 'K': K_COMP, 'M': M_TGT,
           't_batch': T_BATCH, 'sort_hash': SORT_HASH, 'epochs': EPOCHS, 'quick': QUICK,
           'batch_sizes': BATCH_SIZES,
           'torch': rhgt.torch.__version__,
           'gpu': (rhgt.torch.cuda.get_device_name(0) if rhgt.torch.cuda.is_available() else 'CPU'),
           'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    with open(os.path.join(OUT, 'sep_config.json'), 'w') as fh:
        json.dump(cfg, fh, indent=2)
    print(f'⚙️  {cfg}')

    df = prepare_df(load_dataframe(rhgt, orig_save))
    num_cols, syscall_cols, proto_list = feature_columns(df)
    print(f'📊 {len(df)} samples | {len(num_cols)} continuous columns | {len(syscall_cols)} syscall | '
          f'{len(proto_list)} protocol')

    done = done_units()
    units = [(m, False) for m in MODELS] + ([(m, True) for m in MODELS] if WITH_FT else [])
    unit_times = []
    for seed in SEEDS:
        todo = [(m, ft) for m, ft in units
                if (seed, f'{m} (robust + strict FT)' if ft else m) not in done]
        if not todo:
            print(f'♻️  [SEED {seed}] already done — skipping.')
            continue
        print(f'\n🌱 [SEED {seed}] 80/20 split by Hash, stratified by (label, ISA)')
        rhgt.seed_everything(seed)
        df_tr, df_te = rhgt.group_safe_stratified_split(df, group_col='Hash', label_col='label',
                                                        test_size=0.2, seed=seed)
        with open(os.path.join(SDIR, f'seed{seed}_test_hashes.txt'), 'w') as fh:
            fh.write('\n'.join(df_te['Hash'].astype(str)))
        g_tr, scaler, arch_le = rhgt.build_hetero_graph_hin(df_tr, num_cols, syscall_cols,
                                                            proto_list, mode='train')
        g_te, _, _ = rhgt.build_hetero_graph_hin(df_te, num_cols, syscall_cols, proto_list,
                                                 scaler=scaler, arch_le=arch_le, mode='test')
        g_tr = g_tr.to(rhgt.device)
        anchors = rhgt.RectifiedStrictInductiveEngine.extract_canonical_anchors(g_tr)
        build = rhgt.RectifiedStrictInductiveEngine.build_isolated_ego_graphs_batch
        ctx = {
            'g_tr': g_tr, 'g_te': g_te, 'metadata': g_tr.metadata(),
            'egos_tr': build(df_tr, num_cols, syscall_cols, proto_list, scaler, arch_le, anchors),
            'egos_te': build(df_te, num_cols, syscall_cols, proto_list, scaler, arch_le, anchors),
            'y_tr': df_tr['label'].values.astype(int),
            'y_te': df_te['label'].values.astype(int),
            'arch_te': np.array([str(x).upper().replace('X86', 'x86') for x in df_te['arch']], dtype='U8'),
            'hash_te': df_te['Hash'].astype(str).values,
            'x_te_file': scaler.transform(rhgt.clean_numeric_array(df_te, num_cols)).astype(np.float32),
            'act': active_entities(rhgt, df_te, syscall_cols, proto_list, arch_le),
        }
        t0 = time.perf_counter()
        ctx['lc_order'], ctx['lc_graphs'] = build_label_comp_graphs(
            rhgt, df_te, num_cols, syscall_cols, proto_list, scaler, arch_le, seed, LIMIT)
        print(f'   🧩 E6: {sum(len(v) for v in ctx["lc_graphs"].values())} label-composition graphs '
              f'({time.perf_counter() - t0:.0f} s)')

        for name, ft in todo:
            if not budget_left(unit_times):
                print('💾 All completed results written. Rerun the command to continue.')
                return
            print(f'   ⚡ [SEED {seed}] {name}{" (robust + strict FT)" if ft else ""}')
            t0 = time.time()
            run_unit(rhgt, ctx, seed, name, ft=ft)
            unit_times.append(time.time() - t0)
            print(f'      ⏱️  {unit_times[-1] / 60:.1f} min | total {(time.time() - T0) / 3600:.2f} hours')
        del ctx, g_tr, g_te
        rhgt.clean_gpu_memory()
    print(f'\n✅ Done. Results in {OUT}. Run: python separate_mechanisms.py analyze')


# ---------------------------------------------------------------------------------
# 9. The `analyze` command — CPU only
# ---------------------------------------------------------------------------------
def _paired(x, y):
    """Returns (mean diff, sd, t, p_t, p_wilcoxon, p_corr) for x − y paired by seed."""
    from scipy import stats
    d = np.asarray(x, float) - np.asarray(y, float)
    n = len(d)
    md = float(d.mean())
    sd = float(d.std(ddof=1)) if n > 1 else float('nan')
    if n < 2 or sd == 0:
        return md, sd, float('nan'), float('nan'), float('nan'), float('nan')
    t = md / (sd / math.sqrt(n))
    p_t = 2 * stats.t.sf(abs(t), n - 1)
    t_c = md / (sd * math.sqrt(1 / n + TEST_TRAIN_RATIO))     # Nadeau & Bengio (2003)
    p_c = 2 * stats.t.sf(abs(t_c), n - 1)
    try:
        p_w = float(stats.wilcoxon(d, method='exact').pvalue) if np.any(d != 0) else 1.0
    except Exception:
        p_w = float('nan')
    return md, sd, t, p_t, p_w, p_c


def _holm(ps):
    ps = np.asarray(ps, float)
    out = np.full_like(ps, np.nan)
    ok = ~np.isnan(ps)
    idx = np.argsort(ps[ok])
    m = ok.sum()
    adj, run = np.empty(m), 0.0
    vals = ps[ok][idx]
    for r, v in enumerate(vals):
        run = max(run, min(1.0, (m - r) * v))
        adj[r] = run
    tmp = np.empty(m)
    tmp[idx] = adj
    out[ok] = tmp
    return out


def _f(x, k=4):
    return '—' if x is None or (isinstance(x, float) and math.isnan(x)) else f'{x:.{k}f}'


def _p(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return '—'
    return f'{x:.1e}' if x < 1e-3 else f'{x:.3f}'


def cmd_analyze():
    if not os.path.exists(RES_CSV):
        sys.exit(f'❌ Missing {RES_CSV}')
    R = pd.read_csv(RES_CSV)
    R = R.drop_duplicates(subset=['Seed', 'Model', 'Exp', 'Variant', 'Architecture'], keep='last')
    O = R[R.Architecture == 'Overall']
    L = []
    w = L.append

    def f1(model, exp, variant=None, arch='Overall'):
        d = R[(R.Model == model) & (R.Exp == exp) & (R.Architecture == arch)]
        if variant is not None:
            d = d[d.Variant == variant]
        return d.set_index('Seed')['F1']

    models = [m for m in O.Model.unique() if '(robust + strict FT)' not in m]
    w('# Mechanism-separation experiment results\n')
    w(f'Source: `{RES_CSV}` — {O.Seed.nunique()} seed, {len(models)} standard models.\n')
    quick = (O['N'].min() < 1000) if 'N' in O else False
    exp_T = 'T'
    S_key, I_key = ('S_subset', 'I_subset') if (O.Exp == 'S_subset').any() else ('S', 'I')
    I_var = 'bs=256'
    if quick:
        w('> ⚠️ Smoke-test run (limited samples) — NOT for the paper.\n')

    # ---- A. S / I / T --------------------------------------------------------
    w('## A. Protocol S, I, T and the share of the gap explained by population statistics\n')
    w('share_T = (F1_T − F1_I) / (F1_S − F1_I). ≈1: topology bias; ≈0: requires test↔test information.\n')
    w('| Model | F1_S | F1_I | F1_T | S−I | p_Holm | p_corr | T−I | S−T | p_Holm(S−T) | share_T |')
    w('|---|---|---|---|---|---|---|---|---|---|---|')
    rowsA, pSI, pST = [], [], []
    for m in models:
        s = f1(m, S_key)
        i = f1(m, I_key, I_var)
        t = f1(m, exp_T)
        seeds = s.index.intersection(i.index).intersection(t.index)
        if len(seeds) == 0:
            continue
        s, i, t = s[seeds], i[seeds], t[seeds]
        si = _paired(s, i)
        st = _paired(s, t)
        ti = _paired(t, i)
        share = (t.mean() - i.mean()) / (s.mean() - i.mean()) if abs(s.mean() - i.mean()) > 1e-9 else float('nan')
        rowsA.append((m, s, i, t, si, st, ti, share))
        pSI.append(si[3])
        pST.append(st[3])
    hSI, hST = _holm(pSI), _holm(pST)
    for k, (m, s, i, t, si, st, ti, share) in enumerate(rowsA):
        w(f'| {m} | {_f(s.mean())}±{_f(s.std(), 4)} | {_f(i.mean())}±{_f(i.std(), 4)} | '
          f'{_f(t.mean())}±{_f(t.std(), 4)} | {si[0]:+.4f} | {_p(hSI[k])} | {_p(si[5])} | '
          f'{ti[0]:+.4f} | {st[0]:+.4f} | {_p(hST[k])} | {_f(share, 2)} |')
    w('')
    big = [(m, sh) for (m, s, i, t, si, st, ti, sh) in rowsA if si[0] >= 0.02 and not math.isnan(sh)]
    ns_A = O[O.Exp == exp_T].Seed.nunique()
    if ns_A < 3:
        w(f'→ Only {ns_A} seed(s) — not enough to conclude (need ≥ 3, 10 recommended).')
        big_ok = False
    else:
        big_ok = True
    if not big and big_ok:
        w('→ No model has F1_S − F1_I ≥ 0.02 in this run — Table 14 is not reproduced; '
          'recheck the configuration before interpreting.')
    if big and big_ok:
        mean_share = float(np.mean([sh for _, sh in big]))
        w(f'**Mean share_T over {len(big)} models with gap S−I ≥ 0.02: {mean_share:.2f}.**')
        if mean_share >= 0.8:
            w('→ Most of the advantage of Protocol S can be reproduced with the TRAIN graph, without any other test file. '
              'The gap is **topology bias / aggregate entity statistics**, not '
              'leakage. The title, Abstract, §1.1–1.3, §10.6 and §11 must be reframed.')
        elif mean_share <= 0.2:
            w('→ The train graph does NOT replace the accompanying test files: the advantage of S needs '
              'test↔test information. The "shared-entity leakage" claim is **confirmed**; report this table as '
              'direct evidence.')
        else:
            w('→ Both mechanisms coexist. Report share_T as the decomposition ratio '
              '(topology bias vs. test↔test information).')
    w('')

    # ---- A2. theo ISA --------------------------------------------------------
    w('### A2. By architecture (mean over seeds and over models with F1_S − F1_I ≥ 0.02)\n')
    w('| ISA | F1_S | F1_I | F1_T | S−I | T−I |')
    w('|---|---|---|---|---|---|')
    bigm = [m for m, _ in big] or models
    for a in ARCHS:
        vals = []
        for m in bigm:
            s, i, t = f1(m, S_key, arch=a), f1(m, I_key, I_var, arch=a), f1(m, exp_T, arch=a)
            seeds = s.index.intersection(i.index).intersection(t.index)
            if len(seeds):
                vals.append((s[seeds].mean(), i[seeds].mean(), t[seeds].mean()))
        if vals:
            v = np.mean(vals, 0)
            w(f'| {a} | {_f(v[0])} | {_f(v[1])} | {_f(v[2])} | {v[0]-v[1]:+.4f} | {v[2]-v[1]:+.4f} |')
    w('')

    # ---- B. Label composition ----------------------------------------------------
    if os.path.exists(DP_CSV):
        D = pd.read_csv(DP_CSV).drop_duplicates(subset=['Seed', 'Model', 'Group'], keep='last')
        w('## B. Label composition of the accompanying test files (E6)\n')
        w('Δp = p(malware | all-malware companions) − p(malware | all-benign companions), same target file, '
          'same number of companion files per ISA.\n')
        w('| Model | mean Δp | p (t, over seeds) | mean |Δp| | label-flip rate | F1_benign-comp | F1_malware-comp | F1_natural |')
        w('|---|---|---|---|---|---|---|---|')
        from scipy import stats
        for m in models:
            d = D[(D.Model == m) & (D.Group == 'Overall')]
            if d.empty:
                continue
            mdp = d.MeanDp.values
            p = stats.ttest_1samp(mdp, 0).pvalue if len(mdp) > 1 and mdp.std() > 0 else float('nan')
            fb, fm, fn = (f1(m, f'LC_{c}').mean() for c in ('benign', 'malware', 'natural'))
            w(f'| {m} | {mdp.mean():+.4f} | {_p(p)} | {d.MeanAbsDp.mean():.4f} | '
              f'{d.FlipRate.mean():.4f} | {_f(fb)} | {_f(fm)} | {_f(fn)} |')
        w('')
        w('By ISA (mean Δp, pooled over all models):')
        for a in ARCHS:
            d = D[D.Group == a]
            if not d.empty:
                w(f'- {a}: {d.MeanDp.mean():+.4f} (label flips {d.FlipRate.mean():.4f})')
        w('')
        allm = D[D.Group == 'Overall']
        if not allm.empty:
            ns = allm.Seed.nunique()
            per_model_p = []
            for m in models:
                v = allm[allm.Model == m].MeanDp.values
                per_model_p.append(stats.ttest_1samp(v, 0).pvalue
                                   if len(v) > 1 and v.std() > 0 else float('nan'))
            hp = _holm(per_model_p)
            sig = [m for m, q, v in zip(models, hp, [allm[allm.Model == mm].MeanDp.mean() for mm in models])
                   if not math.isnan(q) and q < 0.05 and v >= 0.01]
            if ns < 3:
                w(f'→ Only {ns} seed(s) — not enough to conclude (need ≥ 3, 10 recommended).')
            elif sig:
                w(f'→ In {len(sig)}/{len(models)} models ({", ".join(sig)}), predictions are pulled toward the label '
                  'of the companion files (mean Δp ≥ 0.01, p_Holm < 0.05). This is **direct evidence** of '
                  'the leakage channel — put this table in §7 as the main result.')
            else:
                w('→ No model has mean Δp ≥ 0.01 with p_Holm < 0.05: **no evidence for '
                  '"labels propagate between test files"**. Remove this phrase from the Abstract.')
        w('')

    # ---- C. Batch-invariance ---------------------------------------------------
    if os.path.exists(BI_CSV):
        B = pd.read_csv(BI_CSV).drop_duplicates(subset=['Seed', 'Model', 'BatchSize'], keep='last')
        w('## C. Batch-invariance on real data (relative to batch = 1)\n')
        w('| Model | max|Δlogit| bs16 | bs128 | bs256 | label flips (total, bs256) | F1_I bs1 | F1_I bs256 | S − I(bs1) | p_Holm |')
        w('|---|---|---|---|---|---|---|---|---|---|')
        allm = sorted(B.Model.unique())
        ps, cache = [], []
        for m in allm:
            s = f1(m, 'S')
            i1 = f1(m, 'I', 'bs=1')
            seeds = s.index.intersection(i1.index)
            st = _paired(s[seeds], i1[seeds]) if len(seeds) else (float('nan'),) * 6
            ps.append(st[3])
            cache.append((m, st))
        hs = _holm(ps)
        for k, (m, st) in enumerate(cache):
            b = B[B.Model == m]
            mx = {bs: b[b.BatchSize == bs].MaxAbsLogitDiff.max() for bs in (16, 128, 256)}
            fl = int(b[b.BatchSize == 256].PredFlips.sum())
            w(f'| {m} | {mx[16]:.1e} | {mx[128]:.1e} | {mx[256]:.1e} | {fl} | '
              f'{_f(f1(m, "I", "bs=1").mean())} | {_f(f1(m, "I", "bs=256").mean())} | '
              f'{st[0]:+.4f} | {_p(hs[k])} |')
        w('')
        if 'HAN' in allm:
            st = dict(cache)['HAN']
            if B[B.Model == 'HAN'].Seed.nunique() < 3:
                w('→ HAN: not enough seeds to conclude (need ≥ 3).')
            elif abs(st[0]) < 0.005 and (math.isnan(st[3]) or st[3] > 0.05):
                w('→ HAN at batch = 1 (fully isolated) still shows NO gap: the immunity of HAN is '
                  'a **genuine architectural property**. Rewrite §3.7, §7.5 and the sentence in the Abstract; keep the † mark '
                  'only as a note on batch-dependence.')
            else:
                w(f'→ HAN at batch = 1 has gap {st[0]:+.4f}: coupling through the batch masked the gap. '
                  'The §7.5 argument stands — replace the synthetic-graph numbers with this table.')
        w('')

    # ---- D. Threshold calibration ----------------------------------------------------
    w('## D. Threshold calibration only (E5) vs. realign\n')
    has_ft = any('(robust + strict FT)' in m for m in O.Model.unique())
    w('| Model | F1_I @0.5 | F1_I @train-threshold | F1_I @oracle | train-threshold (median) | '
      + ('F1_I realign | share recovered by threshold |' if has_ft else '|'))
    w('|---|---|---|---|---|' + ('---|---|' if has_ft else ''))
    for m in models:
        i0 = f1(m, 'I', I_var).mean()
        it = f1(m, 'I_thr_train', I_var).mean()
        io = f1(m, 'I_thr_oracle', I_var).mean()
        th = R[(R.Model == m) & (R.Exp == 'I_thr_train') & (R.Architecture == 'Overall')].Threshold.median()
        line = f'| {m} | {_f(i0)} | {_f(it)} | {_f(io)} | {_f(th, 4)} |'
        if has_ft:
            fm = f'{m} (robust + strict FT)'
            ifz = f1(fm, 'I', I_var).mean()
            share = (it - i0) / (ifz - i0) if abs(ifz - i0) > 1e-9 else float('nan')
            line += f' {_f(ifz)} | {_f(share, 2)} |'
        w(line)
    w('')
    w('Note: the train-threshold is chosen on the ego-graphs of the train set (in-sample, never touches test); '
      'the oracle threshold is chosen on test, for diagnosis only.\n')

    # ---- E. Does S inflate or does I drop? -------------------------------------------------
    if has_ft:
        w('## E. Protocol S vs. the best inductive estimate in the SAME run\n')
        w('If Protocol S really "inflates", then F1_S > F1*_I (I after realign).\n')
        w('| Model | F1_S (standard) | F1*_I (robust + strict FT) | S − I* | p_Holm |')
        w('|---|---|---|---|---|')
        ps, cache = [], []
        for m in models:
            s = f1(m, 'S')
            i_ = f1(f'{m} (robust + strict FT)', 'I', I_var)
            seeds = s.index.intersection(i_.index)
            if len(seeds) == 0:
                continue
            st = _paired(s[seeds], i_[seeds])
            ps.append(st[3])
            cache.append((m, s[seeds].mean(), i_[seeds].mean(), st))
        hs = _holm(ps)
        for k, (m, sm, im, st) in enumerate(cache):
            w(f'| {m} | {_f(sm)} | {_f(im)} | {st[0]:+.4f} | {_p(hs[k])} |')
        w('')

    w('---\nTests: paired t over seeds, exact Wilcoxon (in the CSV), Holm within each family; '
      'p_corr = Nadeau–Bengio corrected resampled t (n_test/n_train = 0.25).')
    txt = '\n'.join(L)
    path = os.path.join(OUT, 'SEPARATION_REPORT.md')
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(txt)
    print(txt)
    print(f'\n📝 {path}')


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'run'
    if cmd == 'run':
        cmd_run()
    elif cmd == 'analyze':
        cmd_analyze()
    else:
        sys.exit('Usage: python separate_mechanisms.py [run|analyze]')
