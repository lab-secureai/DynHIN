# =========================================================================================
# R-HGT dynamic-HIN pipeline: training and evaluation.
# Environment variables: RHGT_SEEDS, RHGT_EXTRA, RHGT_SCENARIOS, RHGT_RESUME,
#                  RHGT_SKIP_MAIN, RHGT_SKIP_LOFO, RHGT_LOFO_FAMILIES,
#                  RHGT_ATTACK_ITERS, RHGT_ATTACK_RESTARTS, RHGT_SAVE_PROBS
# =========================================================================================
# =========================================================================================
# PUBLICATION-GRADE RESEARCH PIPELINE: MULTIMODAL HIN GNNs & BASELINES BENCHMARK
# STRICT-INDUCTIVE INFERENCE WITH CANONICAL TRAINING-ONLY ENTITY ANCHORS
# ZERO TEST-TO-TEST MESSAGE PASSING FOR THE MAIN BENCHMARK, 10-SEED REPRODUCIBILITY,
# HASH-GROUPED STRATIFIED SPLITTING & CONTROLLED 6-STAGE R-HGT PIPELINE ABLATION
# =========================================================================================

import os
import gc
import json
import time
import random
import copy
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any

# GPU Memory & PyTorch Optimizations
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, precision_recall_curve, roc_auc_score,
    confusion_matrix, auc
)

# 1. Dependency Ingestion
try:
    import duckdb
except ImportError:
    os.system("pip install -q duckdb")
    import duckdb

try:
    from torch_geometric.data import Data, HeteroData, Batch
    import torch_geometric.transforms as T
    from torch_geometric.nn import HeteroConv, SAGEConv, Linear, HGTConv, GATConv, HANConv, GCNConv
except ImportError:
    print("⏳ Installing PyTorch Geometric & Scientific Ecosystem...")
    os.system("pip install -q torch_geometric scikit-learn pandas pyarrow fastparquet openpyxl imbalanced-learn duckdb")
    from torch_geometric.data import Data, HeteroData, Batch
    import torch_geometric.transforms as T
    from torch_geometric.nn import HeteroConv, SAGEConv, Linear, HGTConv, GATConv, HANConv, GCNConv

# =========================================================================================
# 2. CONFIGURATION, DIRECTORIES & 10 REPRODUCIBLE SEEDS
# =========================================================================================
DEFAULT_SEEDS = [10, 42, 100, 314, 555, 777, 999, 1234, 2024, 2026]
# Publication runs use all 10 seeds by default. Set RHGT_QUICK_TEST=1 only for smoke tests.
QUICK_TEST = os.environ.get('RHGT_QUICK_TEST', '0') == '1'
SEEDS = [DEFAULT_SEEDS[0]] if QUICK_TEST else DEFAULT_SEEDS

# ===================== TIME BUDGET CONTROL v2 =====================
# Designed for time-limited sessions (Kaggle ~9 hours). Everything is
# resumable: seeds that already completed are skipped on the next run.
_env = os.environ.get

if _env('RHGT_SEEDS'):                      # e.g. "10,42,100"
    SEEDS = [int(s) for s in _env('RHGT_SEEDS').split(',') if s.strip()]

EXTRA_GROUPS = _env('RHGT_EXTRA', '').upper()        # '', 'A', 'AC', 'ABC'
RESUME = _env('RHGT_RESUME', '1') == '1'
SKIP_MAIN = _env('RHGT_SKIP_MAIN', '0') == '1'
SKIP_LOFO = _env('RHGT_SKIP_LOFO', '0') == '1'
SCENARIO_SET = _env('RHGT_SCENARIOS', 'all').lower()  # all | clean | noise | attack
LOFO_FAMILIES_ENV = _env('RHGT_LOFO_FAMILIES', '')    # e.g. "Mirai,DarkNexus"
SAVE_PROBS = _env('RHGT_SAVE_PROBS', '1') == '1'

print(f"⚙️  v2 | seeds={SEEDS} | extra={EXTRA_GROUPS or 'none'} | "
      f"scenarios={SCENARIO_SET} | resume={RESUME} | "
      f"skip_main={SKIP_MAIN} | skip_lofo={SKIP_LOFO}")

# Strict-inductive alignment fine-tuning. Override from shell if needed.
STRICT_FT_EPOCHS = int(os.environ.get('RHGT_STRICT_FT_EPOCHS', '2' if QUICK_TEST else '15'))
STRICT_FT_LR = float(os.environ.get('RHGT_STRICT_FT_LR', '5e-4'))
STRICT_FT_BATCH_SIZE = int(os.environ.get('RHGT_STRICT_FT_BATCH', '64'))
STRICT_FT_ADV_EPS = float(os.environ.get('RHGT_STRICT_FT_ADV_EPS', '0.03'))
STRICT_FT_CLEAN_WEIGHT = float(os.environ.get('RHGT_STRICT_FT_CLEAN_WEIGHT', '0.70'))
STRICT_FT_GRAD_CLIP = float(os.environ.get('RHGT_STRICT_FT_GRAD_CLIP', '5.0'))

# =====================================================================
# [v3 / I1] WALL-CLOCK TIME BUDGET
#
# Kaggle allows 12 hours per GPU session. With "Save & Run All", the notebook MUST
# finish within 12 hours; if it overruns, NOTHING is saved and both the session
# and the GPU quota are lost. Set RHGT_TIME_BUDGET_S so that the pipeline stops
# CLEANLY before that limit: it writes all results obtained so far and exits the loop;
# a later session with RHGT_RESUME=1 continues where it stopped.
#
#     RHGT_TIME_BUDGET_S=39600    # 11 hours, leaving a 1-hour safety margin
#
# The stop decision is based on the MEASURED duration of completed iterations, not
# on a manual estimate: if the next iteration (80th percentile, for safety) is unlikely
# to finish before the deadline, stop now instead of starting it and being killed midway.
# =====================================================================
TIME_BUDGET_S = float(os.environ.get('RHGT_TIME_BUDGET_S', '0') or 0)
_RUN_T0 = time.time()
_ITER_TIMES = []          # duration of each (seed, family) pair in LOFO
_MAIN_TIMES = []          # duration of each seed in the main loop  [v3 / I4]
_LOFO_LAST = [None]
_MAIN_LAST = [None]
_BUDGET_HIT = [False]


def _budget_exceeded(now=None, tag='', times=None):
    """True if the next iteration is unlikely to finish within the time budget.

    times: durations of the COMPLETED iterations of the loop being considered.
    The main loop and the LOFO loop have very different costs, so two separate
    lists are used; mixing them would give a wrong estimate.
    """
    if _BUDGET_HIT[0]:          # [v3 / I4] once one loop has stopped, all loops stop
        return True
    if TIME_BUDGET_S <= 0:
        return False
    if now is None:
        now = time.time()
    if times is None:
        times = _ITER_TIMES
    elapsed = now - _RUN_T0
    left = TIME_BUDGET_S - elapsed
    if times:
        xs = sorted(times)
        need = xs[min(len(xs) - 1, int(0.8 * len(xs)))]
    else:
        need = 0.0
    if left <= need:
        print(f"\n{'=' * 70}")
        print(f"⏳ [v3] STOPPING: TIME BUDGET REACHED")
        print(f"{'=' * 70}")
        print(f"   used      : {elapsed / 3600:.2f} h / {TIME_BUDGET_S / 3600:.2f} h")
        print(f"   remaining : {left / 60:.0f} min")
        print(f"   next iter : needs ~{need / 60:.0f} min  ({tag})")
        print(f"   → not enough time; stopping cleanly and writing the results obtained so far.")
        return True
    if times:
        print(f"      ⏱️  used {elapsed / 3600:.2f} h | remaining {left / 3600:.2f} h | "
              f"last iteration {times[-1] / 60:.1f} min")
    return False
LOCAL_DATA_DIR = '/content/local_dataset'
DRIVE_DATA_DIR = '/content/drive/MyDrive/dataset/Dong'

if os.path.exists(LOCAL_DATA_DIR) and len(os.listdir(LOCAL_DATA_DIR)) > 0:
    BASE_DATA_DIR = LOCAL_DATA_DIR
    print(f"⚡ Using Local NVMe SSD: {BASE_DATA_DIR}")
elif os.path.exists(DRIVE_DATA_DIR):
    BASE_DATA_DIR = DRIVE_DATA_DIR
    print(f"📂 Using Google Drive Directory: {BASE_DATA_DIR}")
else:
    BASE_DATA_DIR = '.'

DRIVE_SAVE_DIR = '/content/drive/MyDrive/dataset/Dong/experiments_hin_frozen_inductive_sota_20.8All'
BASE_SAVE_DIR = DRIVE_SAVE_DIR if os.path.exists('/content/drive/MyDrive') else './experiments_hin_frozen_inductive_sota_20.8All'

# [v2] The base paths are hardcoded for Colab. On Kaggle (or any other machine)
# point them elsewhere via environment variables:
#   RHGT_DATA_DIR=/kaggle/input/<dataset>        (containing arm/ mips/ mipsel/ x86/)
#   RHGT_SAVE_DIR=/kaggle/working/experiments    (results; allows resuming in a later session)
if os.environ.get('RHGT_DATA_DIR'):
    BASE_DATA_DIR = os.environ['RHGT_DATA_DIR']
    print(f"📂 [v2] BASE_DATA_DIR = {BASE_DATA_DIR}")
if os.environ.get('RHGT_SAVE_DIR'):
    BASE_SAVE_DIR = os.environ['RHGT_SAVE_DIR']
    print(f"💾 [v2] BASE_SAVE_DIR = {BASE_SAVE_DIR}")
if os.environ.get('RHGT_CACHE_PARQUET'):
    # Parquet already merged in a previous session - skip the DuckDB read step entirely (a few minutes).
    LOCAL_DATA_DIR = os.path.dirname(os.environ['RHGT_CACHE_PARQUET']) or '.'
    print(f"⚡ [v2] Using parquet cache: {os.environ['RHGT_CACHE_PARQUET']}")

ROBUSTNESS_GLOBAL_DIR = os.path.join(BASE_SAVE_DIR, 'robustness_global')
ROBUSTNESS_ARCH_DIR = os.path.join(BASE_SAVE_DIR, 'robustness_per_architecture')
LOFO_DIR = os.path.join(BASE_SAVE_DIR, 'lofo_zeroday_generalization')
CHECKPOINT_DIR = os.path.join(BASE_SAVE_DIR, 'checkpoints')
STAT_DIR = os.path.join(BASE_SAVE_DIR, 'dataset_statistics')

PROB_DUMP_DIR = os.path.join(BASE_SAVE_DIR, 'prob_dumps')   # [v2] for the threshold sweep

for d in [ROBUSTNESS_GLOBAL_DIR, ROBUSTNESS_ARCH_DIR, LOFO_DIR, CHECKPOINT_DIR,
          STAT_DIR, PROB_DUMP_DIR]:
    os.makedirs(d, exist_ok=True)

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"⚡ Active Compute Device: {device} | Pure Inductive Frozen Engine: Ready")


def _write_run_config():
    """[v3] Write the full configuration to run_config.json.

    The paper states "no hyperparameter tuning; all values fixed in advance" -
    this file is verifiable evidence for that claim, and also documents the
    "experimental environment" that journals typically require.
    """
    import json as _json, platform as _plat, datetime as _dt
    cfg = {
        # [v3 / H3] utcnow() is deprecated in Python 3.12 (a warning was seen in an earlier session).
        'timestamp_utc': _dt.datetime.now(_dt.timezone.utc)
                            .isoformat().replace('+00:00', 'Z'),
        'python': _plat.python_version(),
        'platform': _plat.platform(),
        'gpu': (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'),
        'versions': {},
        'seeds': SEEDS,
        'hyperparameters': {
            'hidden_dim': 64, 'heads': 4, 'layers': 2,
            'dropout_hin': 0.2, 'dropout_tabular': 0.3,
            'knn_k_hin': 3, 'knn_k_homogeneous': 5, 'test_size': 0.2,
            'epochs_tabular': 30, 'epochs_fusion': 25, 'epochs_hin': 100,
            'lr_tabular': 1e-3, 'lr_fusion_backbone': 1e-4,
            'lr_fusion_head': 1e-3, 'lr_hin': 5e-3, 'weight_decay_hin': 5e-4,
            'adv_eps_stage5': 0.05, 'adv_lambda_stage5': 0.5,
            'strict_ft_epochs': STRICT_FT_EPOCHS,
            'strict_ft_lr': STRICT_FT_LR,
            'strict_ft_batch': STRICT_FT_BATCH_SIZE,
            'strict_ft_adv_eps': STRICT_FT_ADV_EPS,
            'strict_ft_clean_weight': STRICT_FT_CLEAN_WEIGHT,
            'strict_ft_grad_clip': STRICT_FT_GRAD_CLIP,
            'strict_ft_optimizer': 'AdamW',
            'strict_ft_scheduler': 'CosineAnnealingLR',
        },
        'env': {k: v for k, v in os.environ.items() if k.startswith('RHGT_')},
        'no_tuning_declaration': (
            'All hyperparameters were fixed a priori and were NOT tuned on the '
            'test set. No validation split is used because no hyperparameter '
            'selection step was performed.'),
    }
    for _m in ('torch', 'torch_geometric', 'numpy', 'pandas', 'sklearn', 'duckdb'):
        try:
            cfg['versions'][_m] = __import__(_m).__version__
        except Exception:
            cfg['versions'][_m] = 'n/a'
    try:
        with open(os.path.join(BASE_SAVE_DIR, 'run_config.json'), 'w',
                  encoding='utf-8') as _f:
            _json.dump(cfg, _f, indent=2, ensure_ascii=False)
        print(f"📝 [v3] run_config.json written "
              f"(torch {cfg['versions']['torch']}, "
              f"PyG {cfg['versions']['torch_geometric']}, gpu {cfg['gpu']})")
    except Exception as _e:
        print(f"[!] Could not write run_config.json: {_e}")


_write_run_config()

def set_eval_mode_safe(model):
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.LSTM, nn.GRU, nn.RNN)):
            m.train()
            if hasattr(m, 'dropout'):
                m.dropout = 0.0

def clean_gpu_memory(*objs):
    """[v2 / FIX C1] The original only ran `del` on local names INSIDE the function, so
    nothing was reclaimed: the names in the caller's frame still held references, and memory
    accumulated across seeds and caused OOM. Call sites now `del` explicitly before
    calling this function; the function only performs garbage collection.
    """
    del objs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _mix_seed(seed, level=0.0, salt=0):
    """[v2 / FIX C5] Collision-free seed hashing.

    The old formula `seed + int(level*1000)` mapped different (seed, noise level) pairs
    to the same value, making supposedly 'independent' runs correlated.
    """
    x = (int(seed) * 1_000_003) ^ (int(round(float(level) * 1e6)) * 2_654_435_761)
    x ^= int(salt) * 40_503
    return int(x) % (2 ** 31 - 1)

def clean_numeric_array(df, cols):
    if not cols:
        return np.zeros((len(df), 0), dtype=np.float32)
    vals = df[cols].replace([np.inf, -np.inf], np.nan).fillna(0).values
    vals = np.nan_to_num(vals, nan=0.0, posinf=1e9, neginf=-1e9)
    vals = np.clip(vals, -1e9, 1e9).astype(np.float32)
    return vals

def knn_graph_custom(x, k=5):
    num_nodes = x.size(0)
    if num_nodes <= 1:
        idx = torch.arange(num_nodes, device=x.device, dtype=torch.long)
        return torch.stack([idx, idx], dim=0)
    k_eff = min(int(k), num_nodes - 1)
    dist = torch.cdist(x, x)
    _, knn_idx = torch.topk(dist, k=k_eff + 1, largest=False)
    knn_idx = knn_idx[:, 1:]
    row = torch.arange(num_nodes, device=x.device).repeat_interleave(k_eff)
    col = knn_idx.reshape(-1)
    return torch.stack([row, col], dim=0)

# =========================================================================================
# 3. DATASET INGESTION VIA DUCKDB WITH COMPOSITE ARCHITECTURE DEDUPLICATION
# =========================================================================================
def read_parquet_with_duckdb(file_path):
    if not os.path.exists(file_path):
        return pd.DataFrame()

    con = duckdb.connect(database=':memory:')
    try:
        desc_df = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{file_path}', union_by_name=True) LIMIT 0").df()
    except Exception:
        con.close()
        return pd.DataFrame()

    numeric_types = ['DOUBLE', 'FLOAT', 'REAL', 'DECIMAL', 'TINYINT', 'SMALLINT', 'INTEGER', 'BIGINT', 'HUGEINT', 'UTINYINT', 'USMALLINT', 'UINTEGER', 'UBIGINT']
    num_cols, cat_cols = [], []

    for _, row in desc_df.iterrows():
        c_name = row['column_name']
        c_type = str(row['column_type']).upper()
        if c_name == 'Hash':
            continue
        if any(t in c_type for t in numeric_types):
            num_cols.append(c_name)
        elif c_name in ['MalwareFamily', 'Arch', 'arch', 'label']:
            cat_cols.append(c_name)

    if 'Hash' in desc_df['column_name'].values and len(num_cols) > 0:
        agg_parts = [f'AVG(TRY_CAST("{c}" AS FLOAT)) as "{c}"' for c in num_cols]
        for c in cat_cols:
            agg_parts.append(f'FIRST("{c}") as "{c}"')
        agg_sql = ", ".join(agg_parts)
        query = f'SELECT "Hash", {agg_sql} FROM read_parquet(\'{file_path}\', union_by_name=True) GROUP BY "Hash"'
        df_agg = con.execute(query).df()
    else:
        df_agg = con.execute(f"SELECT * FROM read_parquet('{file_path}', union_by_name=True)").df()

    for c in df_agg.select_dtypes(include=['float64']).columns:
        df_agg[c] = df_agg[c].astype(np.float32)

    con.close()
    return df_agg

def load_and_merge_multimodal_parquet(base_data_dir=BASE_DATA_DIR):
    local_cache_path = os.path.join(LOCAL_DATA_DIR, 'cached_merged_multimodal.parquet')
    drive_cache_path = os.path.join(BASE_SAVE_DIR, 'cached_merged_multimodal.parquet')

    if os.path.exists(local_cache_path):
        print(f"⚡ Loading from NVMe cached parquet: {local_cache_path}")
        return pd.read_parquet(local_cache_path).copy()
    elif os.path.exists(drive_cache_path):
        print(f"📂 Loading from Drive cached parquet: {drive_cache_path}")
        return pd.read_parquet(drive_cache_path).copy()

    arch_list = ['arm', 'mips', 'mipsel', 'x86']
    dfs_arch = []

    for arch in arch_list:
        arch_dir = os.path.join(base_data_dir, arch)
        pcap_path = os.path.join(arch_dir, 'pcap.parquet')
        sar_path = os.path.join(arch_dir, 'sar.parquet')
        strace_path = os.path.join(arch_dir, 'strace.parquet')

        if not os.path.exists(pcap_path):
            continue

        df_pcap = read_parquet_with_duckdb(pcap_path)
        df_sar = read_parquet_with_duckdb(sar_path) if os.path.exists(sar_path) else pd.DataFrame()
        df_strace = read_parquet_with_duckdb(strace_path) if os.path.exists(strace_path) else pd.DataFrame()

        common_keys = ['Hash']
        if not df_sar.empty and not df_strace.empty:
            cols_sar = [c for c in df_sar.columns if c not in df_pcap.columns or c in common_keys]
            cols_strace = [c for c in df_strace.columns if (c not in df_pcap.columns and c not in df_sar.columns) or c in common_keys]

            df_merged_arch = df_pcap.merge(df_sar[cols_sar], on='Hash', how='inner')
            del df_pcap, df_sar; gc.collect()
            df_merged_arch = df_merged_arch.merge(df_strace[cols_strace], on='Hash', how='inner')
            del df_strace; gc.collect()
        else:
            df_merged_arch = df_pcap

        df_merged_arch['arch'] = arch
        dfs_arch.append(df_merged_arch)

    if not dfs_arch:
        print("⚠️ Warning: No raw parquet data found. Returning empty DataFrame.")
        return pd.DataFrame()

    df_all = pd.concat(dfs_arch, ignore_index=True).copy()
    del dfs_arch; gc.collect()

    if 'MalwareFamily' in df_all.columns and 'label' not in df_all.columns:
        df_all['label'] = (df_all['MalwareFamily'].astype(str).str.lower() != 'benign').astype(int)

    if 'Hash' in df_all.columns:
        initial_count = len(df_all)
        if 'arch' in df_all.columns:
            df_all = df_all.drop_duplicates(subset=['Hash', 'arch']).reset_index(drop=True)
            print(f"🔒 Data Integrity Audit: Removed {initial_count - len(df_all)} intra-architecture collisions (preserved cross-architecture pairings).")
        else:
            df_all = df_all.drop_duplicates(subset=['Hash']).reset_index(drop=True)
            print(f"🔒 Data Integrity Audit: Removed {initial_count - len(df_all)} cross-architecture hash duplicates.")

    try:
        df_all.to_parquet(local_cache_path, engine='pyarrow', index=False)
        df_all.to_parquet(drive_cache_path, engine='pyarrow', index=False)
    except Exception:
        pass

    return df_all

# =========================================================================================
# 4. HASH-GROUPED SAFE DATA SPLITTING (ZERO INTRA-HASH / CROSS-ARCH CONTAMINATION)
# =========================================================================================
def group_safe_stratified_split(df, group_col='Hash', label_col='label', test_size=0.2, seed=42):
    """Group-disjoint split with approximate sample-level stratification.

    StratifiedGroupKFold keeps every Hash in exactly one partition while balancing
    label x architecture strata as closely as possible. GroupShuffleSplit is used
    only as a fallback when the strata are too sparse for stratified group folds.
    """
    if len(df) < 2:
        raise ValueError("Need at least two rows for a train/test split.")

    stratify_series = (
        df['stratify_col'].astype(str)
        if 'stratify_col' in df.columns
        else df[label_col].astype(str) + '_' + df.get('arch', pd.Series(['NA'] * len(df), index=df.index)).astype(str)
    )

    if group_col not in df.columns:
        raise ValueError(f"Strict publication split requires group column '{group_col}'.")
    if df[group_col].nunique() <= 1:
        raise ValueError(f"Strict publication split requires at least two unique {group_col} groups.")

    groups = df[group_col].astype(str).values
    n_splits = max(2, int(round(1.0 / float(test_size))))
    n_splits = min(n_splits, int(df[group_col].nunique()))

    best_split = None
    try:
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        target_ratio = float(test_size)
        global_dist = stratify_series.value_counts(normalize=True)
        best_score = float('inf')
        for train_idx, test_idx in sgkf.split(df, stratify_series.values, groups=groups):
            te_strata = stratify_series.iloc[test_idx].value_counts(normalize=True)
            aligned = global_dist.index.union(te_strata.index)
            dist_error = (global_dist.reindex(aligned, fill_value=0.0) - te_strata.reindex(aligned, fill_value=0.0)).abs().sum()
            size_error = abs(len(test_idx) / len(df) - target_ratio)
            score = float(dist_error + 2.0 * size_error)
            if score < best_score:
                best_score = score
                best_split = (train_idx, test_idx)
    except ValueError:
        best_split = None

    if best_split is None:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        best_split = next(gss.split(df, stratify_series.values, groups=groups))

    train_idx, test_idx = best_split
    df_tr = df.iloc[train_idx].copy().reset_index(drop=True)
    df_te = df.iloc[test_idx].copy().reset_index(drop=True)

    train_hashes = set(df_tr[group_col].astype(str).unique())
    test_hashes = set(df_te[group_col].astype(str).unique())
    overlap = len(train_hashes.intersection(test_hashes))
    assert overlap == 0, f"FATAL DATA LEAKAGE: Found {overlap} overlapping hashes between Train and Test splits!"

    return df_tr, df_te

# =========================================================================================
# 5. HIN GRAPH GENERATORS & RECTIFIED STRICT INDUCTIVE ISOLATION ENGINE
# =========================================================================================
def build_hetero_graph_hin(df, num_cols, syscall_cols, proto_list, scaler=None, arch_le=None, mode='train', label_col='label', k_sim=3):
    x_num = clean_numeric_array(df, num_cols)

    if mode == 'train':
        scaler = StandardScaler().fit(x_num)
        arch_le = LabelEncoder().fit(df['arch'].astype(str))

    x_num_scaled = scaler.transform(x_num).astype(np.float32)
    known_archs = set(arch_le.classes_)

    arch_ids = np.array([arch_le.transform([str(x)])[0] if str(x) in known_archs else -1 for x in df['arch']])
    mask_arch = arch_ids != -1
    src_f_arch = np.arange(len(df))[mask_arch]
    dst_f_arch = arch_ids[mask_arch]

    src_f_sys, dst_f_sys = [], []
    if len(syscall_cols) > 0:
        sys_mat = clean_numeric_array(df, syscall_cols)
        nz_f, nz_s = np.where(sys_mat > 0)
        src_f_sys = nz_f.tolist()
        dst_f_sys = nz_s.tolist()

    src_f_proto, dst_f_proto = [], []
    if len(proto_list) > 0:
        proto_mat = clean_numeric_array(df, proto_list)
        nz_fp, nz_p = np.where(proto_mat > 0)
        src_f_proto = nz_fp.tolist()
        dst_f_proto = nz_p.tolist()

    num_syscalls = max(len(syscall_cols), 1)
    num_archs = max(len(arch_le.classes_), 1)
    num_protos = max(len(proto_list), 1)

    data = HeteroData()
    data['file'].x = torch.from_numpy(x_num_scaled)
    data['file'].y = torch.tensor(df[label_col].values, dtype=torch.long)
    data['syscall'].x = torch.eye(num_syscalls, dtype=torch.float32)
    data['arch'].x = torch.eye(num_archs, dtype=torch.float32)
    if len(proto_list) > 0:
        data['protocol'].x = torch.eye(num_protos, dtype=torch.float32)

    if mode == 'train':
        file_sim_edges = knn_graph_custom(data['file'].x, k=k_sim)
        data['file', 'similar', 'file'].edge_index = file_sim_edges
    else:
        num_test_files = data['file'].x.size(0)
        self_loop = torch.arange(num_test_files, device=data['file'].x.device).unsqueeze(0).repeat(2, 1)
        data['file', 'similar', 'file'].edge_index = self_loop

    if len(src_f_sys) > 0:
        data['file', 'executes', 'syscall'].edge_index = torch.tensor([src_f_sys, dst_f_sys], dtype=torch.long)
    else:
        data['file', 'executes', 'syscall'].edge_index = torch.empty((2, 0), dtype=torch.long)

    if len(src_f_arch) > 0:
        data['file', 'runs_on', 'arch'].edge_index = torch.tensor([src_f_arch.tolist(), dst_f_arch.tolist()], dtype=torch.long)
    else:
        data['file', 'runs_on', 'arch'].edge_index = torch.empty((2, 0), dtype=torch.long)

    if len(src_f_proto) > 0:
        data['file', 'uses_proto', 'protocol'].edge_index = torch.tensor([src_f_proto, dst_f_proto], dtype=torch.long)
    elif len(proto_list) > 0:
        data['file', 'uses_proto', 'protocol'].edge_index = torch.empty((2, 0), dtype=torch.long)

    data = T.ToUndirected()(data)
    return data, scaler, arch_le

def build_homogeneous_multimodal_graph(df, all_feature_cols, k=5, scaler=None, mode='train', label_col='label'):
    x_all = clean_numeric_array(df, all_feature_cols)

    if mode == 'train':
        scaler = StandardScaler().fit(x_all)

    x_scaled = scaler.transform(x_all).astype(np.float32)
    x_tensor = torch.from_numpy(x_scaled)
    y_tensor = torch.tensor(df[label_col].values, dtype=torch.long)

    if mode == 'train':
        edge_index = knn_graph_custom(x_tensor, k=k)
    else:
        num_test_nodes = x_tensor.size(0)
        edge_index = torch.arange(num_test_nodes, device=x_tensor.device).unsqueeze(0).repeat(2, 1)

    data = Data(x=x_tensor, edge_index=edge_index, y=y_tensor)
    return data, scaler

# =========================================================================================
# 6. RECTIFIED STRICT INDUCTIVE EGO-GRAPH ENGINE (ZERO DOUBLE PROJECTION & ZERO 2-HOP LEAK)
# =========================================================================================
class RectifiedStrictInductiveEngine:
    @staticmethod
    def extract_canonical_anchors(g_train: HeteroData) -> Dict[str, torch.Tensor]:
        anchors = {}
        for n_type in ['syscall', 'arch', 'protocol']:
            if n_type in g_train.x_dict:
                anchors[n_type] = g_train.x_dict[n_type].detach().cpu().clone()
        return anchors

    @staticmethod
    def build_isolated_ego_graphs_batch(
        df_te: pd.DataFrame,
        num_cols: List[str],
        syscall_cols: List[str],
        proto_list: List[str],
        scaler: StandardScaler,
        arch_le: LabelEncoder,
        canonical_anchors: Dict[str, torch.Tensor],
        label_col: str = 'label'
    ) -> List[HeteroData]:
        x_num = clean_numeric_array(df_te, num_cols)
        x_num_scaled = scaler.transform(x_num).astype(np.float32)
        known_archs = set(arch_le.classes_)
        arch_ids = [arch_le.transform([str(x)])[0] if str(x) in known_archs else -1 for x in df_te['arch']]

        sys_mat = clean_numeric_array(df_te, syscall_cols) if len(syscall_cols) > 0 else np.zeros((len(df_te), 0))
        proto_mat = clean_numeric_array(df_te, proto_list) if len(proto_list) > 0 else np.zeros((len(df_te), 0))
        labels = df_te[label_col].values

        ego_graph_list = []
        for i in range(len(df_te)):
            data_i = HeteroData()
            data_i['file'].x = torch.tensor(x_num_scaled[i:i+1], dtype=torch.float32)
            data_i['file'].y = torch.tensor([labels[i]], dtype=torch.long)

            # [v3] The original loaded the ENTIRE anchor set into EACH ego-graph, so a batch of 256
            # samples had 256·(K_sys + 4 + K_proto) anchor nodes instead of K_sys + 4 +
            # K_proto: O(N·K) complexity instead of O(N + K). This is why strict
            # inference was ~50x slower than shared (0.41 ms/sample vs 0.008 ms/sample) and why
            # the computational-cost table was inflated.
            #
            # The v3 version keeps only the anchor ROWS that are actually connected, but KEEPS the
            # one-hot feature dimensions (K_sys, 4, K_proto) unchanged so that they match the
            # Linear(-1, hidden) layer materialised from the training graph.
            #
            # Mathematically equivalent: the representation of a 'file' node only receives
            # messages from CONNECTED anchors; unconnected anchors are isolated components
            # and contribute nothing. Checked with verify_slim_ego.py.
            _slim = os.environ.get('RHGT_SLIM_EGO', '1') == '1'

            a_id = arch_ids[i]
            active_sys = (np.where(sys_mat[i] > 0)[0] if sys_mat.shape[1] > 0
                          else np.empty(0, dtype=np.int64))
            active_proto = (np.where(proto_mat[i] > 0)[0] if proto_mat.shape[1] > 0
                            else np.empty(0, dtype=np.int64))

            if _slim:
                _sel = {'syscall': active_sys,
                        'protocol': active_proto,
                        'arch': (np.array([a_id], dtype=np.int64) if a_id >= 0
                                 else np.empty(0, dtype=np.int64))}
                for n_type, feat in canonical_anchors.items():
                    idx = _sel.get(n_type, np.empty(0, dtype=np.int64))
                    if len(idx) > 0:
                        data_i[n_type].x = feat[torch.as_tensor(idx, dtype=torch.long)]
                    else:
                        # Keep one dummy node (with no edges) so that the node type always exists,
                        # avoiding a metadata mismatch between ego-graphs in the same batch.
                        data_i[n_type].x = torch.zeros_like(feat[:1])
                dst_sys = list(range(len(active_sys)))
                dst_proto = list(range(len(active_proto)))
                dst_arch = [0] if a_id >= 0 else []
            else:
                for n_type, feat in canonical_anchors.items():
                    data_i[n_type].x = feat
                dst_sys = active_sys.tolist()
                dst_proto = active_proto.tolist()
                dst_arch = [a_id] if a_id >= 0 else []

            data_i['file', 'runs_on', 'arch'].edge_index = (
                torch.tensor([[0] * len(dst_arch), dst_arch], dtype=torch.long)
                if len(dst_arch) > 0 else torch.empty((2, 0), dtype=torch.long))

            data_i['file', 'executes', 'syscall'].edge_index = (
                torch.tensor([[0] * len(dst_sys), dst_sys], dtype=torch.long)
                if len(dst_sys) > 0 else torch.empty((2, 0), dtype=torch.long))

            if proto_mat.shape[1] > 0 or 'protocol' in canonical_anchors:
                data_i['file', 'uses_proto', 'protocol'].edge_index = (
                    torch.tensor([[0] * len(dst_proto), dst_proto], dtype=torch.long)
                    if len(dst_proto) > 0 else torch.empty((2, 0), dtype=torch.long))

            data_i['file', 'similar', 'file'].edge_index = torch.tensor([[0], [0]], dtype=torch.long)
            data_i = T.ToUndirected()(data_i)
            ego_graph_list.append(data_i)

        return ego_graph_list

def predict_hin_isolated_batch(model: nn.Module, ego_graphs: List[HeteroData], batch_size: int = 256) -> Tuple[np.ndarray, float]:
    model.eval()
    probs_list = []
    num_samples = len(ego_graphs)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_start = time.perf_counter()

    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch_chunk = ego_graphs[i:i+batch_size]
            batched_data = Batch.from_data_list(batch_chunk).to(device)
            logits = model(batched_data.x_dict, batched_data.edge_index_dict)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            probs_list.append(probs)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_end = time.perf_counter()

    infer_ms_sample = ((t_end - t_start) / max(num_samples, 1)) * 1000.0
    return np.concatenate(probs_list, axis=0), infer_ms_sample

def perturb_isolated_ego_graphs(ego_graphs: List[HeteroData], noise_type: str, level: float, seed: int) -> List[HeteroData]:
    """Perturb each test ego-graph with an independent deterministic RNG stream."""
    return [
        apply_noise_to_hetero_graph(g, noise_type, level, seed=seed + 100003 * idx)
        for idx, g in enumerate(ego_graphs)
    ]

# =========================================================================================
# 7. RECTIFIED STOCHASTIC NOISE & SYMMETRIC ADVERSARIAL ATTACK ENGINES
# =========================================================================================
def apply_feature_noise_standardized(x_scaled_arr, noise_type, level, seed=42):
    np.random.seed(seed + int(level * 1000))
    x_noisy = x_scaled_arr.copy()
    if noise_type == 'Gaussian_Noise':
        noise = np.random.normal(0, level, size=x_scaled_arr.shape).astype(np.float32)
        x_noisy = x_noisy + noise
    elif noise_type == 'Feature_Masking':
        mask = np.random.binomial(1, 1.0 - level, size=x_scaled_arr.shape).astype(np.float32)
        x_noisy = x_noisy * mask
    return x_noisy

def apply_noise_to_hetero_graph(graph, noise_type, level, seed=42):
    """[v2 / FIX C4] Use a LOCAL torch.Generator instead of the global torch.manual_seed.
    The original reseeded the global generator in the middle of the evaluation loop, so the
    PGD random restarts in the next scenario were no longer independent - attack strength
    became a function of the scenario iteration order.
    """
    graph_noisy = graph.clone()
    g_rng = torch.Generator(device='cpu').manual_seed(_mix_seed(seed, level, 11))

    if noise_type == 'Gaussian_Noise':
        g_x = graph_noisy['file'].x
        noise = torch.randn(g_x.shape, generator=g_rng, dtype=g_x.dtype).to(g_x.device)
        graph_noisy['file'].x = g_x + noise * level

    elif noise_type == 'Feature_Masking':
        g_x = graph_noisy['file'].x
        mask = (torch.rand(g_x.shape, generator=g_rng, dtype=g_x.dtype) >= level).to(g_x.device)
        graph_noisy['file'].x = g_x * mask.to(g_x.dtype)

    elif noise_type == 'Edge_Perturbation':
        for et in graph_noisy.edge_types:
            edge_idx = graph_noisy[et].edge_index
            if edge_idx.numel() > 0:
                num_edges = edge_idx.size(1)
                keep_mask = (torch.rand(num_edges, generator=g_rng) >= level).to(edge_idx.device)
                graph_noisy[et].edge_index = edge_idx[:, keep_mask]

    return graph_noisy

def apply_noise_to_homo_graph(graph, noise_type, level, seed=42):
    """[v2 / FIX C4] See the note in apply_noise_to_hetero_graph."""
    graph_noisy = graph.clone()
    g_rng = torch.Generator(device='cpu').manual_seed(_mix_seed(seed, level, 22))

    if noise_type == 'Gaussian_Noise':
        x = graph_noisy.x
        noise = torch.randn(x.shape, generator=g_rng, dtype=x.dtype).to(x.device)
        graph_noisy.x = x + noise * level

    elif noise_type == 'Feature_Masking':
        x = graph_noisy.x
        mask = (torch.rand(x.shape, generator=g_rng, dtype=x.dtype) >= level).to(x.device)
        graph_noisy.x = x * mask.to(x.dtype)

    elif noise_type == 'Edge_Perturbation':
        edge_idx = graph_noisy.edge_index
        if edge_idx.numel() > 0:
            num_edges = edge_idx.size(1)
            keep_mask = (torch.rand(num_edges, generator=g_rng) >= level).to(edge_idx.device)
            graph_noisy.edge_index = edge_idx[:, keep_mask]

    return graph_noisy

def attack_tensor_model_rectified(model, x_tensor, y_tensor, attack_type='PGD', eps=0.1, alpha=None, iters=20, num_restarts=5, batch_size=512):
    set_eval_mode_safe(model)
    if alpha is None:
        alpha = max((eps / iters) if attack_type == 'BIM' else (2.0 * eps / iters), 1e-4)

    x_adv_list = []
    num_samples = x_tensor.size(0)

    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        x_chunk = x_tensor[start_idx:end_idx].clone().detach().to(device)
        y_chunk = y_tensor[start_idx:end_idx].clone().detach().to(device)

        with torch.enable_grad():
            if attack_type == 'FGSM':
                x_adv = x_chunk.clone().detach().requires_grad_(True)
                out = model(x_adv)
                if isinstance(out, tuple): out = out[0]
                loss = F.cross_entropy(out, y_chunk)
                grad = torch.autograd.grad(loss, x_adv)[0]
                x_adv = x_chunk + eps * grad.sign()
                x_adv_list.append(x_adv.detach().cpu())

            elif attack_type == 'BIM':
                x_adv = x_chunk.clone().detach()
                for _ in range(iters):
                    x_adv.requires_grad_(True)
                    out = model(x_adv)
                    if isinstance(out, tuple): out = out[0]
                    loss = F.cross_entropy(out, y_chunk)
                    grad = torch.autograd.grad(loss, x_adv)[0]
                    x_adv = x_adv.detach() + alpha * grad.sign()
                    eta = torch.clamp(x_adv - x_chunk, min=-eps, max=eps)
                    x_adv = (x_chunk + eta).detach()
                x_adv_list.append(x_adv.detach().cpu())

            elif attack_type == 'PGD':
                best_loss = torch.zeros(x_chunk.size(0), device=device) - 1e9
                best_x_adv = x_chunk.clone().detach()

                for restart in range(max(1, num_restarts)):
                    noise = torch.FloatTensor(x_chunk.shape).uniform_(-eps, eps).to(device)
                    x_adv_curr = (x_chunk + noise).clone().detach()

                    for _ in range(iters):
                        x_adv_curr.requires_grad_(True)
                        out = model(x_adv_curr)
                        if isinstance(out, tuple): out = out[0]
                        loss = F.cross_entropy(out, y_chunk)
                        grad = torch.autograd.grad(loss, x_adv_curr)[0]
                        x_adv_curr = x_adv_curr.detach() + alpha * grad.sign()
                        eta = torch.clamp(x_adv_curr - x_chunk, min=-eps, max=eps)
                        x_adv_curr = (x_chunk + eta).detach()

                    out_final = model(x_adv_curr)
                    if isinstance(out_final, tuple): out_final = out_final[0]
                    loss_per_sample = F.cross_entropy(out_final, y_chunk, reduction='none')

                    better_mask = (loss_per_sample > best_loss).unsqueeze(-1)
                    best_loss = torch.where(loss_per_sample > best_loss, loss_per_sample, best_loss)
                    best_x_adv = torch.where(better_mask, x_adv_curr, best_x_adv)

                x_adv_list.append(best_x_adv.detach().cpu())

    model.eval()
    clean_gpu_memory()
    return torch.cat(x_adv_list, dim=0).to(device)

def attack_multimodal_fusion_model_rectified(model, xp_clean, xs_clean, xst_clean, y_true_tensor, attack_type='PGD', eps=0.1, alpha=None, iters=20, num_restarts=5, batch_size=512):
    set_eval_mode_safe(model)
    if alpha is None:
        alpha = max((eps / iters) if attack_type == 'BIM' else (2.0 * eps / iters), 1e-4)

    xp_adv_list, xs_adv_list, xst_adv_list = [], [], []
    num_samples = xp_clean.size(0)

    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        xp_chunk = xp_clean[start_idx:end_idx].clone().detach().to(device)
        xs_chunk = xs_clean[start_idx:end_idx].clone().detach().to(device)
        xst_chunk = xst_clean[start_idx:end_idx].clone().detach().to(device)
        y_chunk = y_true_tensor[start_idx:end_idx].clone().detach().to(device)

        with torch.enable_grad():
            if attack_type == 'FGSM':
                xp_adv = xp_chunk.clone().detach().requires_grad_(True)
                xs_adv = xs_chunk.clone().detach().requires_grad_(True)
                xst_adv = xst_chunk.clone().detach().requires_grad_(True)

                logits, _, _ = model(xp_adv, xs_adv, xst_adv)
                loss = F.cross_entropy(logits, y_chunk)
                grads = torch.autograd.grad(loss, [xp_adv, xs_adv, xst_adv])

                xp_adv = xp_chunk + eps * grads[0].sign()
                xs_adv = xs_chunk + eps * grads[1].sign()
                xst_adv = xst_chunk + eps * grads[2].sign()
                xp_adv_list.append(xp_adv.detach().cpu())
                xs_adv_list.append(xs_adv.detach().cpu())
                xst_adv_list.append(xst_adv.detach().cpu())

            elif attack_type == 'BIM':
                xp_adv = xp_chunk.clone().detach()
                xs_adv = xs_chunk.clone().detach()
                xst_adv = xst_chunk.clone().detach()

                for _ in range(iters):
                    xp_adv.requires_grad_(True)
                    xs_adv.requires_grad_(True)
                    xst_adv.requires_grad_(True)

                    logits, _, _ = model(xp_adv, xs_adv, xst_adv)
                    loss = F.cross_entropy(logits, y_chunk)
                    grads = torch.autograd.grad(loss, [xp_adv, xs_adv, xst_adv])

                    xp_adv = xp_adv.detach() + alpha * grads[0].sign()
                    xs_adv = xs_adv.detach() + alpha * grads[1].sign()
                    xst_adv = xst_adv.detach() + alpha * grads[2].sign()

                    eta_p = torch.clamp(xp_adv - xp_chunk, min=-eps, max=eps)
                    eta_s = torch.clamp(xs_adv - xs_chunk, min=-eps, max=eps)
                    eta_st = torch.clamp(xst_adv - xst_chunk, min=-eps, max=eps)

                    xp_adv = (xp_chunk + eta_p).detach()
                    xs_adv = (xs_chunk + eta_s).detach()
                    xst_adv = (xst_chunk + eta_st).detach()

                xp_adv_list.append(xp_adv.detach().cpu())
                xs_adv_list.append(xs_adv.detach().cpu())
                xst_adv_list.append(xst_adv.detach().cpu())

            elif attack_type == 'PGD':
                best_loss = torch.zeros(xp_chunk.size(0), device=device) - 1e9
                best_xp = xp_chunk.clone().detach()
                best_xs = xs_chunk.clone().detach()
                best_xst = xst_chunk.clone().detach()

                for restart in range(max(1, num_restarts)):
                    noise_p = torch.FloatTensor(xp_chunk.shape).uniform_(-eps, eps).to(device)
                    noise_s = torch.FloatTensor(xs_chunk.shape).uniform_(-eps, eps).to(device)
                    noise_st = torch.FloatTensor(xst_chunk.shape).uniform_(-eps, eps).to(device)

                    xp_adv_curr = (xp_chunk + noise_p).clone().detach()
                    xs_adv_curr = (xs_chunk + noise_s).clone().detach()
                    xst_adv_curr = (xst_chunk + noise_st).clone().detach()

                    for _ in range(iters):
                        xp_adv_curr.requires_grad_(True)
                        xs_adv_curr.requires_grad_(True)
                        xst_adv_curr.requires_grad_(True)

                        logits, _, _ = model(xp_adv_curr, xs_adv_curr, xst_adv_curr)
                        loss = F.cross_entropy(logits, y_chunk)
                        grads = torch.autograd.grad(loss, [xp_adv_curr, xs_adv_curr, xst_adv_curr])

                        xp_adv_curr = xp_adv_curr.detach() + alpha * grads[0].sign()
                        xs_adv_curr = xs_adv_curr.detach() + alpha * grads[1].sign()
                        xst_adv_curr = xst_adv_curr.detach() + alpha * grads[2].sign()

                        eta_p = torch.clamp(xp_adv_curr - xp_chunk, min=-eps, max=eps)
                        eta_s = torch.clamp(xs_adv_curr - xs_chunk, min=-eps, max=eps)
                        eta_st = torch.clamp(xst_adv_curr - xst_chunk, min=-eps, max=eps)

                        xp_adv_curr = (xp_chunk + eta_p).detach()
                        xs_adv_curr = (xs_chunk + eta_s).detach()
                        xst_adv_curr = (xst_chunk + eta_st).detach()

                    out_final, _, _ = model(xp_adv_curr, xs_adv_curr, xst_adv_curr)
                    loss_per_sample = F.cross_entropy(out_final, y_chunk, reduction='none')

                    better_mask = (loss_per_sample > best_loss).unsqueeze(-1)
                    best_loss = torch.where(loss_per_sample > best_loss, loss_per_sample, best_loss)
                    best_xp = torch.where(better_mask, xp_adv_curr, best_xp)
                    best_xs = torch.where(better_mask, xs_adv_curr, best_xs)
                    best_xst = torch.where(better_mask, xst_adv_curr, best_xst)

                xp_adv_list.append(best_xp.detach().cpu())
                xs_adv_list.append(best_xs.detach().cpu())
                xst_adv_list.append(best_xst.detach().cpu())

    model.eval()
    clean_gpu_memory()
    return (
        torch.cat(xp_adv_list, dim=0).to(device),
        torch.cat(xs_adv_list, dim=0).to(device),
        torch.cat(xst_adv_list, dim=0).to(device)
    )

def attack_hetero_gnn_model_rectified(model, graph, attack_type='PGD', eps=0.1, alpha=None, iters=20, num_restarts=5):
    set_eval_mode_safe(model)
    if alpha is None:
        alpha = max((eps / iters) if attack_type == 'BIM' else (2.0 * eps / iters), 1e-4)

    graph_adv = graph.clone().to(device)
    x_file_orig = graph['file'].x.clone().detach().to(device)
    y_true = graph['file'].y.clone().detach().to(device)

    with torch.enable_grad():
        if attack_type == 'FGSM':
            x_file_adv = x_file_orig.clone().detach().requires_grad_(True)
            x_dict = {k: v for k, v in graph_adv.x_dict.items()}
            x_dict['file'] = x_file_adv

            out = model(x_dict, graph_adv.edge_index_dict)
            loss = F.cross_entropy(out, y_true)
            grad = torch.autograd.grad(loss, x_file_adv)[0]
            x_file_adv = x_file_orig + eps * grad.sign()
            graph_adv['file'].x = x_file_adv.detach()
            model.eval()
            return graph_adv

        elif attack_type == 'BIM':
            x_file_adv = x_file_orig.clone().detach()
            for _ in range(iters):
                x_file_adv.requires_grad_(True)
                x_dict = {k: v for k, v in graph_adv.x_dict.items()}
                x_dict['file'] = x_file_adv

                out = model(x_dict, graph_adv.edge_index_dict)
                loss = F.cross_entropy(out, y_true)
                grad = torch.autograd.grad(loss, x_file_adv)[0]

                x_file_adv = x_file_adv.detach() + alpha * grad.sign()
                eta = torch.clamp(x_file_adv - x_file_orig, min=-eps, max=eps)
                x_file_adv = (x_file_orig + eta).detach()

            graph_adv['file'].x = x_file_adv.detach()
            model.eval()
            return graph_adv

        elif attack_type == 'PGD':
            best_loss = torch.zeros(x_file_orig.size(0), device=device) - 1e9
            best_x_adv = x_file_orig.clone().detach()

            for restart in range(max(1, num_restarts)):
                noise = torch.FloatTensor(x_file_orig.shape).uniform_(-eps, eps).to(device)
                x_adv_curr = (x_file_orig + noise).clone().detach()

                for _ in range(iters):
                    x_adv_curr.requires_grad_(True)
                    x_dict = {k: v for k, v in graph_adv.x_dict.items()}
                    x_dict['file'] = x_adv_curr

                    out = model(x_dict, graph_adv.edge_index_dict)
                    loss = F.cross_entropy(out, y_true)
                    grad = torch.autograd.grad(loss, x_adv_curr)[0]

                    x_adv_curr = x_adv_curr.detach() + alpha * grad.sign()
                    eta = torch.clamp(x_adv_curr - x_file_orig, min=-eps, max=eps)
                    x_adv_curr = (x_file_orig + eta).detach()

                x_dict = {k: v for k, v in graph_adv.x_dict.items()}
                x_dict['file'] = x_adv_curr
                out_final = model(x_dict, graph_adv.edge_index_dict)
                loss_per_sample = F.cross_entropy(out_final, y_true, reduction='none')

                better_mask = (loss_per_sample > best_loss).unsqueeze(-1)
                best_loss = torch.where(loss_per_sample > best_loss, loss_per_sample, best_loss)
                best_x_adv = torch.where(better_mask, x_adv_curr, best_x_adv)

            graph_adv['file'].x = best_x_adv.detach()
            model.eval()
            return graph_adv

    model.eval()
    return graph_adv

def attack_isolated_ego_graphs_batch(model, ego_graphs: List[HeteroData], attack_type='PGD', eps=0.1, alpha=None, iters=10, num_restarts=2, batch_size=256) -> Tuple[np.ndarray, float]:
    set_eval_mode_safe(model)
    if alpha is None:
        alpha = max((eps / iters) if attack_type == 'BIM' else (2.0 * eps / iters), 1e-4)

    probs_adv_list = []
    num_samples = len(ego_graphs)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_start = time.perf_counter()

    for i in range(0, num_samples, batch_size):
        batch_chunk = [g.clone() for g in ego_graphs[i:i+batch_size]]
        batched_data = Batch.from_data_list(batch_chunk).to(device)
        x_file_orig = batched_data['file'].x.clone().detach().to(device)
        y_true = batched_data['file'].y.clone().detach().to(device)

        with torch.enable_grad():
            if attack_type == 'FGSM':
                x_adv = x_file_orig.clone().detach().requires_grad_(True)
                x_dict = {k: v for k, v in batched_data.x_dict.items()}
                x_dict['file'] = x_adv

                out = model(x_dict, batched_data.edge_index_dict)
                loss = F.cross_entropy(out, y_true)
                grad = torch.autograd.grad(loss, x_adv)[0]
                x_adv = x_file_orig + eps * grad.sign()
                x_dict['file'] = x_adv.detach()
                out_final = model(x_dict, batched_data.edge_index_dict)
                probs = F.softmax(out_final, dim=1).detach().cpu().numpy()
                probs_adv_list.append(probs)

            elif attack_type == 'BIM':
                x_adv = x_file_orig.clone().detach()
                for _ in range(iters):
                    x_adv.requires_grad_(True)
                    x_dict = {k: v for k, v in batched_data.x_dict.items()}
                    x_dict['file'] = x_adv

                    out = model(x_dict, batched_data.edge_index_dict)
                    loss = F.cross_entropy(out, y_true)
                    grad = torch.autograd.grad(loss, x_adv)[0]
                    x_adv = x_adv.detach() + alpha * grad.sign()
                    eta = torch.clamp(x_adv - x_file_orig, min=-eps, max=eps)
                    x_adv = (x_file_orig + eta).detach()

                x_dict = {k: v for k, v in batched_data.x_dict.items()}
                x_dict['file'] = x_adv
                out_final = model(x_dict, batched_data.edge_index_dict)
                probs = F.softmax(out_final, dim=1).detach().cpu().numpy()
                probs_adv_list.append(probs)

            elif attack_type == 'PGD':
                best_loss = torch.zeros(x_file_orig.size(0), device=device) - 1e9
                best_x_adv = x_file_orig.clone().detach()

                for restart in range(max(1, num_restarts)):
                    noise = torch.FloatTensor(x_file_orig.shape).uniform_(-eps, eps).to(device)
                    x_adv_curr = (x_file_orig + noise).clone().detach()

                    for _ in range(iters):
                        x_adv_curr.requires_grad_(True)
                        x_dict = {k: v for k, v in batched_data.x_dict.items()}
                        x_dict['file'] = x_adv_curr

                        out = model(x_dict, batched_data.edge_index_dict)
                        loss = F.cross_entropy(out, y_true)
                        grad = torch.autograd.grad(loss, x_adv_curr)[0]
                        x_adv_curr = x_adv_curr.detach() + alpha * grad.sign()
                        eta = torch.clamp(x_adv_curr - x_file_orig, min=-eps, max=eps)
                        x_adv_curr = (x_file_orig + eta).detach()

                    x_dict = {k: v for k, v in batched_data.x_dict.items()}
                    x_dict['file'] = x_adv_curr
                    out_curr = model(x_dict, batched_data.edge_index_dict)
                    loss_sample = F.cross_entropy(out_curr, y_true, reduction='none')

                    better_mask = (loss_sample > best_loss).unsqueeze(-1)
                    best_loss = torch.where(loss_sample > best_loss, loss_sample, best_loss)
                    best_x_adv = torch.where(better_mask, x_adv_curr, best_x_adv)

                x_dict = {k: v for k, v in batched_data.x_dict.items()}
                x_dict['file'] = best_x_adv
                out_final = model(x_dict, batched_data.edge_index_dict)
                probs = F.softmax(out_final, dim=1).detach().cpu().numpy()
                probs_adv_list.append(probs)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_end = time.perf_counter()

    infer_ms_sample = ((t_end - t_start) / max(num_samples, 1)) * 1000.0
    return np.concatenate(probs_adv_list, axis=0), infer_ms_sample

def attack_homo_gnn_model_rectified(model, graph, attack_type='PGD', eps=0.1, alpha=None, iters=20, num_restarts=5):
    set_eval_mode_safe(model)
    if alpha is None:
        alpha = max((eps / iters) if attack_type == 'BIM' else (2.0 * eps / iters), 1e-4)

    graph_adv = graph.clone().to(device)
    x_orig = graph.x.clone().detach().to(device)
    y_true = graph.y.clone().detach().to(device)

    with torch.enable_grad():
        if attack_type == 'FGSM':
            x_adv = x_orig.clone().detach().requires_grad_(True)
            out = model(x_adv, graph_adv.edge_index)
            loss = F.cross_entropy(out, y_true)
            grad = torch.autograd.grad(loss, x_adv)[0]
            x_adv = x_orig + eps * grad.sign()
            graph_adv.x = x_adv.detach()
            model.eval()
            return graph_adv

        elif attack_type == 'BIM':
            x_adv = x_orig.clone().detach()
            for _ in range(iters):
                x_adv.requires_grad_(True)
                out = model(x_adv, graph_adv.edge_index)
                loss = F.cross_entropy(out, y_true)
                grad = torch.autograd.grad(loss, x_adv)[0]

                x_adv = x_adv.detach() + alpha * grad.sign()
                eta = torch.clamp(x_adv - x_orig, min=-eps, max=eps)
                x_adv = (x_orig + eta).detach()
            graph_adv.x = x_adv.detach()
            model.eval()
            return graph_adv

        elif attack_type == 'PGD':
            best_loss = torch.zeros(x_orig.size(0), device=device) - 1e9
            best_x_adv = x_orig.clone().detach()

            for restart in range(max(1, num_restarts)):
                noise = torch.FloatTensor(x_orig.shape).uniform_(-eps, eps).to(device)
                x_adv_curr = (x_orig + noise).clone().detach()

                for _ in range(iters):
                    x_adv_curr.requires_grad_(True)
                    out = model(x_adv_curr, graph_adv.edge_index)
                    loss = F.cross_entropy(out, y_true)
                    grad = torch.autograd.grad(loss, x_adv_curr)[0]

                    x_adv_curr = x_adv_curr.detach() + alpha * grad.sign()
                    eta = torch.clamp(x_adv_curr - x_orig, min=-eps, max=eps)
                    x_adv_curr = (x_orig + eta).detach()

                out_final = model(x_adv_curr, graph_adv.edge_index)
                loss_per_sample = F.cross_entropy(out_final, y_true, reduction='none')

                better_mask = (loss_per_sample > best_loss).unsqueeze(-1)
                best_loss = torch.where(loss_per_sample > best_loss, loss_per_sample, best_loss)
                best_x_adv = torch.where(better_mask, x_adv_curr, best_x_adv)

            graph_adv.x = best_x_adv.detach()
            model.eval()
            return graph_adv

    model.eval()
    return graph_adv

# -----------------------------------------------------------------------------------------
# PREDICTION HELPERS WITH SYNCHRONIZED CUDA TIMING
# -----------------------------------------------------------------------------------------
def predict_tensor_model(model, bx, batch_size=1024):
    if hasattr(model, 'eval'):
        model.eval()

    if isinstance(bx, np.ndarray):
        bx = torch.tensor(bx, dtype=torch.float32)
    bx = bx.to(device)

    probs_list = []
    num_samples = bx.shape[0]

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_start = time.perf_counter()

    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            chunk = bx[i:i+batch_size]
            out = model(chunk)
            if isinstance(out, tuple): out = out[0]
            probs_list.append(F.softmax(out, dim=1).cpu().numpy())

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_end = time.perf_counter()

    infer_ms_sample = ((t_end - t_start) / max(num_samples, 1)) * 1000.0
    return np.concatenate(probs_list, axis=0), infer_ms_sample

def predict_fusion_model(model, bx_p, bx_s, bx_st, batch_size=1024):
    model.eval()

    if isinstance(bx_p, np.ndarray): bx_p = torch.tensor(bx_p, dtype=torch.float32)
    if isinstance(bx_s, np.ndarray): bx_s = torch.tensor(bx_s, dtype=torch.float32)
    if isinstance(bx_st, np.ndarray): bx_st = torch.tensor(bx_st, dtype=torch.float32)

    bx_p = bx_p.to(device)
    bx_s = bx_s.to(device)
    bx_st = bx_st.to(device)

    probs_list = []
    num_samples = bx_p.shape[0]

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_start = time.perf_counter()

    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            bp = bx_p[i:i+batch_size]
            bs = bx_s[i:i+batch_size]
            bst = bx_st[i:i+batch_size]
            logits, _, _ = model(bp, bs, bst)
            probs_list.append(F.softmax(logits, dim=1).cpu().numpy())

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_end = time.perf_counter()

    infer_ms_sample = ((t_end - t_start) / max(num_samples, 1)) * 1000.0
    return np.concatenate(probs_list, axis=0), infer_ms_sample
class PCAP_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, out_dim=2):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        self.classifier = nn.Linear(hidden_dim // 2, out_dim)

    def forward(self, x):
        feat = self.feature_extractor(x)
        logits = self.classifier(feat)
        return logits, feat

class SAR_Autoencoder(nn.Module):
    """Historical class name; implementation is an encoder-classifier (no decoder).

    Results are labeled 'SAR Encoder' to avoid falsely claiming reconstruction-based autoencoder training.
    """
    def __init__(self, in_dim, latent_dim=64, out_dim=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
            nn.ReLU()
        )
        self.classifier = nn.Linear(latent_dim, out_dim)

    def forward(self, x):
        latent = self.encoder(x)
        logits = self.classifier(latent)
        return logits, latent

class STRACE_CNN_LSTM(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, out_dim=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.lstm = nn.LSTM(input_size=32, hidden_size=hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, out_dim)

    def forward(self, x):
        c = x.unsqueeze(1)
        c = self.conv(c)
        c = c.permute(0, 2, 1)
        out_lstm, _ = self.lstm(c)
        feat = out_lstm[:, -1, :]
        logits = self.fc(feat)
        return logits, feat

class Multimodal_NonHIN_GNN(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, out_dim=2):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.lin_res = nn.Linear(in_dim, hidden_dim)
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(hidden_dim, out_dim))

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = self.ln1(h + self.lin_res(x))
        h1 = F.relu(h)
        h2 = self.conv2(h1, edge_index)
        h2 = self.ln2(h2 + h1)
        logits = self.classifier(F.relu(h2))
        return logits

def train_neural_branch(model, x_tr, y_tr, epochs=30, batch_size=256, lr=0.001):
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t_start = time.perf_counter()

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    ds_tr = TensorDataset(torch.tensor(x_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long))
    # [v2 / FIX C6] drop_last: a final batch of size 1 makes BatchNorm1d raise an error during training.
    loader_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, drop_last=True)

    for epoch in range(epochs):
        model.train()
        for bx, by in loader_tr:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            if isinstance(out, tuple): out = out[0]
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_end = time.perf_counter()

    train_time_sec = t_end - t_start
    peak_gpu_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0

    model.eval()
    return model, train_time_sec, peak_gpu_mem_mb

class MultimodalLateFusionModel(nn.Module):
    def __init__(self, pcap_dim, sar_dim, strace_dim, fused_dim=64, out_dim=2, drop_rate=0.3):
        super().__init__()
        self.pcap_branch = PCAP_MLP(in_dim=pcap_dim, hidden_dim=64, out_dim=out_dim)
        self.sar_branch = SAR_Autoencoder(in_dim=sar_dim, latent_dim=64, out_dim=out_dim)
        self.strace_branch = STRACE_CNN_LSTM(in_dim=strace_dim, hidden_dim=64, out_dim=out_dim)

        self.proj_pcap = nn.Linear(32, fused_dim)
        self.proj_sar = nn.Linear(64, fused_dim)
        self.proj_strace = nn.Linear(128, fused_dim)

        self.attn_net = nn.Sequential(nn.Linear(fused_dim * 3, 64), nn.ReLU(), nn.Linear(64, 3))
        self.classifier = nn.Sequential(nn.Linear(fused_dim, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(drop_rate), nn.Linear(64, out_dim))

    def load_pretrained_branches(self, trained_pcap, trained_sar, trained_strace):
        if trained_pcap is not None: self.pcap_branch.load_state_dict(trained_pcap.state_dict())
        if trained_sar is not None: self.sar_branch.load_state_dict(trained_sar.state_dict())
        if trained_strace is not None: self.strace_branch.load_state_dict(trained_strace.state_dict())

    def forward(self, x_pcap, x_sar, x_strace):
        _, feat_p = self.pcap_branch(x_pcap)
        _, feat_s = self.sar_branch(x_sar)
        _, feat_st = self.strace_branch(x_strace)

        h_p = F.relu(self.proj_pcap(feat_p))
        h_s = F.relu(self.proj_sar(feat_s))
        h_st = F.relu(self.proj_strace(feat_st))

        concat_h = torch.cat([h_p, h_s, h_st], dim=-1)
        attn_weights = F.softmax(self.attn_net(concat_h), dim=-1)

        fused_h = (
            attn_weights[:, 0:1] * h_p +
            attn_weights[:, 1:2] * h_s +
            attn_weights[:, 2:3] * h_st
        )

        logits_fused = self.classifier(fused_h)
        return logits_fused, (h_p, h_s, h_st), attn_weights

def train_end_to_end_joint_fine_tuning(fusion_model, x_tr_pcap, x_tr_sar, x_tr_sys, y_tr, epochs=25, batch_size=256):
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t_start = time.perf_counter()

    fusion_model = fusion_model.to(device)
    pretrained_params = list(fusion_model.pcap_branch.parameters()) + list(fusion_model.sar_branch.parameters()) + list(fusion_model.strace_branch.parameters())
    fusion_head_params = list(fusion_model.proj_pcap.parameters()) + list(fusion_model.proj_sar.parameters()) + list(fusion_model.proj_strace.parameters()) + list(fusion_model.attn_net.parameters()) + list(fusion_model.classifier.parameters())

    optimizer = torch.optim.Adam([{'params': pretrained_params, 'lr': 1e-4}, {'params': fusion_head_params, 'lr': 1e-3}], weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    ds = TensorDataset(torch.tensor(x_tr_pcap, dtype=torch.float32), torch.tensor(x_tr_sar, dtype=torch.float32), torch.tensor(x_tr_sys, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long))
    # [v2 / FIX C6] see the note in train_neural_branch.
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    fusion_model.train()
    for epoch in range(epochs):
        for bp, bs, bst, by in loader:
            bp, bs, bst, by = bp.to(device), bs.to(device), bst.to(device), by.to(device)
            optimizer.zero_grad()
            logits_fused, _, _ = fusion_model(bp, bs, bst)
            total_loss = criterion(logits_fused, by)
            total_loss.backward()
            optimizer.step()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_end = time.perf_counter()

    train_time_sec = t_end - t_start
    peak_gpu_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0

    fusion_model.eval()
    return fusion_model, train_time_sec, peak_gpu_mem_mb


# =========================================================================================
# 8. ALL 13 HIN GNNs & 6-STAGE R-HGT ABLATION SEQUENCE ARCHITECTURES
# =========================================================================================
class HIN_Vanilla_RGCN(nn.Module):
    def __init__(self, hidden, out, metadata):
        super().__init__()
        node_types, edge_types = metadata[0], metadata[1]
        self.node_types = node_types
        self.edge_types = edge_types

        self.lin_dict = nn.ModuleDict({node: Linear(-1, hidden) for node in node_types})
        self.rel_conv1 = nn.ModuleDict({f"{et[0]}_{et[1]}_{et[2]}": Linear(hidden, hidden, bias=False) for et in edge_types})
        self.self_conv1 = nn.ModuleDict({node: Linear(hidden, hidden) for node in node_types})
        self.rel_conv2 = nn.ModuleDict({f"{et[0]}_{et[1]}_{et[2]}": Linear(hidden, hidden, bias=False) for et in edge_types})
        self.self_conv2 = nn.ModuleDict({node: Linear(hidden, hidden) for node in node_types})

        self.drop = nn.Dropout(0.2)
        self.lin = Linear(hidden, out)

    def forward(self, x_dict, edge_index_dict):
        x = {node: F.relu(self.lin_dict[node](feat)) for node, feat in x_dict.items() if node in self.lin_dict}
        out1 = {node: self.self_conv1[node](x[node]) for node in x.keys() if node in self.self_conv1}
        for et in self.edge_types:
            src_type, rel_name, dst_type = et
            edge_index = edge_index_dict.get(et, torch.empty((2, 0), dtype=torch.long, device=x[src_type].device))
            if edge_index.numel() == 0 or src_type not in x or dst_type not in out1: continue
            key = f"{src_type}_{rel_name}_{dst_type}"
            if key not in self.rel_conv1: continue
            src_x = x[src_type]
            h_trans = self.rel_conv1[key](src_x)
            src_idx, dst_idx = edge_index[0], edge_index[1]
            num_dst = x[dst_type].size(0)
            msg = torch.zeros((num_dst, h_trans.size(1)), device=src_x.device, dtype=src_x.dtype)
            msg.index_add_(0, dst_idx, h_trans[src_idx])
            deg = torch.zeros(num_dst, device=src_x.device, dtype=src_x.dtype)
            deg.index_add_(0, dst_idx, torch.ones_like(dst_idx, dtype=src_x.dtype))
            deg = deg.clamp(min=1).unsqueeze(-1)
            out1[dst_type] = out1[dst_type] + (msg / deg)

        x1 = {node: F.relu(self.drop(out1[node])) for node in out1.keys()}
        out2 = {node: self.self_conv2[node](x1[node]) for node in x1.keys() if node in self.self_conv2}
        for et in self.edge_types:
            src_type, rel_name, dst_type = et
            edge_index = edge_index_dict.get(et, torch.empty((2, 0), dtype=torch.long, device=x1[src_type].device))
            if edge_index.numel() == 0 or src_type not in x1 or dst_type not in out2: continue
            key = f"{src_type}_{rel_name}_{dst_type}"
            if key not in self.rel_conv2: continue
            src_x = x1[src_type]
            h_trans = self.rel_conv2[key](src_x)
            src_idx, dst_idx = edge_index[0], edge_index[1]
            num_dst = x1[dst_type].size(0)
            msg = torch.zeros((num_dst, h_trans.size(1)), device=src_x.device, dtype=src_x.dtype)
            msg.index_add_(0, dst_idx, h_trans[src_idx])
            deg = torch.zeros(num_dst, device=src_x.device, dtype=src_x.dtype)
            deg.index_add_(0, dst_idx, torch.ones_like(dst_idx, dtype=src_x.dtype))
            deg = deg.clamp(min=1).unsqueeze(-1)
            out2[dst_type] = out2[dst_type] + (msg / deg)

        x2 = {node: F.relu(self.drop(out2[node])) for node in out2.keys()}
        return self.lin(x2['file'])
class HIN_R2_RGCN(nn.Module):
    """R²-RGCN - NOTE ON HOW IT IS PRESENTED IN THE PAPER.

    This class is NOT an independent third-party baseline. It reuses VERBATIM the
    three core mechanisms of R-HGT:

        • node-level reliability : 1.0 - 0.5 * unc_net(h)      (same as Stage 3/4)
        • residual gate          : gate*h + (1-gate)*drop(m)    (same as Stage 4)
        • per-node-type LayerNorm

    The only difference from R-HGT Stage 4 is the message-passing CORE (degree-normalised mean
    aggregation instead of HGT attention) and the way relation weights are normalised
    (softmax β instead of sigmoid ρ).

    In other words, this is a CORE-REPLACEMENT ABLATION of the proposed model itself.
    Presenting it as an external baseline would be inaccurate - but placed correctly,
    it is the STRONGEST evidence for the contribution: the same three mechanisms with two
    different cores separate the contribution of the mechanisms from that of the HGT core.

    In the paper it is reported as the residual relational baseline "R2-RGCN",
    implemented in this code base rather than taken from an external publication.
    """
    def __init__(self, hidden, out, metadata):
        super().__init__()
        node_types, edge_types = metadata[0], metadata[1]
        self.node_types = node_types
        self.edge_types = edge_types

        self.lin_dict = nn.ModuleDict({node: Linear(-1, hidden) for node in node_types})
        self.rel_conv1 = nn.ModuleDict({f"{et[0]}_{et[1]}_{et[2]}": Linear(hidden, hidden, bias=False) for et in edge_types})
        self.self_conv1 = nn.ModuleDict({node: Linear(hidden, hidden) for node in node_types})
        self.rel_conv2 = nn.ModuleDict({f"{et[0]}_{et[1]}_{et[2]}": Linear(hidden, hidden, bias=False) for et in edge_types})
        self.self_conv2 = nn.ModuleDict({node: Linear(hidden, hidden) for node in node_types})

        self.rel_attn1 = nn.ParameterDict({f"{et[0]}_{et[1]}_{et[2]}": nn.Parameter(torch.ones(1)) for et in edge_types})
        self.rel_attn2 = nn.ParameterDict({f"{et[0]}_{et[1]}_{et[2]}": nn.Parameter(torch.ones(1)) for et in edge_types})

        self.unc_net1 = nn.ModuleDict({node: nn.Sequential(Linear(hidden, hidden // 2), nn.ReLU(), Linear(hidden // 2, 1), nn.Sigmoid()) for node in node_types})
        self.unc_net2 = nn.ModuleDict({node: nn.Sequential(Linear(hidden, hidden // 2), nn.ReLU(), Linear(hidden // 2, 1), nn.Sigmoid()) for node in node_types})

        self.gate_dict1 = nn.ModuleDict({node: nn.Sequential(Linear(hidden * 2, hidden), nn.Sigmoid()) for node in node_types})
        self.gate_dict2 = nn.ModuleDict({node: nn.Sequential(Linear(hidden * 2, hidden), nn.Sigmoid()) for node in node_types})

        self.ln_dict1 = nn.ModuleDict({node: nn.LayerNorm(hidden) for node in node_types})
        self.ln_dict2 = nn.ModuleDict({node: nn.LayerNorm(hidden) for node in node_types})

        self.drop = nn.Dropout(0.2)
        self.lin = Linear(hidden, out)

    def forward(self, x_dict, edge_index_dict):
        x = {node: F.relu(self.lin_dict[node](feat)) for node, feat in x_dict.items() if node in self.lin_dict}

        node_weights1 = {node: 1.0 - 0.5 * self.unc_net1[node](x[node]) for node in x.keys() if node in self.unc_net1}
        out1 = {node: self.self_conv1[node](x[node]) for node in x.keys() if node in self.self_conv1}

        attn_scores1 = torch.stack([self.rel_attn1[f"{et[0]}_{et[1]}_{et[2]}"] for et in self.edge_types if f"{et[0]}_{et[1]}_{et[2]}" in self.rel_attn1])
        attn_weights1 = F.softmax(attn_scores1, dim=0)

        for idx, et in enumerate(self.edge_types):
            src_type, rel_name, dst_type = et
            edge_index = edge_index_dict.get(et, torch.empty((2, 0), dtype=torch.long, device=x[src_type].device))
            if edge_index.numel() == 0 or src_type not in x or dst_type not in out1: continue
            key = f"{src_type}_{rel_name}_{dst_type}"
            if key not in self.rel_conv1: continue
            src_x = x[src_type]
            src_conf = node_weights1.get(src_type, torch.ones((src_x.size(0), 1), device=src_x.device))

            h_trans = self.rel_conv1[key](src_x * src_conf) * (attn_weights1[idx] if idx < len(attn_weights1) else 1.0)
            src_idx, dst_idx = edge_index[0], edge_index[1]
            num_dst = x[dst_type].size(0)

            msg = torch.zeros((num_dst, h_trans.size(1)), device=src_x.device, dtype=src_x.dtype)
            msg.index_add_(0, dst_idx, h_trans[src_idx])
            deg = torch.zeros(num_dst, device=src_x.device, dtype=src_x.dtype)
            deg.index_add_(0, dst_idx, torch.ones_like(dst_idx, dtype=src_x.dtype))
            deg = deg.clamp(min=1).unsqueeze(-1)
            out1[dst_type] = out1[dst_type] + (msg / deg)

        x1 = {}
        for node in out1.keys():
            if node in self.gate_dict1:
                gate = self.gate_dict1[node](torch.cat([x[node], out1[node]], dim=-1))
                fused = gate * x[node] + (1.0 - gate) * self.drop(out1[node])
                x1[node] = F.relu(self.ln_dict1[node](fused))
            else:
                x1[node] = F.relu(out1[node])

        node_weights2 = {node: 1.0 - 0.5 * self.unc_net2[node](x1[node]) for node in x1.keys() if node in self.unc_net2}
        out2 = {node: self.self_conv2[node](x1[node]) for node in x1.keys() if node in self.self_conv2}

        attn_scores2 = torch.stack([self.rel_attn2[f"{et[0]}_{et[1]}_{et[2]}"] for et in self.edge_types if f"{et[0]}_{et[1]}_{et[2]}" in self.rel_attn2])
        attn_weights2 = F.softmax(attn_scores2, dim=0)

        for idx, et in enumerate(self.edge_types):
            src_type, rel_name, dst_type = et
            edge_index = edge_index_dict.get(et, torch.empty((2, 0), dtype=torch.long, device=x1[src_type].device))
            if edge_index.numel() == 0 or src_type not in x1 or dst_type not in out2: continue
            key = f"{src_type}_{rel_name}_{dst_type}"
            if key not in self.rel_conv2: continue
            src_x = x1[src_type]
            src_conf = node_weights2.get(src_type, torch.ones((src_x.size(0), 1), device=src_x.device))

            h_trans = self.rel_conv2[key](src_x * src_conf) * (attn_weights2[idx] if idx < len(attn_weights2) else 1.0)
            src_idx, dst_idx = edge_index[0], edge_index[1]
            num_dst = x1[dst_type].size(0)

            msg = torch.zeros((num_dst, h_trans.size(1)), device=src_x.device, dtype=src_x.dtype)
            msg.index_add_(0, dst_idx, h_trans[src_idx])
            deg = torch.zeros(num_dst, device=src_x.device, dtype=src_x.dtype)
            deg.index_add_(0, dst_idx, torch.ones_like(dst_idx, dtype=src_x.dtype))
            deg = deg.clamp(min=1).unsqueeze(-1)
            out2[dst_type] = out2[dst_type] + (msg / deg)

        x2 = {}
        for node in out2.keys():
            if node in self.gate_dict2:
                gate = self.gate_dict2[node](torch.cat([x1[node], out2[node]], dim=-1))
                fused = gate * x1[node] + (1.0 - gate) * self.drop(out2[node])
                x2[node] = F.relu(self.ln_dict2[node](fused))
            else:
                x2[node] = F.relu(out2[node])

        return self.lin(x2['file'])

class HIN_MAGNN(nn.Module):
    def __init__(self, hidden, out, metadata, attn_dim=32):
        super().__init__()
        node_types = metadata[0]
        self.lin_dict = nn.ModuleDict({node: Linear(-1, hidden) for node in node_types})
        self.mp_sys_f2s = Linear(hidden, hidden)
        self.mp_sys_s2f = Linear(hidden, hidden)
        self.mp_sys_self = Linear(hidden, hidden)
        self.mp_arch_f2a = Linear(hidden, hidden)
        self.mp_arch_a2f = Linear(hidden, hidden)
        self.mp_arch_self = Linear(hidden, hidden)
        self.mp_proto_f2p = Linear(hidden, hidden)
        self.mp_proto_p2f = Linear(hidden, hidden)
        self.mp_proto_self = Linear(hidden, hidden)

        self.w_attn = Linear(hidden, attn_dim)
        self.q_attn = nn.Parameter(torch.randn(attn_dim, 1))
        self.ln_file = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(0.2)
        self.lin = Linear(hidden, out)

    def _aggregate_bipartite(self, x_src, edge_index, num_dst, proj_layer):
        if edge_index.numel() == 0:
            return torch.zeros((num_dst, x_src.size(1)), device=x_src.device, dtype=x_src.dtype)
        src_idx, dst_idx = edge_index[0], edge_index[1]
        h_src = proj_layer(x_src)
        out = torch.zeros((num_dst, h_src.size(1)), device=x_src.device, dtype=x_src.dtype)
        out.index_add_(0, dst_idx, h_src[src_idx])
        deg = torch.zeros(num_dst, device=x_src.device, dtype=x_src.dtype)
        deg.index_add_(0, dst_idx, torch.ones_like(dst_idx, dtype=x_src.dtype))
        deg = deg.clamp(min=1).unsqueeze(-1)
        return out / deg

    def forward(self, x_dict, edge_index_dict):
        x = {node: F.relu(self.lin_dict[node](x_dict[node])) for node in x_dict.keys() if node in self.lin_dict}
        h_file_target = x['file']
        num_files = h_file_target.size(0)
        metapath_embeddings = []

        edge_f_s = edge_index_dict.get(('file', 'executes', 'syscall'), torch.empty((2, 0), dtype=torch.long, device=h_file_target.device))
        if edge_f_s.numel() > 0:
            num_sys = x['syscall'].size(0)
            h_sys = F.relu(self._aggregate_bipartite(h_file_target, edge_f_s, num_sys, self.mp_sys_f2s))
            edge_s_f = edge_f_s[[1, 0]]
            h_file_sys = F.relu(self._aggregate_bipartite(h_sys, edge_s_f, num_files, self.mp_sys_s2f) + self.mp_sys_self(h_file_target))
        else:
            h_file_sys = F.relu(self.mp_sys_self(h_file_target))
        metapath_embeddings.append(h_file_sys)

        edge_f_a = edge_index_dict.get(('file', 'runs_on', 'arch'), torch.empty((2, 0), dtype=torch.long, device=h_file_target.device))
        if edge_f_a.numel() > 0:
            num_arch = x['arch'].size(0)
            h_arch = F.relu(self._aggregate_bipartite(h_file_target, edge_f_a, num_arch, self.mp_arch_f2a))
            edge_a_f = edge_f_a[[1, 0]]
            h_file_arch = F.relu(self._aggregate_bipartite(h_arch, edge_a_f, num_files, self.mp_arch_a2f) + self.mp_arch_self(h_file_target))
        else:
            h_file_arch = F.relu(self.mp_arch_self(h_file_target))
        metapath_embeddings.append(h_file_arch)

        if 'protocol' in x:
            edge_f_p = edge_index_dict.get(('file', 'uses_proto', 'protocol'), torch.empty((2, 0), dtype=torch.long, device=h_file_target.device))
            if edge_f_p.numel() > 0:
                num_proto = x['protocol'].size(0)
                h_proto = F.relu(self._aggregate_bipartite(h_file_target, edge_f_p, num_proto, self.mp_proto_f2p))
                edge_p_f = edge_f_p[[1, 0]]
                h_file_proto = F.relu(self._aggregate_bipartite(h_proto, edge_p_f, num_files, self.mp_proto_p2f) + self.mp_proto_self(h_file_target))
            else:
                h_file_proto = F.relu(self.mp_proto_self(h_file_target))
            metapath_embeddings.append(h_file_proto)

        # [v3 / FIX C3] The original took the MEAN over the node dimension (w.mean(dim=1))
        # before the softmax, so the metapath weights depended on the WHOLE BATCH. In strict
        # mode a batch consists of 256 independent ego-graphs, so the prediction for sample i
        # depended on 255 other test samples - violating the isolation property that the
        # paper claims, and making results change with batch_size.
        # Fix: softmax over the metapath dimension, independently for each node.
        M = len(metapath_embeddings)
        stacked_h = torch.stack(metapath_embeddings, dim=0)                  # (M, N, d)
        attn_proj = torch.tanh(self.w_attn(stacked_h))                       # (M, N, d_a)
        w = torch.matmul(attn_proj, self.q_attn).squeeze(-1) / np.sqrt(32)   # (M, N)
        beta = F.softmax(w, dim=0)                                           # (M, N)
        h_file_aggregated = (beta.unsqueeze(-1) * stacked_h).sum(dim=0)      # (N, d)

        h_file_final = self.ln_file(h_file_target + self.drop(h_file_aggregated))
        return self.lin(h_file_final)
# [v3] The class HIN_Robust_HGT HAS BEEN REMOVED.
#
# Checked by reference count: 0 uses in the whole pipeline. It was an earlier
# version of R-HGT (with node reliability + residual gate, but without the relation
# reliability aggregator) that was never wired into hin_model_specs. Leaving it in
# the public repo would only make readers think it is an architecture used in the paper.
#
# The equivalent architectures actually in use: HIN_HGT_Stage3_NodeReliability
# (no residual gate) and HIN_HGT_Stage4_RelationGating (residual gate + relation).


class HIN_HGT_Stage1_Vanilla(nn.Module):
    def __init__(self, hidden, out, heads, layers, metadata):
        super().__init__()
        self.node_types = metadata[0]
        self.edge_types = metadata[1]
        self.layers = layers

        self.lin_dict = nn.ModuleDict({node: Linear(-1, hidden) for node in self.node_types})
        self.convs = nn.ModuleList([HGTConv(hidden, hidden, metadata, heads) for _ in range(layers)])
        self.drop = nn.Dropout(0.2)
        self.classifier = Linear(hidden, out)

    def forward(self, x_dict, edge_index_dict):
        x = {node: F.relu(self.lin_dict[node](feat)) for node, feat in x_dict.items() if node in self.lin_dict}
        for i in range(self.layers):
            x = self.convs[i](x, edge_index_dict)
            x = {node: F.relu(self.drop(h)) for node, h in x.items()}
        return self.classifier(x['file'])

class HIN_HGT_Stage2_ResLN(nn.Module):
    def __init__(self, hidden, out, heads, layers, metadata):
        super().__init__()
        self.node_types = metadata[0]
        self.edge_types = metadata[1]
        self.layers = layers

        self.lin_dict = nn.ModuleDict({node: Linear(-1, hidden) for node in self.node_types})
        self.convs = nn.ModuleList([HGTConv(hidden, hidden, metadata, heads) for _ in range(layers)])
        self.ln_dict = nn.ModuleList([
            nn.ModuleDict({node: nn.LayerNorm(hidden) for node in self.node_types})
            for _ in range(layers)
        ])
        self.drop = nn.Dropout(0.2)
        self.classifier = Linear(hidden, out)

    def forward(self, x_dict, edge_index_dict):
        x = {node: F.relu(self.lin_dict[node](feat)) for node, feat in x_dict.items() if node in self.lin_dict}
        for i in range(self.layers):
            x_next = self.convs[i](x, edge_index_dict)
            x = {
                node: F.relu(self.ln_dict[i][node](x[node] + self.drop(x_next.get(node, x[node]))))
                for node in x.keys()
            }
        return self.classifier(x['file'])

class HIN_HGT_Stage3_NodeReliability(nn.Module):
    def __init__(self, hidden, out, heads, layers, metadata):
        super().__init__()
        self.node_types = metadata[0]
        self.edge_types = metadata[1]
        self.layers = layers

        self.lin_dict = nn.ModuleDict({node: Linear(-1, hidden) for node in self.node_types})
        self.unc_nets = nn.ModuleList([
            nn.ModuleDict({
                node: nn.Sequential(Linear(hidden, hidden // 2), nn.ReLU(), Linear(hidden // 2, 1), nn.Sigmoid())
                for node in self.node_types
            }) for _ in range(layers)
        ])
        self.convs = nn.ModuleList([HGTConv(hidden, hidden, metadata, heads) for _ in range(layers)])
        self.ln_dict = nn.ModuleList([
            nn.ModuleDict({node: nn.LayerNorm(hidden) for node in self.node_types})
            for _ in range(layers)
        ])
        self.drop = nn.Dropout(0.2)
        self.classifier = Linear(hidden, out)

    def forward(self, x_dict, edge_index_dict):
        x = {node: F.relu(self.lin_dict[node](feat)) for node, feat in x_dict.items() if node in self.lin_dict}
        for i in range(self.layers):
            confidence_dict = {node: 1.0 - 0.5 * self.unc_nets[i][node](h) for node, h in x.items() if node in self.unc_nets[i]}
            x_weighted = {node: x[node] * confidence_dict.get(node, 1.0) for node in x.keys()}
            x_next = self.convs[i](x_weighted, edge_index_dict)
            x = {
                node: F.relu(self.ln_dict[i][node](x[node] + self.drop(x_next.get(node, x[node]))))
                for node in x.keys()
            }
        return self.classifier(x['file'])

class RelationReliabilityAggregator(nn.Module):
    """Differentiable per-relation reliability aggregation used by Stage 4+.

    Each edge type has its own transform and learned sigmoid reliability scalar.
    The resulting relation context is added to the HGT message before the node-level
    residual gate, making the "relation reliability" claim explicit in code.
    """
    def __init__(self, hidden, metadata):
        super().__init__()
        self.edge_types = list(metadata[1])
        self.rel_transforms = nn.ModuleDict({
            f"{src}__{rel}__{dst}": Linear(hidden, hidden, bias=False)
            for src, rel, dst in self.edge_types
        })
        self.rel_logits = nn.ParameterDict({
            f"{src}__{rel}__{dst}": nn.Parameter(torch.zeros(1))
            for src, rel, dst in self.edge_types
        })

    def forward(self, x_dict, edge_index_dict):
        out = {node: torch.zeros_like(h) for node, h in x_dict.items()}
        for src_type, rel_name, dst_type in self.edge_types:
            if src_type not in x_dict or dst_type not in x_dict:
                continue
            et = (src_type, rel_name, dst_type)
            edge_index = edge_index_dict.get(et)
            if edge_index is None or edge_index.numel() == 0:
                continue

            key = f"{src_type}__{rel_name}__{dst_type}"
            src_idx, dst_idx = edge_index[0], edge_index[1]
            h_src = self.rel_transforms[key](x_dict[src_type])
            num_dst = x_dict[dst_type].size(0)

            msg = torch.zeros((num_dst, h_src.size(-1)), device=h_src.device, dtype=h_src.dtype)
            msg.index_add_(0, dst_idx, h_src[src_idx])
            deg = torch.zeros(num_dst, device=h_src.device, dtype=h_src.dtype)
            deg.index_add_(0, dst_idx, torch.ones_like(dst_idx, dtype=h_src.dtype))
            msg = msg / deg.clamp(min=1).unsqueeze(-1)

            reliability = torch.sigmoid(self.rel_logits[key])
            out[dst_type] = out[dst_type] + reliability * msg
        return out

class HIN_HGT_Stage4_RelationGating(nn.Module):
    def __init__(self, hidden, out, heads, layers, metadata):
        super().__init__()
        self.node_types = metadata[0]
        self.edge_types = metadata[1]
        self.layers = layers

        self.lin_dict = nn.ModuleDict({node: Linear(-1, hidden) for node in self.node_types})
        self.unc_nets = nn.ModuleList([
            nn.ModuleDict({
                node: nn.Sequential(Linear(hidden, hidden // 2), nn.ReLU(), Linear(hidden // 2, 1), nn.Sigmoid())
                for node in self.node_types
            }) for _ in range(layers)
        ])
        self.convs = nn.ModuleList([HGTConv(hidden, hidden, metadata, heads) for _ in range(layers)])
        self.rel_aggs = nn.ModuleList([RelationReliabilityAggregator(hidden, metadata) for _ in range(layers)])
        self.gates = nn.ModuleList([
            nn.ModuleDict({
                node: nn.Sequential(Linear(hidden * 2, hidden), nn.Sigmoid())
                for node in self.node_types
            }) for _ in range(layers)
        ])
        self.ln_dict = nn.ModuleList([
            nn.ModuleDict({node: nn.LayerNorm(hidden) for node in self.node_types})
            for _ in range(layers)
        ])
        self.drop = nn.Dropout(0.2)
        self.classifier = Linear(hidden, out)

    def forward(self, x_dict, edge_index_dict):
        x = {node: F.relu(self.lin_dict[node](feat)) for node, feat in x_dict.items() if node in self.lin_dict}
        for i in range(self.layers):
            confidence_dict = {
                node: 1.0 - 0.5 * self.unc_nets[i][node](h)
                for node, h in x.items() if node in self.unc_nets[i]
            }
            x_weighted = {node: x[node] * confidence_dict.get(node, 1.0) for node in x.keys()}
            x_hgt = self.convs[i](x_weighted, edge_index_dict)
            x_rel = self.rel_aggs[i](x_weighted, edge_index_dict)

            x_fused = {}
            for node in x.keys():
                h_orig = x[node]
                h_new = x_hgt.get(node, h_orig) + x_rel.get(node, torch.zeros_like(h_orig))
                gate = self.gates[i][node](torch.cat([h_orig, h_new], dim=-1))
                fused = gate * h_orig + (1.0 - gate) * self.drop(h_new)
                x_fused[node] = F.relu(self.ln_dict[i][node](fused))
            x = x_fused
        return self.classifier(x['file'])

class HIN_Full_R_HGT(HIN_HGT_Stage4_RelationGating):
    """Full R-HGT architecture. Stage 5 adds robust training; Stage 6 adds strict-inductive alignment fine-tuning."""
    pass

class HIN_Vanilla_HGT(nn.Module):
    """NO LONGER USED as a separate training configuration (since v3).

    This class has an AST fingerprint IDENTICAL to HIN_HGT_Stage1_Vanilla - the same
    architecture written twice. Before v3 both were trained, producing
    four result rows with four different names for ONE architecture.

    Now 'Multimodal + HIN HGT' is an alias of 'R-HGT Stage 1 (Vanilla HGT)',
    so only one checkpoint exists, exposed under two inference modes. The class
    is kept only because instantiate_hin_model() still references its name in a
    type-check list; do not add it back to hin_model_specs.
    """
    def __init__(self, hidden, out, heads, layers, metadata):
        super().__init__()
        self.node_types = metadata[0]
        self.edge_types = metadata[1]
        self.layers = layers

        self.lin_dict = nn.ModuleDict({node: Linear(-1, hidden) for node in self.node_types})
        self.convs = nn.ModuleList([HGTConv(hidden, hidden, metadata, heads) for _ in range(layers)])
        self.drop = nn.Dropout(0.2)
        self.classifier = Linear(hidden, out)

    def forward(self, x_dict, edge_index_dict):
        x = {node: F.relu(self.lin_dict[node](feat)) for node, feat in x_dict.items() if node in self.lin_dict}
        for i in range(self.layers):
            x = self.convs[i](x, edge_index_dict)
            x = {node: F.relu(self.drop(h)) for node, h in x.items()}
        return self.classifier(x['file'])

class HIN_GAT(nn.Module):
    def __init__(self, hidden, out, heads, metadata):
        super().__init__()
        node_types, edge_types = metadata[0], metadata[1]
        self.lin_dict = nn.ModuleDict({node: Linear(-1, hidden) for node in node_types})
        self.conv1 = HeteroConv({et: GATConv((-1, -1), hidden // heads, heads=heads, add_self_loops=False) for et in edge_types}, aggr='sum')
        self.conv2 = HeteroConv({et: GATConv((-1, -1), hidden // heads, heads=heads, add_self_loops=False) for et in edge_types}, aggr='sum')
        self.ln1 = nn.ModuleDict({node: nn.LayerNorm(hidden) for node in node_types})
        self.ln2 = nn.ModuleDict({node: nn.LayerNorm(hidden) for node in node_types})
        self.drop = nn.Dropout(0.2)
        self.lin = Linear(hidden, out)

    def forward(self, x_dict, edge_index_dict):
        x = {node: F.relu(self.lin_dict[node](feat)) for node, feat in x_dict.items() if node in self.lin_dict}
        out1 = self.conv1(x, edge_index_dict)
        x1 = {
            k: F.relu(self.ln1[k](x[k] + self.drop(out1.get(k, torch.zeros_like(x[k])))))
            for k in x.keys() if k in self.ln1
        }
        out2 = self.conv2(x1, edge_index_dict)
        x2 = {
            k: F.relu(self.ln2[k](x1[k] + self.drop(out2.get(k, torch.zeros_like(x1[k])))))
            for k in x1.keys() if k in self.ln2
        }
        return self.lin(x2['file'])

class HIN_HAN(nn.Module):
    def __init__(self, hidden, out, heads, metadata):
        super().__init__()
        node_types = metadata[0]
        self.lin_dict = nn.ModuleDict({node: Linear(-1, hidden) for node in node_types})
        self.conv1 = HANConv(hidden, hidden, metadata, heads=heads)
        self.ln = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(0.2)
        self.lin = Linear(hidden, out)

    def forward(self, x_dict, edge_index_dict):
        x = {node: F.relu(self.lin_dict[node](feat)) for node, feat in x_dict.items() if node in self.lin_dict}
        out_dict = self.conv1(x, edge_index_dict)
        h_file = F.relu(self.ln(x['file'] + self.drop(out_dict.get('file', x['file']))))
        return self.lin(h_file)

# =========================================================================================
# 9. TRAINING ENGINES
# =========================================================================================
def train_standard_hin_gnn(model, graph, epochs=100, lr=0.01, weight_decay=5e-4):
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t_start = time.perf_counter()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(graph.x_dict, graph.edge_index_dict)
        loss = criterion(out, graph['file'].y)
        loss.backward()
        optimizer.step()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_end = time.perf_counter()

    train_time_sec = t_end - t_start
    peak_gpu_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0

    model.eval()
    return model, train_time_sec, peak_gpu_mem_mb

def train_robust_hin_gnn_step(model, graph, epochs=100, lr=0.005, weight_decay=5e-4, adv_eps=0.05,
                              class_weight=False):
    """[v2] class_weight=True for the 'robust_weighted' variant - the missing cell of the
    2x2 ablation that separates the threshold effect from the representation effect."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t_start = time.perf_counter()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if class_weight:                                              # [v2]
        _y = graph['file'].y.detach().cpu().numpy()
        _c = np.bincount(_y, minlength=2).astype(np.float32)
        _c[_c == 0] = 1.0
        _w = _c.sum() / (2.0 * _c)
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(_w, dtype=torch.float32, device=device))
        print(f"      ⚖️  Adversarial training WITH class weights: {_w.round(3).tolist()}")
    else:
        criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()

        out_clean = model(graph.x_dict, graph.edge_index_dict)
        loss_clean = criterion(out_clean, graph['file'].y)

        x_file_orig = graph['file'].x.detach().requires_grad_(True)
        x_dict_adv = {k: v for k, v in graph.x_dict.items()}
        x_dict_adv['file'] = x_file_orig

        out_tmp = model(x_dict_adv, graph.edge_index_dict)
        loss_tmp = criterion(out_tmp, graph['file'].y)
        grad = torch.autograd.grad(loss_tmp, x_file_orig, retain_graph=False, create_graph=False)[0]

        x_dict_adv['file'] = (x_file_orig + adv_eps * grad.sign()).detach()

        out_adv = model(x_dict_adv, graph.edge_index_dict)
        loss_adv = criterion(out_adv, graph['file'].y)

        total_loss = 0.5 * loss_clean + 0.5 * loss_adv
        total_loss.backward()
        optimizer.step()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_end = time.perf_counter()

    train_time_sec = t_end - t_start
    peak_gpu_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0

    model.eval()
    return model, train_time_sec, peak_gpu_mem_mb

train_robust_hin_gnn = train_robust_hin_gnn_step

def fine_tune_strict_inductive_hin(
    model: nn.Module,
    ego_graphs: List[HeteroData],
    epochs: int = STRICT_FT_EPOCHS,
    lr: float = STRICT_FT_LR,
    weight_decay: float = 1e-4,
    adv_eps: float = STRICT_FT_ADV_EPS,
    clean_weight: float = STRICT_FT_CLEAN_WEIGHT,
    batch_size: int = STRICT_FT_BATCH_SIZE,
    seed: int = 42,
    grad_clip: float = STRICT_FT_GRAD_CLIP,
    use_class_weight: bool = True,      # [v2] disabled to create the Stage 6a variant
):
    """
    Align a pretrained R-HGT checkpoint to the exact strict-isolated inference regime.

    Only TRAIN ego-graphs are used. Each minibatch is a disjoint union of one-file
    ego-graphs, so there is no file-to-file message passing. A class-balanced clean
    objective restores minority-class recall, while a light FGSM term preserves the
    adversarial robustness learned by Stage 5.
    """
    if len(ego_graphs) == 0:
        return model, 0.0, 0.0

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t_start = time.perf_counter()

    model = model.to(device)
    labels_np = np.asarray([int(g['file'].y.view(-1)[0].item()) for g in ego_graphs], dtype=np.int64)
    counts = np.bincount(labels_np, minlength=2).astype(np.float32)
    counts[counts == 0] = 1.0
    class_weights = counts.sum() / (2.0 * counts)
    if use_class_weight:                                          # [v2]
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
        )
    else:
        criterion = nn.CrossEntropyLoss()
        print("      ⚖️  Strict fine-tuning WITHOUT class weights (variant 6a).")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=lr * 0.1
    )
    adv_weight = max(0.0, 1.0 - clean_weight)

    for epoch in range(epochs):
        model.train()
        rng = np.random.RandomState(seed + 104729 * (epoch + 1))
        order = rng.permutation(len(ego_graphs))
        running_loss = 0.0
        seen = 0

        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            batch_chunk = [ego_graphs[int(j)] for j in idx]
            batched = Batch.from_data_list(batch_chunk).to(device)
            y = batched['file'].y

            optimizer.zero_grad(set_to_none=True)

            out_clean = model(batched.x_dict, batched.edge_index_dict)
            loss_clean = criterion(out_clean, y)

            # Generate a light feature-space FGSM perturbation through the same strict graph.
            x_orig = batched['file'].x.detach().clone().requires_grad_(True)
            x_tmp = {k: v for k, v in batched.x_dict.items()}
            x_tmp['file'] = x_orig
            out_tmp = model(x_tmp, batched.edge_index_dict)
            loss_tmp = criterion(out_tmp, y)
            grad = torch.autograd.grad(loss_tmp, x_orig, retain_graph=False, create_graph=False)[0]

            x_adv = (x_orig + adv_eps * grad.sign()).detach()
            x_adv_dict = {k: v for k, v in batched.x_dict.items()}
            x_adv_dict['file'] = x_adv
            out_adv = model(x_adv_dict, batched.edge_index_dict)
            loss_adv = criterion(out_adv, y)

            total_loss = clean_weight * loss_clean + adv_weight * loss_adv
            total_loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            n = int(y.numel())
            running_loss += float(total_loss.detach().cpu()) * n
            seen += n

            del batched, out_clean, out_tmp, out_adv, x_orig, x_tmp, x_adv, x_adv_dict

        scheduler.step()
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            avg_loss = running_loss / max(seen, 1)
            print(
                f"      🔧 Strict-FT epoch {epoch + 1:02d}/{epochs}: "
                f"loss={avg_loss:.6f}, lr={scheduler.get_last_lr()[0]:.2e}"
            )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_end = time.perf_counter()
    peak_gpu_mem_mb = (
        torch.cuda.max_memory_allocated() / (1024 * 1024)
        if torch.cuda.is_available() else 0.0
    )
    model.eval()
    return model, t_end - t_start, peak_gpu_mem_mb

# =========================================================================================
# 10. METRICS & SUMMARY
# =========================================================================================
def evaluate_metrics(y_true, y_pred, y_prob, name, run_id, arch_mask=None, arch_name="Overall",
                     train_time_sec=0.0, infer_ms_sample=0.0, peak_gpu_mem_mb=0.0):
    if arch_mask is not None:
        y_true = y_true[arch_mask]
        y_pred = y_pred[arch_mask]
        y_prob = y_prob[arch_mask]

    if len(y_true) == 0:
        return {}

    if y_prob.ndim > 1 and y_prob.shape[1] > 1:
        probs = y_prob[:, 1]
    else:
        probs = y_prob

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        spec = 0.0

    try: roc_auc = roc_auc_score(y_true, probs)
    except Exception: roc_auc = 0.5

    try:
        p, r, _ = precision_recall_curve(y_true, probs)
        pr_auc = auc(r, p)
    except Exception: pr_auc = 0.5

    return {
        'Config': name, 'Run': run_id, 'Architecture': arch_name, 'Accuracy': acc,
        'Precision': prec, 'Recall': rec, 'Specificity': spec, 'F1': f1, 'MCC': mcc,
        'ROC-AUC': roc_auc, 'PR-AUC': pr_auc,
        'Training Time (s)': train_time_sec,
        'Inference (ms/sample)': infer_ms_sample,
        'Peak GPU Memory (MB)': peak_gpu_mem_mb
    }

def format_summary_dataframe(df, group_cols, metric_cols):
    agg_dict = {m: ['mean', 'std'] for m in metric_cols if m in df.columns}
    # [v2 / FIX C2] dropna=False: if a group key is missing, the row still appears
    # instead of silently disappearing.
    grouped = df.groupby(group_cols, dropna=False).agg(agg_dict).reset_index()

    res_df = pd.DataFrame()
    for col in group_cols:
        res_df[col] = grouped[col]

    for m in metric_cols:
        if (m, 'mean') in grouped.columns:
            means = grouped[(m, 'mean')]
            stds = grouped[(m, 'std')].fillna(0.0)
            res_df[m] = [f"{mean:.4f} ± {std:.4f}" for mean, std in zip(means, stds)]

    return res_df

# =========================================================================================
# 11. MAIN BENCHMARK PIPELINE
# =========================================================================================
def main():
    print("=" * 100)
    print("🚀 STARTING RECTIFIED BENCHMARK: GENUINE STRICT INDUCTIVE HIN GNN PIPELINE")
    print("    - MAIN HIN BENCHMARK: ZERO TEST-TO-TEST MESSAGE PASSING VIA ISOLATED EGO-GRAPHS")
    print("    - DATASET: CIC-YNU-IoTMal2026 (ARM, MIPS, MIPSEL, X86)")
    print("    - HASH-GROUPED SAFE SPLITTING: ELIMINATING INTRA-SAMPLE CROSS-ARCH CONTAMINATION")
    print("    - CONTROLLED PIPELINE ABLATION: STAGES 1-5 -> STAGE 6 STRICT-INDUCTIVE ALIGNMENT FINE-TUNING")
    print("=" * 100)

    df = load_and_merge_multimodal_parquet(BASE_DATA_DIR)
    if df.empty:
        print("❌ Error: No dataset available. Please ensure data directory exists.")
        return

    all_cols = df.columns.tolist()
    syscall_cols = [c for c in all_cols if c.startswith('Call_') or 'syscall' in c.lower()]
    proto_list = [c for c in ['TCP', 'UDP', 'ICMP', 'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SSH', 'IRC', 'DHCP', 'ARP', 'IGMP', 'IPv', 'LLC'] if c in all_cols]
    exclude_cols = set(['label', 'arch', 'Arch', 'Hash', 'MalwareFamily', 'stratify_col'] + syscall_cols + proto_list)
    pcap_cols = [c for c in all_cols if c in ['Header_Length', 'Protocol Type', 'Time_To_Live', 'Rate', 'Tot sum', 'Min', 'Max', 'AVG', 'Std', 'Tot size', 'IAT', 'Number', 'Variance'] or c.endswith('_flag_number') or c.endswith('_count')]
    sar_cols = [c for c in all_cols if c not in exclude_cols and c not in pcap_cols]
    num_cols = pcap_cols + sar_cols
    all_feature_cols = pcap_cols + sar_cols + syscall_cols

    print(f"📊 Multimodal Feature Breakdown: {len(pcap_cols)} PCAP | {len(sar_cols)} SAR | {len(syscall_cols)} STRACE -> Total: {len(all_feature_cols)} Dims")

    # [v3] The column-grouping heuristic relies on a HARDCODED list of names; if the parquet
    # schema differs by a single character, a group may be empty and the model would get in_dim = 0
    # without a clear error. The original dataset paper reports PCAP = 40 and SAR+STRACE =
    # 392-461 features - compare against the counts printed on the line above.
    for _gname, _gcols in (('pcap', pcap_cols), ('sar', sar_cols),
                           ('syscall', syscall_cols)):
        assert len(_gcols) > 0, (
            f"Feature group '{_gname}' is EMPTY - the column-grouping heuristic does not "
            f"match the parquet schema. Check the column names before continuing.")
    print("   ↳ sar_cols takes ALL remaining columns. Review the list below manually to make "
          "sure no identifier / timestamp columns slipped in as features:")
    print(f"   ↳ {sar_cols[:30]}{' ...' if len(sar_cols) > 30 else ''}")

    df_clean = df.copy()
    df_clean['stratify_col'] = df_clean['label'].astype(str) + "_" + df_clean['arch'].astype(str)
    valid_keys = df_clean['stratify_col'].value_counts()[lambda x: x > 1].index
    df_valid = df_clean[df_clean['stratify_col'].isin(valid_keys)].copy()

    # Tables 1 & 2 Generation
    df_stat = df_clean.copy()
    arch_map = {'arm': 'ARM', 'mips': 'MIPS', 'mipsel': 'MIPSEL', 'x86': 'x86'}
    df_stat['Arch_Norm'] = df_stat['arch'].astype(str).str.lower().map(arch_map).fillna(df_stat['arch'])
    df_stat['Is_Benign'] = df_stat['MalwareFamily'].astype(str).str.lower() == 'benign'

    table1_rows = []
    arch_order = ['ARM', 'MIPS', 'MIPSEL', 'x86']
    for arch in arch_order:
        sub = df_stat[df_stat['Arch_Norm'] == arch]
        benign_cnt = int(sub['Is_Benign'].sum())
        malware_cnt = int((~sub['Is_Benign']).sum())
        total_cnt = len(sub)
        mal_fams = sorted(sub[~sub['Is_Benign']]['MalwareFamily'].astype(str).unique().tolist())
        mal_fams_str = ", ".join(mal_fams) if mal_fams else "—"
        table1_rows.append({'Architecture': arch, 'Benign': benign_cnt, 'Malware': malware_cnt, 'Total': total_cnt, 'Malware Families': mal_fams_str})

    table1_rows.append({'Architecture': 'Total', 'Benign': int(df_stat['Is_Benign'].sum()), 'Malware': int((~df_stat['Is_Benign']).sum()), 'Total': len(df_stat), 'Malware Families': '—'})
    pd.DataFrame(table1_rows).to_csv(os.path.join(STAT_DIR, 'table1_architecture_summary.csv'), index=False)

    df_malware = df_stat[~df_stat['Is_Benign']].copy()
    total_malware_samples = len(df_malware)
    target_families = ['Mirai', 'Gafgyt', 'DarkNexus', 'Generic', 'Tsunami']
    counts = df_malware['MalwareFamily'].value_counts()
    table2_rows = []
    other_cnt = 0
    for fam in target_families:
        cnt = int(counts.get(fam, 0))
        pct = (cnt / total_malware_samples * 100) if total_malware_samples > 0 else 0
        table2_rows.append({'Malware Family': fam, '# Samples': cnt, 'Percentage (%)': round(pct, 2)})
    for fam, cnt in counts.items():
        if fam not in target_families: other_cnt += cnt
    other_pct = (other_cnt / total_malware_samples * 100) if total_malware_samples > 0 else 0
    table2_rows.append({'Malware Family': 'Other', '# Samples': int(other_cnt), 'Percentage (%)': round(other_pct, 2)})
    table2_rows.append({'Malware Family': 'Total malware', '# Samples': total_malware_samples, 'Percentage (%)': 100.0})
    pd.DataFrame(table2_rows).to_csv(os.path.join(STAT_DIR, 'table2_malware_family_breakdown.csv'), index=False)

    all_eval_metrics_cols = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1', 'MCC', 'ROC-AUC', 'PR-AUC', 'Training Time (s)', 'Inference (ms/sample)', 'Peak GPU Memory (MB)']
    target_archs = ['ARM', 'MIPS', 'Mipsel', 'x86']

    noise_scenarios = [
        {'type': 'Clean', 'levels': [0.0]},
        {'type': 'Gaussian_Noise', 'levels': [0.05, 0.1, 0.2, 0.5]},
        {'type': 'Feature_Masking', 'levels': [0.1, 0.2, 0.3, 0.5]},
        {'type': 'Edge_Perturbation', 'levels': [0.1, 0.2, 0.3, 0.5]},
        {'type': 'FGSM_Attack', 'levels': [0.01, 0.05, 0.1, 0.2]},
        {'type': 'BIM_Attack', 'levels': [0.01, 0.05, 0.1, 0.2]},
        {'type': 'PGD_Attack', 'levels': [0.01, 0.05, 0.1, 0.2]}
    ]

    # [v2] Filter by RHGT_SCENARIOS to split one long run into several sessions.
    _NOISE_ONLY = ['Gaussian_Noise', 'Feature_Masking', 'Edge_Perturbation']
    _ATTACK_ONLY = ['FGSM_Attack', 'BIM_Attack', 'PGD_Attack']
    if SCENARIO_SET == 'clean':
        noise_scenarios = [s for s in noise_scenarios if s['type'] == 'Clean']
    elif SCENARIO_SET == 'noise':
        noise_scenarios = [s for s in noise_scenarios if s['type'] in ['Clean'] + _NOISE_ONLY]
    elif SCENARIO_SET == 'attack':
        noise_scenarios = [s for s in noise_scenarios if s['type'] in ['Clean'] + _ATTACK_ONLY]
    print(f"🧪 Evaluation scenarios: {[s['type'] for s in noise_scenarios]}")

    # Same white-box attack budget for every comparable model family.
    # [v2] PGD accounts for most of the evaluation time (27 ms/sample vs 0.41 ms for
    # clean). The default below is 5 restarts; the paper's runs set
    # RHGT_ATTACK_RESTARTS=3 (see results/*/run_config.json), which saves ~40% of the time.
    ATTACK_ITERS = int(os.environ.get('RHGT_ATTACK_ITERS', '20'))
    ATTACK_RESTARTS = int(os.environ.get('RHGT_ATTACK_RESTARTS', '5'))

    # [v3 / J1] Restrict the set of configs subjected to noise and attacks.
    #
    # Attack evaluation is hundreds of times more expensive than clean evaluation: PGD 20 steps × 3
    # restarts × 4 epsilon levels, plus BIM and FGSM, is 324 forward-backward passes
    # for EACH config over the whole test set. In strict mode a single forward pass
    # already costs 0.362 ms/sample, so 14 strict configs × 3,404 samples
    # ≈ 4.7 hours for ONE seed - a 12-hour Kaggle session only fits 2 seeds.
    #
    # The robustness curves only need a few configs. Set RHGT_ATTACK_CONFIGS to a
    # comma-separated list of names to attack only those configs; all other configs are STILL
    # fully evaluated in the Clean scenario, so the main table loses no rows.
    # Empty (default) = keep the previous behaviour.
    _ATK_CFG_ENV = os.environ.get('RHGT_ATTACK_CONFIGS', '').strip()
    ATTACK_CONFIGS = {c.strip() for c in _ATK_CFG_ENV.split(',') if c.strip()}
    if ATTACK_CONFIGS:
        print(f"🎯 [v3] Noise/attacks apply only to {len(ATTACK_CONFIGS)} HIN configs "
              f"listed in RHGT_ATTACK_CONFIGS; Clean still runs for all.")
        for _c in sorted(ATTACK_CONFIGS):
            print(f"      • {_c}")

    # Main benchmark: all external HIN baselines and the Proposed model use strict isolated inference.
    # Pipeline ablation: Stage 6 starts from the exact Stage-5 checkpoint, then fine-tunes ONLY on
    # isolated TRAIN ego-graphs before strict isolated TEST inference. This aligns train/test graph
    # regimes without using test data and turns Stage 6 into a genuine additional method component.
    hin_model_specs = [
        {'cls': HIN_Vanilla_RGCN, 'name': 'Multimodal + HIN RGCN', 'kwargs': {}, 'train_mode': 'standard', 'eval_mode': 'strict'},
        {'cls': HIN_R2_RGCN, 'name': 'Multimodal + HIN R2-RGCN', 'kwargs': {}, 'train_mode': 'standard', 'eval_mode': 'strict'},
        {'cls': HIN_MAGNN, 'name': 'Multimodal + HIN MAGNN', 'kwargs': {}, 'train_mode': 'standard', 'eval_mode': 'strict'},
        {'cls': HIN_HAN, 'name': 'Multimodal + HIN HAN', 'kwargs': {'heads': 4}, 'train_mode': 'standard', 'eval_mode': 'strict'},
        {'cls': HIN_GAT, 'name': 'Multimodal + HIN GAT', 'kwargs': {'heads': 4}, 'train_mode': 'standard', 'eval_mode': 'strict'},
        # [v3] REMOVED the separate training config 'Multimodal + HIN HGT'.
        # HIN_Vanilla_HGT and HIN_HGT_Stage1_Vanilla have IDENTICAL AST fingerprints -
        # the same architecture written twice. Training both produced four result rows
        # with four names for ONE architecture, differing only by random
        # initialisation; in the paper that would look like four methods. Now a single
        # checkpoint is exposed under two eval modes.
        {'cls': HIN_HGT_Stage1_Vanilla, 'name': 'R-HGT Stage 1 (Vanilla HGT)', 'kwargs': {'heads': 4, 'layers': 2}, 'train_mode': 'standard', 'eval_mode': 'shared'},
        {'alias_of': 'R-HGT Stage 1 (Vanilla HGT)', 'name': 'Multimodal + HIN HGT', 'eval_mode': 'strict'},
        {'cls': HIN_HGT_Stage2_ResLN, 'name': 'R-HGT Stage 2 (+ Residual + LayerNorm)', 'kwargs': {'heads': 4, 'layers': 2}, 'train_mode': 'standard', 'eval_mode': 'shared'},
        {'cls': HIN_HGT_Stage3_NodeReliability, 'name': 'R-HGT Stage 3 (+ Node Reliability)', 'kwargs': {'heads': 4, 'layers': 2}, 'train_mode': 'standard', 'eval_mode': 'shared'},
        {'cls': HIN_HGT_Stage4_RelationGating, 'name': 'R-HGT Stage 4 (+ Relation Reliability / Gating)', 'kwargs': {'heads': 4, 'layers': 2}, 'train_mode': 'standard', 'eval_mode': 'shared'},
        {'cls': HIN_Full_R_HGT, 'name': 'R-HGT Stage 5 (+ Robust Training)', 'kwargs': {'heads': 4, 'layers': 2}, 'train_mode': 'robust', 'eval_mode': 'shared'},
        {'alias_of': 'R-HGT Stage 5 (+ Robust Training)', 'name': 'R-HGT Strict Control (No Fine-Tuning)', 'eval_mode': 'strict'},
        {'clone_of': 'R-HGT Stage 5 (+ Robust Training)', 'name': 'R-HGT Stage 6 (+ Strict-Inductive Fine-Tuning)', 'train_mode': 'strict_finetune', 'eval_mode': 'strict'},
        {'alias_of': 'R-HGT Stage 6 (+ Strict-Inductive Fine-Tuning)', 'name': 'Multimodal + HIN R-HGT', 'eval_mode': 'strict'},
    ]

    # ================= v2 ADDITIONS =================
    # Enable with RHGT_EXTRA, e.g. RHGT_EXTRA="AC" to run groups A and C.
    # An alias MUST come after its base model in the list.
    if 'A' in EXTRA_GROUPS:
        # No additional training: the alias shares the checkpoint and is only re-evaluated
        # under a different inference mode. This turns the "protocol gap" from an isolated
        # observation into a general result.
        hin_model_specs += [
            {'alias_of': 'Multimodal + HIN RGCN',    'name': 'HIN RGCN [shared eval]',    'eval_mode': 'shared'},
            {'alias_of': 'Multimodal + HIN R2-RGCN', 'name': 'HIN R2-RGCN [shared eval]', 'eval_mode': 'shared'},
            {'alias_of': 'Multimodal + HIN MAGNN',   'name': 'HIN MAGNN [shared eval]',   'eval_mode': 'shared'},
            {'alias_of': 'Multimodal + HIN HAN',     'name': 'HIN HAN [shared eval]',     'eval_mode': 'shared'},
            {'alias_of': 'Multimodal + HIN GAT',     'name': 'HIN GAT [shared eval]',     'eval_mode': 'shared'},
            # [v3] REMOVED 'HIN HGT [shared eval]' and 'R-HGT Stage 1 [strict eval]':
            # after G1, 'R-HGT Stage 1' (shared) and 'Multimodal + HIN HGT'
            # (strict) already cover both modes for that exact architecture.
            {'alias_of': 'R-HGT Stage 2 (+ Residual + LayerNorm)',          'name': 'R-HGT Stage 2 [strict eval]', 'eval_mode': 'strict'},
            {'alias_of': 'R-HGT Stage 3 (+ Node Reliability)',              'name': 'R-HGT Stage 3 [strict eval]', 'eval_mode': 'strict'},
            {'alias_of': 'R-HGT Stage 4 (+ Relation Reliability / Gating)', 'name': 'R-HGT Stage 4 [strict eval]', 'eval_mode': 'strict'},
        ]

    if 'B' in EXTRA_GROUPS:
        # ALL 6 HIN baselines get EXACTLY the same procedure as the proposed model
        # (Stage 5 adversarial training + Stage 6 strict fine-tuning). This is
        # what makes the main comparison table sound: if R-HGT still leads when every
        # baseline gets the same procedure, the advantage comes from the ARCHITECTURE and
        # not from being the only model trained adversarially.
        _B_BASE = [
            ('HIN RGCN',    HIN_Vanilla_RGCN, {}),
            ('HIN R2-RGCN', HIN_R2_RGCN,      {}),
            ('HIN MAGNN',   HIN_MAGNN,        {}),
            ('HIN HAN',     HIN_HAN,          {'heads': 4}),
            ('HIN GAT',     HIN_GAT,          {'heads': 4}),
            ('HIN HGT',     HIN_Vanilla_HGT,  {'heads': 4, 'layers': 2}),
        ]
        for _nm, _cls, _kw in _B_BASE:
            hin_model_specs += [
                {'cls': _cls, 'name': f'{_nm} (robust)', 'kwargs': _kw,
                 'train_mode': 'robust', 'eval_mode': 'strict'},
                {'clone_of': f'{_nm} (robust)', 'name': f'{_nm} (robust + strict FT)',
                 'train_mode': 'strict_finetune', 'eval_mode': 'strict'},
            ]

    if 'C' in EXTRA_GROUPS:
        # 2x2 ablation separating class weights from strict fine-tuning:
        #                     no class weight | class weight
        #   no strict FT :      Stage 5       |   Stage 5b  <-- new
        #   strict FT    :      Stage 6a <-- new |   Stage 6
        hin_model_specs += [
            {'clone_of': 'R-HGT Stage 5 (+ Robust Training)',
             'name': 'R-HGT Stage 6a (FT, no class weight)',
             'train_mode': 'strict_finetune_noweight', 'eval_mode': 'strict'},
            {'cls': HIN_Full_R_HGT, 'name': 'R-HGT Stage 5b (class-weighted, no FT)',
             'kwargs': {'heads': 4, 'layers': 2},
             'train_mode': 'robust_weighted', 'eval_mode': 'strict'},
        ]

    print(f"🧬 hin_model_specs: {len(hin_model_specs)} configs "
          f"(extra groups: {EXTRA_GROUPS or 'none'})")

    def instantiate_hin_model(spec, metadata):
        model_cls = spec['cls']
        kwargs = spec.get('kwargs', {})
        if model_cls in [HIN_HAN, HIN_GAT]:
            return model_cls(hidden=64, out=2, metadata=metadata, **kwargs).to(device)
        if model_cls in [HIN_Vanilla_HGT, HIN_HGT_Stage1_Vanilla, HIN_HGT_Stage2_ResLN,
                         HIN_HGT_Stage3_NodeReliability, HIN_HGT_Stage4_RelationGating, HIN_Full_R_HGT]:
            return model_cls(hidden=64, out=2, metadata=metadata, **kwargs).to(device)
        return model_cls(hidden=64, out=2, metadata=metadata).to(device)

    robustness_global_records = []
    robustness_arch_records = []
    raw_glob_path = os.path.join(ROBUSTNESS_GLOBAL_DIR, "overall_robustness_raw_10seed.csv")
    raw_arch_path = os.path.join(ROBUSTNESS_ARCH_DIR, "per_architecture_robustness_raw_10seed.csv")

    def apply_tabular_noise(X, noise_type, level, seed=42):
        if noise_type == 'Clean' or level == 0: return X
        X_noisy = X.clone() if isinstance(X, torch.Tensor) else torch.tensor(X, dtype=torch.float32).clone()
        g = torch.Generator().manual_seed(seed)
        if noise_type == 'Gaussian_Noise': X_noisy = X_noisy + torch.randn(X_noisy.shape, generator=g) * level
        elif noise_type == 'Feature_Masking': X_noisy = X_noisy * (torch.rand(X_noisy.shape, generator=g) > level).float()
        return X_noisy

    # ===================== RESUME v2 =====================
    # A seed counts as DONE only when it has ALL requested noise scenarios AND ALL requested
    # model configs. This way, a later session with
    # RHGT_SCENARIOS=attack or a wider RHGT_EXTRA adds exactly the missing part
    # instead of wrongly skipping the seed.
    _done_main = set()
    _req_types = {s['type'] for s in noise_scenarios}
    _req_cfgs = {s['name'] for s in hin_model_specs}
    if RESUME:
        for _p, _sink in ((raw_glob_path, robustness_global_records),
                          (raw_arch_path, robustness_arch_records)):
            if os.path.exists(_p):
                try:
                    _prev = pd.read_csv(_p)
                    _sink.extend(_prev.to_dict('records'))
                    print(f"♻️  Reloaded {len(_prev)} records from {os.path.basename(_p)}")
                except Exception as _e:
                    print(f"[!] Could not read {_p}: {_e}")
        _seen = {}
        for _r in robustness_global_records:
            try:
                _k = int(_r.get('Run'))
            except Exception:
                continue
            _e = _seen.setdefault(_k, {'types': set(), 'cfgs': set()})
            _e['types'].add(str(_r.get('Noise_Type')))
            _e['cfgs'].add(str(_r.get('Config')))
        for _k, _v in _seen.items():
            if _req_types.issubset(_v['types']) and _req_cfgs.issubset(_v['cfgs']):
                _done_main.add(_k)
            else:
                print(f"♻️  [SEED {_k}] has partial results but is missing "
                      f"{len(_req_types - _v['types'])} scenarios / "
                      f"{len(_req_cfgs - _v['cfgs'])} configs → this seed will be re-run.")
        if _done_main:
            print(f"♻️  Completed seeds: {sorted(_done_main)}")

    for seed in ([] if SKIP_MAIN else SEEDS):
        if RESUME and seed in _done_main:
            print(f"♻️  [SEED {seed}] already has complete results - skipping.")
            continue
        print(f"\n🌱 [SEED {seed}/10] Executing Group-Safe Hash Split (80:20)...")
        seed_everything(seed)

        df_tr, df_te = group_safe_stratified_split(df_valid, group_col='Hash', label_col='label', test_size=0.2, seed=seed)
        y_tr, y_te = df_tr['label'].values, df_te['label'].values
        arch_te_arr = df_te['arch'].astype(str).values

        # A. Preprocessing
        x_tr_pcap = clean_numeric_array(df_tr, pcap_cols)
        x_te_pcap = clean_numeric_array(df_te, pcap_cols)
        sc_pcap = StandardScaler().fit(x_tr_pcap)
        x_tr_pcap_sc = sc_pcap.transform(x_tr_pcap)
        x_te_pcap_sc = sc_pcap.transform(x_te_pcap)

        x_tr_sar = clean_numeric_array(df_tr, sar_cols)
        x_te_sar = clean_numeric_array(df_te, sar_cols)
        sc_sar = StandardScaler().fit(x_tr_sar)
        x_tr_sar_sc = sc_sar.transform(x_tr_sar)
        x_te_sar_sc = sc_sar.transform(x_te_sar)

        x_tr_sys = clean_numeric_array(df_tr, syscall_cols)
        x_te_sys = clean_numeric_array(df_te, syscall_cols)
        sc_sys = StandardScaler().fit(x_tr_sys)
        x_tr_sys_sc = sc_sys.transform(x_tr_sys)
        x_te_sys_sc = sc_sys.transform(x_te_sys)

        # B. Train Baselines
        pcap_m = PCAP_MLP(in_dim=len(pcap_cols))
        trained_pcap, t_tr_pcap, mem_pcap = train_neural_branch(pcap_m, x_tr_pcap_sc, y_tr)

        sar_m = SAR_Autoencoder(in_dim=len(sar_cols))
        trained_sar, t_tr_sar, mem_sar = train_neural_branch(sar_m, x_tr_sar_sc, y_tr)

        sys_m = STRACE_CNN_LSTM(in_dim=len(syscall_cols))
        trained_sys, t_tr_sys, mem_sys = train_neural_branch(sys_m, x_tr_sys_sc, y_tr)

        fusion_model = MultimodalLateFusionModel(pcap_dim=len(pcap_cols), sar_dim=len(sar_cols), strace_dim=len(syscall_cols)).to(device)
        fusion_model.load_pretrained_branches(trained_pcap, trained_sar, trained_sys)
        fusion_model, t_tr_fusion, mem_fusion = train_end_to_end_joint_fine_tuning(fusion_model, x_tr_pcap_sc, x_tr_sar_sc, x_tr_sys_sc, y_tr)

        g_homo_tr, scaler_homo = build_homogeneous_multimodal_graph(df_tr, all_feature_cols, k=5, mode='train')
        g_homo_te_clean, _ = build_homogeneous_multimodal_graph(df_te, all_feature_cols, k=5, scaler=scaler_homo, mode='test')

        g_homo_tr = g_homo_tr.to(device)
        model_non_hin = Multimodal_NonHIN_GNN(in_dim=g_homo_tr.x.size(1), hidden_dim=64, out_dim=2).to(device)
        opt_non_hin = torch.optim.Adam(model_non_hin.parameters(), lr=0.01, weight_decay=5e-4)

        if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        t_start_non_hin = time.perf_counter()
        model_non_hin.train()
        for epoch in range(100):
            opt_non_hin.zero_grad()
            out = model_non_hin(g_homo_tr.x, g_homo_tr.edge_index)
            loss = F.cross_entropy(out, g_homo_tr.y)
            loss.backward()
            opt_non_hin.step()
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t_tr_non_hin = time.perf_counter() - t_start_non_hin
        mem_non_hin = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
        model_non_hin.eval()

        # C. Train HIN GNNs & Extract Canonical Training Entity Anchors
        print(f"\n🌐 [SEED {seed}] Constructing Heterogeneous Information Network (HIN) Graphs...")
        g_hin_tr, scaler_hin, arch_le_hin = build_hetero_graph_hin(df_tr, num_cols, syscall_cols, proto_list, mode='train')
        g_hin_te_clean, _, _ = build_hetero_graph_hin(df_te, num_cols, syscall_cols, proto_list, scaler=scaler_hin, arch_le=arch_le_hin, mode='test')
        g_hin_tr = g_hin_tr.to(device)
        metadata_hin = g_hin_tr.metadata()

        canonical_anchors = RectifiedStrictInductiveEngine.extract_canonical_anchors(g_hin_tr)
        ego_graphs_train = RectifiedStrictInductiveEngine.build_isolated_ego_graphs_batch(
            df_tr, num_cols, syscall_cols, proto_list, scaler_hin, arch_le_hin, canonical_anchors
        )
        ego_graphs_test = RectifiedStrictInductiveEngine.build_isolated_ego_graphs_batch(
            df_te, num_cols, syscall_cols, proto_list, scaler_hin, arch_le_hin, canonical_anchors
        )

        trained_hin_models = {}

        for spec in hin_model_specs:
            model_name = spec['name']
            print(f"   ⚡ Preparing HIN Architecture: {model_name}...")

            if 'alias_of' in spec:
                base_name = spec['alias_of']
                if base_name not in trained_hin_models:
                    raise RuntimeError(f"Alias base model not trained yet: {base_name}")
                base = trained_hin_models[base_name]
                trained_hin_models[model_name] = {
                    'model': base['model'],
                    'train_time': base['train_time'],
                    'peak_mem': base['peak_mem'],
                    'eval_mode': spec.get('eval_mode', 'strict'),
                    'shared_weights_with': base_name,
                    'initialized_from': base.get('initialized_from'),
                    'train_mode': base.get('train_mode', 'alias'),
                }
                continue

            if 'clone_of' in spec:
                base_name = spec['clone_of']
                if base_name not in trained_hin_models:
                    raise RuntimeError(f"Fine-tune base model not trained yet: {base_name}")
                base = trained_hin_models[base_name]
                hin_model = copy.deepcopy(base['model']).to(device)
                print(f"      🔁 Initializing strict fine-tuning from: {base_name}")
                hin_model, t_ft, mem_ft = fine_tune_strict_inductive_hin(
                    hin_model, ego_graphs_train,
                    epochs=STRICT_FT_EPOCHS, lr=STRICT_FT_LR,
                    weight_decay=1e-4, adv_eps=STRICT_FT_ADV_EPS,
                    clean_weight=STRICT_FT_CLEAN_WEIGHT,
                    batch_size=STRICT_FT_BATCH_SIZE, seed=seed,
                    use_class_weight=(spec.get('train_mode') != 'strict_finetune_noweight'),  # [v2]
                )
                trained_hin_models[model_name] = {
                    'model': hin_model,
                    'train_time': base['train_time'] + t_ft,
                    'peak_mem': max(base['peak_mem'], mem_ft),
                    'eval_mode': spec.get('eval_mode', 'strict'),
                    'shared_weights_with': None,
                    'initialized_from': base_name,
                    'train_mode': spec.get('train_mode', 'strict_finetune'),   # [v2]
                }
                continue

            hin_model = instantiate_hin_model(spec, metadata_hin)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
            t_start_hin = time.perf_counter()

            if spec.get('train_mode') in ('robust', 'robust_weighted'):
                hin_model, _, _ = train_robust_hin_gnn(
                    hin_model, g_hin_tr, epochs=100, lr=0.005, weight_decay=5e-4, adv_eps=0.05,
                    class_weight=(spec.get('train_mode') == 'robust_weighted')   # [v2]
                )
            else:
                hin_model, _, _ = train_standard_hin_gnn(
                    hin_model, g_hin_tr, epochs=100, lr=0.005, weight_decay=5e-4
                )

            if torch.cuda.is_available(): torch.cuda.synchronize()
            t_tr_hin = time.perf_counter() - t_start_hin
            mem_hin = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0

            trained_hin_models[model_name] = {
                'model': hin_model,
                'train_time': t_tr_hin,
                'peak_mem': mem_hin,
                'eval_mode': spec.get('eval_mode', 'strict'),
                'shared_weights_with': None,
                'initialized_from': None,
                'train_mode': spec.get('train_mode', 'standard'),
            }

        # D. Evaluation Across Noise & Attack Scenarios
        print(f"\n🧪 [SEED {seed}] Executing Comprehensive Evaluation Across Noise & Attack Scenarios...")
        X_pcap_eval = x_te_pcap_sc
        X_sar_eval  = x_te_sar_sc
        X_sys_eval  = x_te_sys_sc

        m_pcap   = trained_pcap
        m_sar    = trained_sar
        m_sys    = trained_sys
        m_fusion = fusion_model

        t_pcap_v   = t_tr_pcap
        t_sar_v    = t_tr_sar
        t_sys_v    = t_tr_sys
        t_fusion_v = t_tr_fusion

        mem_pcap_v   = mem_pcap
        mem_sar_v    = mem_sar
        mem_sys_v    = mem_sys
        mem_fusion_v = mem_fusion

        def record_evaluation_metrics(preds_eval, probs_eval, model_label, t_train, infer_ms, mem_peak, noise_type_val, noise_level_val, inference_regime='inductive'):
            # [v2] Save per-sample probabilities in the Clean scenario. This is a required input
            # for the threshold-sweep experiment: otherwise there is no way to know the F1 of
            # Strict Control at the optimal threshold, and that number determines how much of
            # the +0.1023 F1 is due to genuine fine-tuning.
            if SAVE_PROBS and noise_type_val == 'Clean':
                try:
                    _safe = "".join(c if (c.isalnum() or c in '._-') else '_'
                                    for c in str(model_label))
                    np.savez_compressed(
                        os.path.join(PROB_DUMP_DIR, f"clean_seed{seed}_{_safe}.npz"),
                        y_true=np.asarray(y_te, dtype=np.int8),
                        y_prob=np.asarray(probs_eval, dtype=np.float32),
                        arch=np.asarray(arch_te_arr, dtype='U8'),
                        # [v3 / K1] Danh tinh mau test (Hash).
                        #
                        # Khong co truong nay thi khong the tinh lai chi so tren
                        # mot TAP CON cua tap test - vi du sau khi bo mau gan
                        # trung - boi thu tu tap test KHONG tai tao duoc ngoai
                        # phien chay. Phep chia dung StratifiedGroupKFold voi
                        # shuffle=True, ket qua phu thuoc THU TU DONG cua
                        # DataFrame, ma thu tu do lai do DuckDB quyet dinh luc
                        # GROUP BY; DuckDB khac phien ban tra ve thu tu khac.
                        # Da kiem chung: tai tao ngoai phien bang DuckDB 1.5.5
                        # lech 1.424/3.404 nhan so voi ban chay tren Kaggle
                        # (DuckDB 1.3.2), du kich thuoc tap test va phan bo tang
                        # khop chinh xac.
                        #
                        # Luu Hash bien moi phan tich tap con thanh viec CHI CAN
                        # CPU, va lam phep chia kiem chung duoc boi nguoi khac.
                        hash=np.asarray(df_te['Hash'].astype(str).to_numpy(), dtype='U64'),
                        regime=np.asarray([inference_regime], dtype='U32'))
                except Exception as _e:
                    print(f"      [!] Could not save y_prob ({model_label}): {_e}")

            m_glob = evaluate_metrics(y_te, preds_eval, probs_eval, model_label, seed,
                                      train_time_sec=t_train, infer_ms_sample=infer_ms, peak_gpu_mem_mb=mem_peak)
            m_glob['Noise_Type'] = noise_type_val; m_glob['Noise_Level'] = noise_level_val; m_glob['Inference_Regime'] = inference_regime
            m_glob['Attack_Steps'] = ATTACK_ITERS if noise_type_val in ['BIM_Attack', 'PGD_Attack'] else (1 if noise_type_val == 'FGSM_Attack' else 0)
            m_glob['Attack_Restarts'] = ATTACK_RESTARTS if noise_type_val == 'PGD_Attack' else 0
            robustness_global_records.append(m_glob)

            for arch_item in target_archs:
                mask = (np.array([a.upper() for a in arch_te_arr]) == arch_item.upper())
                if mask.sum() > 0:
                    m_arch = evaluate_metrics(y_te, preds_eval, probs_eval, model_label, seed,
                                              arch_mask=mask, arch_name=arch_item,
                                              train_time_sec=t_train, infer_ms_sample=infer_ms, peak_gpu_mem_mb=mem_peak)
                    m_arch['Noise_Type'] = noise_type_val; m_arch['Noise_Level'] = noise_level_val; m_arch['Inference_Regime'] = inference_regime
                    m_arch['Attack_Steps'] = ATTACK_ITERS if noise_type_val in ['BIM_Attack', 'PGD_Attack'] else (1 if noise_type_val == 'FGSM_Attack' else 0)
                    m_arch['Attack_Restarts'] = ATTACK_RESTARTS if noise_type_val == 'PGD_Attack' else 0
                    robustness_arch_records.append(m_arch)

        for scen in noise_scenarios:
            st_type = scen['type']
            for lvl in scen['levels']:

                # 1. PCAP MLP
                if st_type in ['Gaussian_Noise', 'Feature_Masking']:
                    bx_p_eval = apply_tabular_noise(X_pcap_eval, st_type, lvl, seed=seed)
                elif st_type in ['FGSM_Attack', 'BIM_Attack', 'PGD_Attack']:
                    bx_p_eval = attack_tensor_model_rectified(
                        m_pcap, torch.tensor(X_pcap_eval, dtype=torch.float32),
                        torch.tensor(y_te, dtype=torch.long),
                        attack_type=st_type.replace('_Attack', ''), eps=lvl,
                        iters=ATTACK_ITERS, num_restarts=ATTACK_RESTARTS
                    )
                else:
                    bx_p_eval = torch.tensor(X_pcap_eval, dtype=torch.float32)

                probs_b1, ms_b1 = predict_tensor_model(m_pcap, bx_p_eval)
                preds_b1 = np.argmax(probs_b1, axis=1)
                if st_type != 'Edge_Perturbation':
                    record_evaluation_metrics(preds_b1, probs_b1, "Branch 1: PCAP MLP", t_pcap_v, ms_b1, mem_pcap_v, st_type, lvl, "tabular_inductive")

                # 2. SAR Autoencoder
                if st_type in ['Gaussian_Noise', 'Feature_Masking']:
                    bx_s_eval = apply_tabular_noise(X_sar_eval, st_type, lvl, seed=seed)
                elif st_type in ['FGSM_Attack', 'BIM_Attack', 'PGD_Attack']:
                    bx_s_eval = attack_tensor_model_rectified(
                        m_sar, torch.tensor(X_sar_eval, dtype=torch.float32),
                        torch.tensor(y_te, dtype=torch.long),
                        attack_type=st_type.replace('_Attack', ''), eps=lvl,
                        iters=ATTACK_ITERS, num_restarts=ATTACK_RESTARTS
                    )
                else:
                    bx_s_eval = torch.tensor(X_sar_eval, dtype=torch.float32)

                probs_b2, ms_b2 = predict_tensor_model(m_sar, bx_s_eval)
                preds_b2 = np.argmax(probs_b2, axis=1)
                if st_type != 'Edge_Perturbation':
                    record_evaluation_metrics(preds_b2, probs_b2, "Branch 2: SAR Encoder", t_sar_v, ms_b2, mem_sar_v, st_type, lvl, "tabular_inductive")

                # 3. STRACE CNN-LSTM
                if st_type in ['Gaussian_Noise', 'Feature_Masking']:
                    bx_st_eval = apply_tabular_noise(X_sys_eval, st_type, lvl, seed=seed)
                elif st_type in ['FGSM_Attack', 'BIM_Attack', 'PGD_Attack']:
                    bx_st_eval = attack_tensor_model_rectified(
                        m_sys, torch.tensor(X_sys_eval, dtype=torch.float32),
                        torch.tensor(y_te, dtype=torch.long),
                        attack_type=st_type.replace('_Attack', ''), eps=lvl,
                        iters=ATTACK_ITERS, num_restarts=ATTACK_RESTARTS
                    )
                else:
                    bx_st_eval = torch.tensor(X_sys_eval, dtype=torch.float32)

                probs_b3, ms_b3 = predict_tensor_model(m_sys, bx_st_eval)
                preds_b3 = np.argmax(probs_b3, axis=1)
                if st_type != 'Edge_Perturbation':
                    record_evaluation_metrics(preds_b3, probs_b3, "Branch 3: STRACE CNN-LSTM", t_sys_v, ms_b3, mem_sys_v, st_type, lvl, "tabular_inductive")

                # 4. Multimodal Late Fusion
                if st_type in ['Gaussian_Noise', 'Feature_Masking']:
                    bx_p_f = apply_tabular_noise(X_pcap_eval, st_type, lvl, seed=seed)
                    bx_s_f = apply_tabular_noise(X_sar_eval, st_type, lvl, seed=seed)
                    bx_st_f = apply_tabular_noise(X_sys_eval, st_type, lvl, seed=seed)
                elif st_type in ['FGSM_Attack', 'BIM_Attack', 'PGD_Attack']:
                    bx_p_f, bx_s_f, bx_st_f = attack_multimodal_fusion_model_rectified(
                        m_fusion,
                        torch.tensor(X_pcap_eval, dtype=torch.float32),
                        torch.tensor(X_sar_eval, dtype=torch.float32),
                        torch.tensor(X_sys_eval, dtype=torch.float32),
                        torch.tensor(y_te, dtype=torch.long),
                        attack_type=st_type.replace('_Attack', ''),
                        eps=lvl, iters=ATTACK_ITERS, num_restarts=ATTACK_RESTARTS
                    )
                else:
                    bx_p_f, bx_s_f, bx_st_f = X_pcap_eval, X_sar_eval, X_sys_eval

                probs_fusion, ms_fusion = predict_fusion_model(m_fusion, bx_p_f, bx_s_f, bx_st_f)
                preds_fusion = np.argmax(probs_fusion, axis=1)
                if st_type != 'Edge_Perturbation':
                    record_evaluation_metrics(preds_fusion, probs_fusion, "Multimodal Late Fusion", t_fusion_v, ms_fusion, mem_fusion_v, st_type, lvl, "tabular_inductive")

                # 5. Multimodal Non-HIN GNN (Strict Inductive Self-Loops)
                set_eval_mode_safe(model_non_hin)
                if st_type == 'Clean':
                    g_eval_nh = g_homo_te_clean.to(device)
                elif st_type in ['Gaussian_Noise', 'Feature_Masking', 'Edge_Perturbation']:
                    g_eval_nh = apply_noise_to_homo_graph(g_homo_te_clean, st_type, lvl, seed=seed).to(device)
                elif st_type == 'FGSM_Attack':
                    g_eval_nh = attack_homo_gnn_model_rectified(model_non_hin, g_homo_te_clean, attack_type='FGSM', eps=lvl)
                elif st_type == 'BIM_Attack':
                    g_eval_nh = attack_homo_gnn_model_rectified(model_non_hin, g_homo_te_clean, attack_type='BIM', eps=lvl, iters=ATTACK_ITERS)
                elif st_type == 'PGD_Attack':
                    g_eval_nh = attack_homo_gnn_model_rectified(model_non_hin, g_homo_te_clean, attack_type='PGD', eps=lvl, iters=ATTACK_ITERS, num_restarts=ATTACK_RESTARTS)

                g_eval_nh = g_eval_nh.to(device)
                if torch.cuda.is_available(): torch.cuda.synchronize()
                t_inf_start_nh = time.perf_counter()

                with torch.no_grad():
                    out_nh = model_non_hin(g_eval_nh.x, g_eval_nh.edge_index)
                    probs_nh = F.softmax(out_nh, dim=1).cpu().numpy()
                    preds_nh = np.argmax(probs_nh, axis=1)

                if torch.cuda.is_available(): torch.cuda.synchronize()
                ms_nh = ((time.perf_counter() - t_inf_start_nh) / max(len(preds_nh), 1)) * 1000.0
                record_evaluation_metrics(preds_nh, probs_nh, "Multimodal Non-HIN GNN", t_tr_non_hin, ms_nh, mem_non_hin, st_type, lvl, "self_loop_inductive")

                # 6 - 18. 13 HIN GNNs
                for m_name, m_info in trained_hin_models.items():
                    # [v3 / J2] Clean always runs for ALL configs; noise and attacks only run
                    # for the selected set (empty set = all).
                    if ATTACK_CONFIGS and st_type != 'Clean' and m_name not in ATTACK_CONFIGS:
                        continue
                    hin_m = m_info['model']
                    t_tr_h = m_info['train_time']
                    mem_h = m_info['peak_mem']

                    set_eval_mode_safe(hin_m)

                    if m_info.get('eval_mode') == 'strict':
                        if st_type == 'Clean':
                            probs_h, ms_h = predict_hin_isolated_batch(hin_m, ego_graphs_test)
                        elif st_type in ['Gaussian_Noise', 'Feature_Masking']:
                            ego_noisy = perturb_isolated_ego_graphs(ego_graphs_test, st_type, lvl, seed)
                            probs_h, ms_h = predict_hin_isolated_batch(hin_m, ego_noisy)
                        elif st_type == 'Edge_Perturbation':
                            ego_noisy = perturb_isolated_ego_graphs(ego_graphs_test, st_type, lvl, seed)
                            probs_h, ms_h = predict_hin_isolated_batch(hin_m, ego_noisy)
                        elif st_type in ['FGSM_Attack', 'BIM_Attack', 'PGD_Attack']:
                            probs_h, ms_h = attack_isolated_ego_graphs_batch(
                                hin_m, ego_graphs_test, attack_type=st_type.replace('_Attack', ''), eps=lvl,
                                iters=ATTACK_ITERS, num_restarts=ATTACK_RESTARTS
                            )
                        preds_h = np.argmax(probs_h, axis=1)
                        record_evaluation_metrics(preds_h, probs_h, m_name, t_tr_h, ms_h, mem_h, st_type, lvl, 'strict_isolated_ego')

                    else:
                        if st_type == 'Clean':
                            g_eval_hin = g_hin_te_clean.to(device)
                        elif st_type in ['Gaussian_Noise', 'Feature_Masking', 'Edge_Perturbation']:
                            g_eval_hin = apply_noise_to_hetero_graph(g_hin_te_clean, st_type, lvl, seed=seed).to(device)
                        elif st_type == 'FGSM_Attack':
                            g_eval_hin = attack_hetero_gnn_model_rectified(hin_m, g_hin_te_clean, attack_type='FGSM', eps=lvl)
                        elif st_type == 'BIM_Attack':
                            g_eval_hin = attack_hetero_gnn_model_rectified(hin_m, g_hin_te_clean, attack_type='BIM', eps=lvl, iters=ATTACK_ITERS)
                        elif st_type == 'PGD_Attack':
                            g_eval_hin = attack_hetero_gnn_model_rectified(hin_m, g_hin_te_clean, attack_type='PGD', eps=lvl, iters=ATTACK_ITERS, num_restarts=ATTACK_RESTARTS)

                        g_eval_hin = g_eval_hin.to(device)
                        if torch.cuda.is_available(): torch.cuda.synchronize()
                        t_inf_start = time.perf_counter()

                        with torch.no_grad():
                            logits_h = hin_m(g_eval_hin.x_dict, g_eval_hin.edge_index_dict)
                            probs_h = F.softmax(logits_h, dim=1).cpu().numpy()
                            preds_h = np.argmax(probs_h, axis=1)

                        if torch.cuda.is_available(): torch.cuda.synchronize()
                        ms_h = ((time.perf_counter() - t_inf_start) / max(len(preds_h), 1)) * 1000.0

                        record_evaluation_metrics(preds_h, probs_h, m_name, t_tr_h, ms_h, mem_h, st_type, lvl, 'shared_entity_ablation')

        pd.DataFrame(robustness_global_records).to_csv(raw_glob_path, index=False)
        pd.DataFrame(robustness_arch_records).to_csv(raw_arch_path, index=False)
        print(f"💾 [SEED {seed}] Checkpoint saved & hardware metrics flushed to disk.")

        # [v3 / I4] Time budget for the main loop.
        #
        # Noise/attack sessions run here and are hundreds of times more expensive than clean
        # ones: PGD iterates 20 steps × 5 restarts × 4 epsilon levels, multiplied by the number
        # of configs and test samples. Kaggle kills the session at the 12-hour mark and an overrunning
        # "Save & Run All" saves NOTHING, so the pipeline must stop by itself beforehand.
        _now_m = time.time()
        _MAIN_TIMES.append(_now_m - (_MAIN_LAST[0] if _MAIN_LAST[0] is not None else _RUN_T0))
        _MAIN_LAST[0] = _now_m
        if _budget_exceeded(_now_m, f"next seed after {seed}", _MAIN_TIMES):
            _BUDGET_HIT[0] = True
            print(f"⏹️  Main loop stopped by the time budget after seed {seed}.")
            print(f"   Wrote {len(robustness_global_records)} global records and "
                  f"{len(robustness_arch_records)} per-architecture records.")
            print("   Next session: attach this output as input and set RHGT_RESUME=1; "
                  "the pipeline will continue with the remaining seeds.")
            break

        # [v2 / FIX C1] Drop the references HERE; the helper function cannot do this.
        del m_pcap, m_sar, m_sys, m_fusion, model_non_hin
        del g_hin_tr, g_hin_te_clean, g_homo_tr, g_homo_te_clean
        del trained_hin_models, ego_graphs_train, ego_graphs_test
        del trained_pcap, trained_sar, trained_sys, fusion_model
        clean_gpu_memory()

    # Export Final Summaries  [v2: guard for the case of no records yet]
    df_rob_glob = pd.DataFrame(robustness_global_records)
    if len(df_rob_glob) > 0:
        df_rob_glob = df_rob_glob.drop_duplicates(
            subset=[c for c in ['Config', 'Run', 'Architecture', 'Noise_Type', 'Noise_Level']
                    if c in df_rob_glob.columns], keep='last')
        df_rob_glob.to_csv(raw_glob_path, index=False)
        summary_rob_glob = format_summary_dataframe(df_rob_glob, ['Noise_Type', 'Noise_Level', 'Config', 'Inference_Regime'], all_eval_metrics_cols)
        summary_rob_glob.to_csv(os.path.join(ROBUSTNESS_GLOBAL_DIR, "overall_robustness_summary_10seed.csv"), index=False)

    df_rob_arch = pd.DataFrame(robustness_arch_records)
    if len(df_rob_arch) > 0:
        df_rob_arch = df_rob_arch.drop_duplicates(
            subset=[c for c in ['Config', 'Run', 'Architecture', 'Noise_Type', 'Noise_Level']
                    if c in df_rob_arch.columns], keep='last')
        df_rob_arch.to_csv(raw_arch_path, index=False)
        summary_rob_arch = format_summary_dataframe(df_rob_arch, ['Architecture', 'Noise_Type', 'Noise_Level', 'Config', 'Inference_Regime'], all_eval_metrics_cols)
        summary_rob_arch.to_csv(os.path.join(ROBUSTNESS_ARCH_DIR, "per_architecture_robustness_summary_10seed.csv"), index=False)

    print("\n✅ SCENARIOS 1 & 2 (GENUINE STRICT INDUCTIVE ROBUSTNESS BENCHMARK) COMPLETED & SAVED!")

    # =====================================================================================
    # SCENARIO 3: ZERO-DAY GENERALIZATION (LOFO) WITH RECTIFIED ROBUST INDUCTIVE TRAINING
    # =====================================================================================
    print("\n" + "=" * 100)
    print("🛡️ SCENARIO 3: ZERO-DAY GENERALIZATION (LOFO) WITH STRICT-INDUCTIVE ALIGNMENT FINE-TUNING (10 SEEDS)")
    print("=" * 100)

    if 'MalwareFamily' in df_clean.columns:
        df_malware_all = df_clean[df_clean['label'] == 1].copy()
        top_families = df_malware_all['MalwareFamily'].value_counts().head(5).index.tolist()
    else:
        top_families = ['Mirai', 'Gafgyt', 'Tsunami', 'Hajime', 'Mozi']

    # [v2] Keep only families with enough samples. On CIC-YNU-IoTMal only Mirai (7368 positive
    # samples) and DarkNexus (219) have enough statistical power; Gafgyt (60), Generic (47)
    # and especially Tsunami (3 samples → Recall takes only 3 possible values) give results that are
    # sampling noise rather than measurements. Dropping these three families also saves ~60% of
    # the LOFO scenario time.
    if LOFO_FAMILIES_ENV:
        _want = [f.strip() for f in LOFO_FAMILIES_ENV.split(',') if f.strip()]
        _avail = set(df_clean['MalwareFamily'].astype(str)) if 'MalwareFamily' in df_clean.columns else set()
        _missing = [f for f in _want if f not in _avail]
        if _missing:
            print(f"[!] LOFO families not present in the data, skipped: {_missing}")
        top_families = [f for f in _want if f in _avail]
        print(f"🎯 LOFO families (manually selected): {top_families}")

    lofo_records = []
    raw_lofo_path = os.path.join(LOFO_DIR, "lofo_zeroday_generalization_raw_10seed.csv")
    summary_lofo_path = os.path.join(LOFO_DIR, "lofo_zeroday_generalization_summary_10seed.csv")

    # ===================== RESUME v2 (LOFO) =====================
    # A (seed, family) pair is DONE only when it has all requested configs - if a later
    # session widens RHGT_EXTRA, the pair is re-run to fill in the missing ones.
    _done_lofo = set()
    _req_cfgs_lofo = {s['name'] for s in hin_model_specs}
    if RESUME and os.path.exists(raw_lofo_path):
        try:
            _prev_l = pd.read_csv(raw_lofo_path)
            lofo_records.extend(_prev_l.to_dict('records'))
            _seen_l = {}
            for _r in lofo_records:
                if pd.isna(_r.get('Seed')) or pd.isna(_r.get('Held_Out_Family')):
                    continue
                _seen_l.setdefault((int(_r['Seed']), str(_r['Held_Out_Family'])),
                                   set()).add(str(_r.get('Config')))
            _done_lofo = {k for k, v in _seen_l.items() if _req_cfgs_lofo.issubset(v)}
            print(f"♻️  Reloaded {len(_prev_l)} LOFO records; "
                  f"{len(_done_lofo)}/{len(_seen_l)} (seed, family) pairs have all configs.")
        except Exception as _e:
            print(f"[!] Could not read {raw_lofo_path}: {_e}")

    for seed in ([] if SKIP_LOFO else SEEDS):
        print(f"\n🌱 [LOFO SEED {seed}/10] Initializing Group-Safe Zero-Day Split...")
        seed_everything(seed)

        df_malware = df_clean[df_clean['label'] == 1].copy()
        df_benign = df_clean[df_clean['label'] == 0].copy()

        df_benign_tr, df_benign_te = group_safe_stratified_split(df_benign, group_col='Hash', label_col='label', test_size=0.2, seed=seed)

        for held_family in top_families:
            if RESUME and (int(seed), str(held_family)) in _done_lofo:      # [v2]
                print(f"  ♻️  [Seed {seed} | {held_family}] already has results - skipping.")
                continue
            _now = time.time()                                              # [v3 / I2]
            if _LOFO_LAST[0] is not None:
                _ITER_TIMES.append(_now - _LOFO_LAST[0])
                # write right after EACH (seed, family) pair instead of waiting for the end of the seed
                pd.DataFrame(lofo_records).to_csv(raw_lofo_path, index=False)
            if _budget_exceeded(_now, f"seed {seed} / family {held_family}", _ITER_TIMES):
                _BUDGET_HIT[0] = True
                break
            _LOFO_LAST[0] = _now
            print(f"  🎯 [Seed {seed} | LOFO] Isolating Family '{held_family}' as Zero-Day...")

            mask_held = (df_malware['MalwareFamily'].astype(str) == str(held_family))
            df_malware_tr = df_malware[~mask_held].copy()
            df_malware_te = df_malware[mask_held].copy()

            df_lofo_tr = pd.concat([df_malware_tr, df_benign_tr], ignore_index=True)

            # [v3] The original forced prevalence to ~50% via min(...), so Precision and
            # PR-AUC did not transfer to the real base rate. By default v3 KEEPS the
            # whole benign test set; set RHGT_LOFO_BALANCE=1 to return to the old
            # behaviour when a comparison with earlier results is needed.
            if os.environ.get('RHGT_LOFO_BALANCE', '0') == '1':
                n_benign_te_sample = min(len(df_malware_te), len(df_benign_te))
                df_lofo_te_benign = df_benign_te.sample(
                    n=n_benign_te_sample, random_state=seed).copy()
            else:
                df_lofo_te_benign = df_benign_te.copy()
            _n_mal, _n_ben = len(df_malware_te), len(df_lofo_te_benign)
            _prev = _n_mal / max(_n_mal + _n_ben, 1)
            print(f"      ↳ LOFO test set: {_n_mal} malware + {_n_ben} benign "
                  f"| prevalence = {_prev:.4f} "
                  f"| F1 of an always-predict-malware classifier = {2*_prev/(1+_prev):.4f}")
            df_lofo_te = pd.concat([df_malware_te, df_lofo_te_benign], ignore_index=True)

            y_tr_lofo = df_lofo_tr['label'].values
            y_te_lofo = df_lofo_te['label'].values
            held_malware_mask = (y_te_lofo == 1)

            # Preprocessing
            x_tr_pcap = clean_numeric_array(df_lofo_tr, pcap_cols)
            x_te_pcap = clean_numeric_array(df_lofo_te, pcap_cols)
            sc_pcap_l = StandardScaler().fit(x_tr_pcap)
            x_tr_pcap_s = sc_pcap_l.transform(x_tr_pcap)
            x_te_pcap_s = sc_pcap_l.transform(x_te_pcap)

            x_tr_sar = clean_numeric_array(df_lofo_tr, sar_cols)
            x_te_sar = clean_numeric_array(df_lofo_te, sar_cols)
            sc_sar_l = StandardScaler().fit(x_tr_sar)
            x_tr_sar_s = sc_sar_l.transform(x_tr_sar)
            x_te_sar_s = sc_sar_l.transform(x_te_sar)

            x_tr_sys = clean_numeric_array(df_lofo_tr, syscall_cols)
            x_te_sys = clean_numeric_array(df_lofo_te, syscall_cols)
            sc_sys_l = StandardScaler().fit(x_tr_sys)
            x_tr_sys_s = sc_sys_l.transform(x_tr_sys)
            x_te_sys_s = sc_sys_l.transform(x_te_sys)

            run_id = f"S{seed}_LOFO_{held_family}"

            # 1. Branch 1: PCAP MLP
            pcap_m_lofo = PCAP_MLP(in_dim=len(pcap_cols))
            trained_pcap_l, t_tr_pcap_l, mem_pcap_l = train_neural_branch(pcap_m_lofo, x_tr_pcap_s, y_tr_lofo)
            bx_p = torch.tensor(x_te_pcap_s, dtype=torch.float32).to(device)
            probs_p, infer_ms_p_l = predict_tensor_model(trained_pcap_l, bx_p)
            preds_p = np.argmax(probs_p, axis=1)

            res_pcap = evaluate_metrics(y_te_lofo, preds_p, probs_p, "Branch 1 Only (PCAP MLP)", run_id,
                                        train_time_sec=t_tr_pcap_l, infer_ms_sample=infer_ms_p_l, peak_gpu_mem_mb=mem_pcap_l)
            res_pcap.update({
                'Held_Out_Family': held_family, 'Seed': seed,
                # [v2 / FIX C2] Missing this key made groupby(dropna=True) SILENTLY
                # drop all non-HIN baselines from the LOFO summary table.
                'Inference_Regime': 'tabular_inductive',
                'Shared_Weights_With': '',
                'ZeroDay_Detection_Rate (Recall)': (preds_p[held_malware_mask] == 1).mean() if held_malware_mask.sum() > 0 else 0.0
            })
            lofo_records.append(res_pcap)

            # 2. Branch 2: SAR Encoder
            sar_m_lofo = SAR_Autoencoder(in_dim=len(sar_cols))
            trained_sar_l, t_tr_sar_l, mem_sar_l = train_neural_branch(sar_m_lofo, x_tr_sar_s, y_tr_lofo)
            bx_s = torch.tensor(x_te_sar_s, dtype=torch.float32).to(device)
            probs_s, infer_ms_s_l = predict_tensor_model(trained_sar_l, bx_s)
            preds_s = np.argmax(probs_s, axis=1)

            res_sar = evaluate_metrics(y_te_lofo, preds_s, probs_s, "Branch 2 Only (SAR Encoder)", run_id,
                                       train_time_sec=t_tr_sar_l, infer_ms_sample=infer_ms_s_l, peak_gpu_mem_mb=mem_sar_l)
            res_sar.update({
                'Held_Out_Family': held_family, 'Seed': seed,
                # [v2 / FIX C2] Missing this key made groupby(dropna=True) SILENTLY
                # drop all non-HIN baselines from the LOFO summary table.
                'Inference_Regime': 'tabular_inductive',
                'Shared_Weights_With': '',
                'ZeroDay_Detection_Rate (Recall)': (preds_s[held_malware_mask] == 1).mean() if held_malware_mask.sum() > 0 else 0.0
            })
            lofo_records.append(res_sar)

            # 3. Branch 3: STRACE CNN-LSTM
            sys_m_lofo = STRACE_CNN_LSTM(in_dim=len(syscall_cols))
            trained_sys_l, t_tr_sys_l, mem_sys_l = train_neural_branch(sys_m_lofo, x_tr_sys_s, y_tr_lofo)
            bx_st = torch.tensor(x_te_sys_s, dtype=torch.float32).to(device)
            probs_st, infer_ms_st_l = predict_tensor_model(trained_sys_l, bx_st)
            preds_st = np.argmax(probs_st, axis=1)

            res_sys = evaluate_metrics(y_te_lofo, preds_st, probs_st, "Branch 3 Only (STRACE CNN-LSTM)", run_id,
                                       train_time_sec=t_tr_sys_l, infer_ms_sample=infer_ms_st_l, peak_gpu_mem_mb=mem_sys_l)
            res_sys.update({
                'Held_Out_Family': held_family, 'Seed': seed,
                # [v2 / FIX C2] Missing this key made groupby(dropna=True) SILENTLY
                # drop all non-HIN baselines from the LOFO summary table.
                'Inference_Regime': 'tabular_inductive',
                'Shared_Weights_With': '',
                'ZeroDay_Detection_Rate (Recall)': (preds_st[held_malware_mask] == 1).mean() if held_malware_mask.sum() > 0 else 0.0
            })
            lofo_records.append(res_sys)

            # 4. Multimodal Late Fusion
            fusion_m_lofo = MultimodalLateFusionModel(pcap_dim=len(pcap_cols), sar_dim=len(sar_cols), strace_dim=len(syscall_cols)).to(device)
            fusion_m_lofo.load_pretrained_branches(trained_pcap_l, trained_sar_l, trained_sys_l)
            fusion_m_lofo, t_tr_f_l, mem_f_l = train_end_to_end_joint_fine_tuning(fusion_m_lofo, x_tr_pcap_s, x_tr_sar_s, x_tr_sys_s, y_tr_lofo)
            probs_f, infer_ms_f_l = predict_fusion_model(fusion_m_lofo, bx_p, bx_s, bx_st)
            preds_f = np.argmax(probs_f, axis=1)

            res_f = evaluate_metrics(y_te_lofo, preds_f, probs_f, "Multimodal Late Fusion (E2E)", run_id,
                                     train_time_sec=t_tr_f_l, infer_ms_sample=infer_ms_f_l, peak_gpu_mem_mb=mem_f_l)
            res_f.update({
                'Held_Out_Family': held_family, 'Seed': seed,
                # [v2 / FIX C2] Missing this key made groupby(dropna=True) SILENTLY
                # drop all non-HIN baselines from the LOFO summary table.
                'Inference_Regime': 'tabular_inductive',
                'Shared_Weights_With': '',
                'ZeroDay_Detection_Rate (Recall)': (preds_f[held_malware_mask] == 1).mean() if held_malware_mask.sum() > 0 else 0.0
            })
            lofo_records.append(res_f)

            # 5. Multimodal Non-HIN GNN (Inductive Zero-Leakage)
            g_homo_tr_l, scaler_homo_l = build_homogeneous_multimodal_graph(df_lofo_tr, all_feature_cols, k=5, mode='train')
            g_homo_te_l, _ = build_homogeneous_multimodal_graph(df_lofo_te, all_feature_cols, k=5, scaler=scaler_homo_l, mode='test')
            g_homo_tr_l = g_homo_tr_l.to(device)

            m_non_hin_l = Multimodal_NonHIN_GNN(in_dim=g_homo_tr_l.x.size(1), hidden_dim=64, out_dim=2).to(device)
            opt_non_l = torch.optim.Adam(m_non_hin_l.parameters(), lr=0.01, weight_decay=5e-4)

            if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
            t_start_non_l = time.perf_counter()

            m_non_hin_l.train()
            for epoch in range(100):
                opt_non_l.zero_grad()
                out = m_non_hin_l(g_homo_tr_l.x, g_homo_tr_l.edge_index)
                loss = F.cross_entropy(out, g_homo_tr_l.y)
                loss.backward()
                opt_non_l.step()

            if torch.cuda.is_available(): torch.cuda.synchronize()
            t_tr_non_l = time.perf_counter() - t_start_non_l
            mem_non_l = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0

            m_non_hin_l.eval()
            g_homo_te_l = g_homo_te_l.to(device)

            if torch.cuda.is_available(): torch.cuda.synchronize()
            t_start_inf = time.perf_counter()

            with torch.no_grad():
                out_non = m_non_hin_l(g_homo_te_l.x, g_homo_te_l.edge_index)
                probs_non = F.softmax(out_non, dim=1).cpu().numpy()
                preds_non = out_non.argmax(dim=1).cpu().numpy()

            if torch.cuda.is_available(): torch.cuda.synchronize()
            infer_ms_non_l = ((time.perf_counter() - t_start_inf) / max(len(y_te_lofo), 1)) * 1000.0

            res_non = evaluate_metrics(y_te_lofo, preds_non, probs_non, "Multimodal Non-HIN GNN (Res-GraphSAGE)", run_id,
                                       train_time_sec=t_tr_non_l, infer_ms_sample=infer_ms_non_l, peak_gpu_mem_mb=mem_non_l)
            res_non.update({
                'Held_Out_Family': held_family, 'Seed': seed,
                # [v2 / FIX C2] Missing this key made groupby(dropna=True) SILENTLY
                # drop all non-HIN baselines from the LOFO summary table.
                'Inference_Regime': 'self_loop_inductive',
                'Shared_Weights_With': '',
                'ZeroDay_Detection_Rate (Recall)': (preds_non[held_malware_mask] == 1).mean() if held_malware_mask.sum() > 0 else 0.0
            })
            lofo_records.append(res_non)

            # 6 - 18. 13 HIN GNNs with Rectified Robust Training & Isolated Inductive Inference
            g_tr_l, scaler_g_l, arch_le_g_l = build_hetero_graph_hin(df_lofo_tr, num_cols, syscall_cols, proto_list, mode='train', k_sim=3)
            g_te_shared_l, _, _ = build_hetero_graph_hin(
                df_lofo_te, num_cols, syscall_cols, proto_list,
                scaler=scaler_g_l, arch_le=arch_le_g_l, mode='test', k_sim=3
            )
            g_tr_l = g_tr_l.to(device)
            metadata_l = g_tr_l.metadata()

            canonical_anchors_l = RectifiedStrictInductiveEngine.extract_canonical_anchors(g_tr_l)
            ego_graphs_lofo_tr = RectifiedStrictInductiveEngine.build_isolated_ego_graphs_batch(
                df_lofo_tr, num_cols, syscall_cols, proto_list, scaler_g_l, arch_le_g_l, canonical_anchors_l
            )
            ego_graphs_lofo_te = RectifiedStrictInductiveEngine.build_isolated_ego_graphs_batch(
                df_lofo_te, num_cols, syscall_cols, proto_list, scaler_g_l, arch_le_g_l, canonical_anchors_l
            )

            trained_lofo_hin = {}
            for spec in hin_model_specs:
                model_name = spec['name']

                if 'alias_of' in spec:
                    base_name = spec['alias_of']
                    if base_name not in trained_lofo_hin:
                        raise RuntimeError(f"LOFO alias base model not trained yet: {base_name}")
                    base = trained_lofo_hin[base_name]
                    trained_lofo_hin[model_name] = {
                        'model': base['model'], 'train_time': base['train_time'], 'peak_mem': base['peak_mem'],
                        'eval_mode': spec.get('eval_mode', 'strict'), 'shared_weights_with': base_name,
                        'initialized_from': base.get('initialized_from'), 'train_mode': base.get('train_mode', 'alias')
                    }
                elif 'clone_of' in spec:
                    base_name = spec['clone_of']
                    if base_name not in trained_lofo_hin:
                        raise RuntimeError(f"LOFO fine-tune base model not trained yet: {base_name}")
                    base = trained_lofo_hin[base_name]
                    m_hin_l = copy.deepcopy(base['model']).to(device)
                    print(f"         🔁 LOFO strict fine-tuning from: {base_name}")
                    m_hin_l, t_ft_l, mem_ft_l = fine_tune_strict_inductive_hin(
                        m_hin_l, ego_graphs_lofo_tr,
                        epochs=STRICT_FT_EPOCHS, lr=STRICT_FT_LR,
                        weight_decay=1e-4, adv_eps=STRICT_FT_ADV_EPS,
                        clean_weight=STRICT_FT_CLEAN_WEIGHT,
                        batch_size=STRICT_FT_BATCH_SIZE, seed=seed,
                        use_class_weight=(spec.get('train_mode') != 'strict_finetune_noweight'),  # [v2]
                    )
                    trained_lofo_hin[model_name] = {
                        'model': m_hin_l,
                        'train_time': base['train_time'] + t_ft_l,
                        'peak_mem': max(base['peak_mem'], mem_ft_l),
                        'eval_mode': spec.get('eval_mode', 'strict'),
                        'shared_weights_with': None,
                        'initialized_from': base_name,
                        'train_mode': 'strict_finetune'
                    }
                else:
                    m_hin_l = instantiate_hin_model(spec, metadata_l)
                    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
                    t_start_hin_l = time.perf_counter()

                    if spec.get('train_mode') in ('robust', 'robust_weighted'):
                        m_hin_l, _, _ = train_robust_hin_gnn(
                            m_hin_l, g_tr_l, epochs=100, lr=0.005, weight_decay=5e-4, adv_eps=0.05,
                            class_weight=(spec.get('train_mode') == 'robust_weighted')   # [v2]
                        )
                    else:
                        m_hin_l, _, _ = train_standard_hin_gnn(
                            m_hin_l, g_tr_l, epochs=100, lr=0.005, weight_decay=5e-4
                        )

                    if torch.cuda.is_available(): torch.cuda.synchronize()
                    t_tr_hin_l = time.perf_counter() - t_start_hin_l
                    mem_hin_l = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
                    trained_lofo_hin[model_name] = {
                        'model': m_hin_l, 'train_time': t_tr_hin_l, 'peak_mem': mem_hin_l,
                        'eval_mode': spec.get('eval_mode', 'strict'), 'shared_weights_with': None,
                        'initialized_from': None, 'train_mode': spec.get('train_mode', 'standard')
                    }

                info = trained_lofo_hin[model_name]
                m_hin_l = info['model']
                m_hin_l.eval()

                if info.get('eval_mode') == 'strict':
                    probs_hin, infer_ms_hin_l = predict_hin_isolated_batch(m_hin_l, ego_graphs_lofo_te)
                    inference_regime = 'strict_isolated_ego'
                else:
                    g_te_eval = g_te_shared_l.to(device)
                    if torch.cuda.is_available(): torch.cuda.synchronize()
                    t_inf = time.perf_counter()
                    with torch.no_grad():
                        out_hin = m_hin_l(g_te_eval.x_dict, g_te_eval.edge_index_dict)
                        probs_hin = F.softmax(out_hin, dim=1).cpu().numpy()
                    if torch.cuda.is_available(): torch.cuda.synchronize()
                    infer_ms_hin_l = ((time.perf_counter() - t_inf) / max(len(y_te_lofo), 1)) * 1000.0
                    inference_regime = 'shared_entity_ablation'

                preds_hin = np.argmax(probs_hin, axis=1)
                res_hin = evaluate_metrics(
                    y_te_lofo, preds_hin, probs_hin, model_name, run_id,
                    train_time_sec=info['train_time'], infer_ms_sample=infer_ms_hin_l, peak_gpu_mem_mb=info['peak_mem']
                )
                res_hin.update({
                    'Held_Out_Family': held_family, 'Seed': seed,
                    'Inference_Regime': inference_regime,
                    'Shared_Weights_With': info.get('shared_weights_with') or '',
                    'ZeroDay_Detection_Rate (Recall)': (preds_hin[held_malware_mask] == 1).mean() if held_malware_mask.sum() > 0 else 0.0
                })
                lofo_records.append(res_hin)

            # [v2 / FIX C1]
            del trained_pcap_l, trained_sar_l, trained_sys_l, fusion_m_lofo, m_non_hin_l
            del g_tr_l, g_te_shared_l, g_homo_tr_l, g_homo_te_l
            del trained_lofo_hin, ego_graphs_lofo_tr, ego_graphs_lofo_te
            clean_gpu_memory()

        pd.DataFrame(lofo_records).to_csv(raw_lofo_path, index=False)
        print(f"💾 [LOFO SEED {seed}] Flushed raw zero-day results to disk.")
        if _BUDGET_HIT[0]:                                                  # [v3 / I3]
            _done_now = len({(r.get('Seed'), r.get('Held_Out_Family'))
                             for r in lofo_records})
            print(f"⏹️  LOFO stopped by the time budget after {_done_now} pairs "
                  f"(seed, family). Results written to {os.path.basename(raw_lofo_path)}.")
            print("   Next session: attach this output as input and set RHGT_RESUME=1; "
                  "the pipeline will resume where it stopped.")
            break

    df_lofo_raw = pd.DataFrame(lofo_records)

    if len(df_lofo_raw) == 0:
        # [v3 / H1] No LOFO records at all - usually because RHGT_SKIP_LOFO=1.
        #
        # BEFORE v3, the block below still ran on an EMPTY DataFrame and raised
        #     KeyError: 'Held_Out_Family'
        # right after Scenarios 1&2 had finished 10 seeds (≈4.3 GPU hours). The
        # main results were still safe because they had been flushed earlier, but the session
        # still ended with a traceback, so Kaggle marked the notebook as FAILED.
        #
        # Worse: df_lofo_raw.to_csv() would OVERWRITE the previous session's LOFO file with
        # an empty file when run with RESUME=0. Now it exits early and writes nothing.
        print("\n⏭️  [v3] Skipping LOFO aggregation: no records "
              f"(RHGT_SKIP_LOFO={'1' if SKIP_LOFO else '0'}, "
              f"RESUME={'1' if RESUME else '0'}).")
        print(f"   NOT overwriting {os.path.basename(raw_lofo_path)} / "
              f"{os.path.basename(summary_lofo_path)}.")
        print("\n✅ DONE: all requested scenarios completed in this session.")
        return

    df_lofo_raw = df_lofo_raw.drop_duplicates(                      # [v2]
        subset=[c for c in ['Config', 'Seed', 'Held_Out_Family', 'Architecture']
                if c in df_lofo_raw.columns], keep='last')
    df_lofo_raw.to_csv(raw_lofo_path, index=False)

    lofo_eval_cols = all_eval_metrics_cols + ['ZeroDay_Detection_Rate (Recall)']
    _miss_grp = [c for c in ['Held_Out_Family', 'Config', 'Inference_Regime']
                 if c not in df_lofo_raw.columns]
    if _miss_grp:
        # [v3 / H2] Defensive: raw LOFO files from older versions may lack the grouping columns.
        print(f"[!] Skipping LOFO aggregation: raw file is missing columns {_miss_grp}. "
              f"The raw file is left unchanged.")
    else:
        summary_lofo = format_summary_dataframe(df_lofo_raw, ['Held_Out_Family', 'Config', 'Inference_Regime'], lofo_eval_cols)
        summary_lofo.to_csv(summary_lofo_path, index=False)

    print("\n✅ ALL ADVANCED EXPERIMENTS (GENUINE STRICT INDUCTIVE SOTA 10 SEEDS) COMPLETED SUCCESSFULLY!")

if __name__ == '__main__':
    main()

