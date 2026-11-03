# DynHIN — Dynamic Heterogeneous Behavior Graphs for Cross-Architecture IoT Malware Detection

Code and per-seed results for the paper

> **Dynamic Heterogeneous Behavior Graph Learning for Cross-Architecture IoT Malware Detection under Strict-Inductive Inference**
> Dac-Tot Tran, Anh-Tu Tran, Khuong Nguyen-An, The-Dung Luong

The paper builds a dynamic heterogeneous information network (HIN) from the PCAP, SAR and STRACE
layers of the CIC-YNU-IoTMal dataset (ARM, MIPS, MIPSEL, x86; 17,023 samples). The graph has
`file`, `syscall`, `arch` and `protocol` nodes. A reliability-aware heterogeneous graph transformer
(R-HGT) and six other HIN encoders are evaluated under **strict-inductive inference**: each unseen
executable is classified in its own isolated ego-graph. The repository also contains the
benchmark audits (raw-feature separability, architecture identifiability, near-duplicate
similarity, family imbalance), robustness experiments and cost measurements reported in the paper.

**The dataset is not redistributed here.** Download it from its authors (see [Data](#data)).

---

## Repository layout

```
DynHIN/
├── README.md
├── LICENSE                  MIT (code). The dataset has its own terms.
├── CITATION.cff
├── requirements.txt
├── verify_claims.py         re-derives the reported numbers from results/ (no GPU, no raw data)
├── data/
│   ├── README.md            how to obtain and place CIC-YNU-IoTMal
│   ├── SHA256SUMS.txt       checksums of the 12 parquet files used in the paper
│   └── check_data.py        checks layout + checksums of your local copy
├── src/                     all code (see "Scripts" below)
└── results/                 per-seed outputs of the runs reported in the paper
    ├── 01-main/             clean benchmark, protocol gap, ablation, LOFO (10 seeds)
    ├── 02-robustness/       noise, edge drop, FGSM/BIM/PGD (10 seeds)
    ├── 03-dedup/            near-duplicate re-evaluation (3 seeds)
    ├── 04-cost/             parameter counts, memory, latency (Tesla T4)
    ├── 05-audit/            benchmark-audit probes P1–P4, cross-machine check
    ├── 06-isolation/        batch-invariance / ego-graph equivalence check
    └── 07-threshold/        threshold-sweep diagnostic
```

---

## Requirements

The paper's runs used Kaggle with one NVIDIA Tesla T4 and:

| Package | Version |
|---|---|
| Python | 3.12 |
| torch | 2.10.0+cu128 |
| torch_geometric | 2.8.0.post1 |
| numpy | 2.0.2 |
| pandas | 2.3.3 |
| scikit-learn | 1.6.1 |
| duckdb | 1.3.2 |

```bash
pip install -r requirements.txt
```

Install PyTorch and PyTorch Geometric for your CUDA version first if `pip` cannot resolve them
(see the PyTorch and PyG installation pages). `verify_claims.py` only needs `numpy` and `pandas`.

---

## Data

CIC-YNU-IoTMal is published by the Canadian Institute for Cybersecurity (UNB) and Yokohama National
University:

- Dataset page: <https://www.unb.ca/cic/datasets/ynu-iot-2026.html>
- Paper: S. Dadkhah et al., *CIC-YNU-IoTMal: A comprehensive multilayer dataset for static and
  dynamic analysis of IoT malware behavior*, Information Systems 140 (2026) 102722.

Use the **parquet** release of each architecture and place the files as follows:

```
data/cic-ynu-iotmal/
├── arm/     pcap.parquet  sar.parquet  strace.parquet
├── mips/    pcap.parquet  sar.parquet  strace.parquet
├── mipsel/  pcap.parquet  sar.parquet  strace.parquet
└── x86/     pcap.parquet  sar.parquet  strace.parquet
```

Then check that your copy matches the files used in the paper:

```bash
python data/check_data.py
```

Details are in [`data/README.md`](data/README.md).

---

## Quick check without a GPU

```bash
pip install numpy pandas
python verify_claims.py            # full listing
python verify_claims.py --quiet    # failures and summary only
```

The script reads only the CSV/TXT files in `results/` and runs 643 checks: 6 on the integrity of the release and 637 numbers printed in the
manuscript, table by table: Tables 2–3 and 6–18, the corrected statistics of Section 5.5
(Nadeau–Bengio corrected resampled *t*, Holm adjustment), the per-architecture protocol gap,
the near-duplicate analysis, the isolation check and the cross-run summary. Each line shows the
recomputed value next to the printed one; the script exits with a non-zero code if any value
differs by more than the rounding of the printed number.
Expected last line: `✅ 643/643 reported numbers reproduced from the released results.`

Three items cannot be re-derived from this release and are listed at the end of the run:
Table 20 (needs the raw CSVs of both independent runs), the upper end of the 76–82% threshold
range in Section 6.2 (from a second run), and the trivial all-malicious F1 in Section 7.3.

---

## Running the experiments

```bash
cd src
export RHGT_DATA_DIR=$(pwd)/../data/cic-ynu-iotmal   # or let rhgt_paths.py detect it
export RHGT_SAVE_DIR=$(pwd)/../experiments
python rhgt_paths.py                                 # prints the detected data directory
python code-rhgt-v2.py
```

`code-rhgt-v2.py` is the full pipeline: data merge (DuckDB), hash-grouped stratified 80/20 splits
for 10 seeds (`10, 42, 100, 314, 555, 777, 999, 1234, 2024, 2026`), single-modality and
multimodal baselines, the HIN encoders, the R-HGT stages S1–S6, shared-entity (Protocol S) and
isolated ego-graph (Protocol I) inference, noise/attack scenarios and leave-one-family-out (LOFO)
evaluation. All hyperparameters are fixed in the code (see Table 5 of the paper) and are recorded
in `run_config.json` of every run.

Main environment variables:

| Variable | Meaning |
|---|---|
| `RHGT_DATA_DIR` | directory containing `arm/ mips/ mipsel/ x86/` |
| `RHGT_SAVE_DIR` | output directory; point it to a previous run to resume |
| `RHGT_CACHE_PARQUET` | pre-merged parquet, skips the DuckDB merge |
| `RHGT_SEEDS` | comma-separated seeds, e.g. `10,42,100` |
| `RHGT_SCENARIOS` | `clean` \| `attack` \| `all` |
| `RHGT_EXTRA` | extra configuration groups: `A`, `B`, `C` or combined (`ABC`) |
| `RHGT_LOFO_FAMILIES` | held-out families, e.g. `Mirai,DarkNexus` |
| `RHGT_SKIP_MAIN` / `RHGT_SKIP_LOFO` | skip one half of the protocol |
| `RHGT_ATTACK_CONFIGS` | configurations that receive noise/attacks |
| `RHGT_ATTACK_ITERS` / `RHGT_ATTACK_RESTARTS` | BIM/PGD steps and PGD restarts (paper: 20 and 3) |
| `RHGT_SAVE_PROBS` | save per-sample probabilities (`prob_dumps/`) for later analyses |
| `RHGT_RESUME` | skip (seed, configuration) pairs that already have results |
| `RHGT_QUICK_TEST` | tiny run to test the pipeline end to end |

On Kaggle, `kaggle_setup.py` sets these variables for predefined sessions, e.g.

```python
%run kaggle_setup.py --phase smoke     # quick end-to-end test
%run kaggle_setup.py --phase full      # all scenarios + LOFO; ~2.7 h per seed on a T4
!python code-rhgt-v2.py
```

Run `python kaggle_setup.py --help` for the list of phases. A full 10-seed run takes about 27 GPU
hours on a T4 and can be split across sessions (`RHGT_RESUME=1`) or machines (`--seeds`), then merged
with `merge_runs.py`.

---

## Scripts (`src/`)

| Script | Purpose | GPU |
|---|---|---|
| `code-rhgt-v2.py` | main training/evaluation pipeline | yes |
| `rhgt_paths.py` | detects the data directory | no |
| `kaggle_setup.py` | predefined Kaggle sessions (environment variables) | no |
| `merge_runs.py` | merges outputs of several sessions/machines | no |
| `make_tables.py` | builds the result tables from per-seed CSVs | no |
| `analyze_phase1.py` | paired statistics (protocol gap, Holm, corrected resampled *t*) | no |
| `analyze_lofo.py` | LOFO summaries | no |
| `separate_mechanisms.py` | protocol-gap mechanism analyses and corrected statistics | yes |
| `threshold_sweep.py` | threshold vs. representation decomposition from `prob_dumps` | no |
| `dedup_reeval.py` | re-evaluation after removing near-duplicate test samples | no |
| `audit_probes.py` | audit probes P1–P4 (separability, architecture, similarity, families) | no |
| `analyze_group_d.py` | additional dataset sanity checks | no |
| `verify_slim_ego.py` | ego-graph equivalence and batch-invariance (isolation) check | yes |
| `measure_params.py` | parameter counts / analytic cross-check | no (CPU) |
| `bench_cost.py` | memory and latency benchmark | yes |
| `crosscheck_runs.py` | compares two independent runs | no |
| `make_fig_hin_sample.py` | draws the HIN schema figure | no |

Each script has a docstring and `--help`.

---

## Where the paper's numbers come from

| Paper content | Folder |
|---|---|
| Clean detection (Table 7), protocol gap (Table 8), realignment (Table 9), ablation S1–S6, S5b, S6a (Table 10) | `results/01-main` |
| Family-held-out Mirai / DarkNexus (Table 15) | `results/01-main/lofo_zeroday_generalization` |
| Noise, masking, edge drop (Table 11); FGSM/BIM/PGD (Tables 12–13, Fig. 4); per-architecture PGD (Table 14) | `results/02-robustness` |
| Near-duplicate sensitivity (Section 8.3) | `results/03-dedup` |
| Parameters, memory, latency (Table 6) | `results/04-cost` |
| Raw-feature separability, single-feature stumps, architecture identifiability, family imbalance (Tables 16–18) | `results/05-audit` |
| Cross-run reproducibility (Section 9.3) | `results/05-audit/crosscheck_l3_l5.md` |
| Batch-invariance (isolation) check (Section 3.3) | `results/06-isolation` |
| Threshold diagnostic (Section 6.2) | `results/07-threshold` |

**Important:** `01-main` and `02-robustness` are two independent runs. Robustness curves must use the
`Clean` rows of `02-robustness` itself as the ε = 0 reference (same checkpoints), not those of `01-main`.

Model names in the CSVs: `R-HGT Stage 6 (+ Strict-Inductive Fine-Tuning)` is the final R-HGT;
`R-HGT Strict Control (No Fine-Tuning)` is S5 under isolated inference; `(shared eval)` denotes
Protocol S; `(robust + strict FT)` denotes the procedure-matched HIN baselines.
HAN results are marked † in the paper: PyG's `HANConv` fails the batch-invariance check.

---

## Reproducibility notes

- GPU training is not bitwise deterministic (scatter-add uses atomic operations). Two independent
  runs with the same code and seeds differ by a median of 0.0033 F1; the largest differences come
  from the unstable STRACE CNN-LSTM baseline. Report distributions, not the last decimal.
- The hash-grouped split depends on the row order produced by the DuckDB `GROUP BY`, which can change
  between DuckDB versions. The paper's runs used DuckDB 1.3.2; with DuckDB 1.5.5 the test-set size and
  class–architecture strata are identical but individual sample assignments differ.
- No validation split is used: all hyperparameters were fixed before the publication runs and were not
  tuned on test results.

---

## License and citation

Code: MIT License (see `LICENSE`). The CIC-YNU-IoTMal dataset is not covered by this license; follow
the terms of its providers and cite their paper.

If you use this code, please cite the paper (see `CITATION.cff`) and the dataset:

```bibtex
@article{dadkhah2026cicynu,
  author  = {Dadkhah, Sajjad and Okey, Ogobuchi Daniel and Maret, Sebin Abraham and Lo, Yen-Wu and
             Firouzi, Amir and Kuki, Ryu and Sasaki, Takayuki and Yoshioka, Katsunari and Ban, Tao and
             Ozawa, Seiichi and Ghorbani, Ali A.},
  title   = {{CIC-YNU-IoTMal}: A comprehensive multilayer dataset for static and dynamic analysis of
             {IoT} malware behavior},
  journal = {Information Systems},
  volume  = {140},
  pages   = {102722},
  year    = {2026}
}
```
