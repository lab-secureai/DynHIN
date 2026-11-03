# Data

This repository does **not** redistribute CIC-YNU-IoTMal. Obtain it from its providers:

- Dataset page: <https://www.unb.ca/cic/datasets/ynu-iot-2026.html> (download link on that page)
- Citation: S. Dadkhah et al., *CIC-YNU-IoTMal: A comprehensive multilayer dataset for static and dynamic
  analysis of IoT malware behavior*, Information Systems 140 (2026) 102722.

## What the paper uses

For each architecture (Arm, MIPS, Mipsel, x86) the dataset ships a `parquet format.tar.xz` archive with
`pcap.parquet`, `sar.parquet` and `strace.parquet`. Extract the three files of each architecture into:

```
data/cic-ynu-iotmal/
├── arm/     pcap.parquet  sar.parquet  strace.parquet
├── mips/    pcap.parquet  sar.parquet  strace.parquet
├── mipsel/  pcap.parquet  sar.parquet  strace.parquet
└── x86/     pcap.parquet  sar.parquet  strace.parquet
```

Folder names are lower case. Only the dynamic PCAP, SAR and STRACE layers are used; static features and
the binaries themselves are not needed.

## Check your copy

```bash
python data/check_data.py
```

`SHA256SUMS.txt` lists the checksums of the 12 files used for the paper (about 257 MB in total). If a
file differs, the dataset may have been re-released and results can change.

## What the pipeline does with the files

`src/code-rhgt-v2.py` merges the three modalities per sample with DuckDB (aggregation by `Hash` and
architecture, inner join across modalities), drops samples without a usable label, and builds:

- 24 PCAP + 502 SAR continuous features (526-dim `file` node features),
- 181 STRACE system-call indicators (edges to 181 `syscall` nodes),
- 14 protocol indicators (edges to `protocol` nodes) and 4 `arch` nodes.

The retained corpus has 17,023 samples (9,323 benign, 7,700 malware). Per-architecture counts, the
family distribution and the inner-join attrition are in `results/01-main/dataset_statistics/` and
in the paper (Tables 2–4).
