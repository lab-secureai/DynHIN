#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rhgt_paths.py — AUTOMATIC DATA-PATH DETECTION.

Why this exists
---------------
The data path is the most error-prone setting when moving between Colab /
Kaggle / a personal machine, and each mistake wastes several minutes of
startup (or worse: the pipeline keeps running with an empty DataFrame). This
module does NOT ask you to remember the path: it scans the usual roots and
finds which directory ACTUALLY contains the data, by checking that files
exist rather than by matching names.

Criteria for "this is a data directory" (either one):
  (a) <dir>/cached_merged_multimodal.parquet exists      → kind = 'cache'
  (b) at least one <dir>/<arch>/pcap.parquet exists      → kind = 'raw'
      with arch ∈ {arm, mips, mipsel, x86}

Usage
-----
In a notebook, BEFORE running the pipeline:

    import rhgt_paths
    rhgt_paths.apply()          # sets os.environ['RHGT_DATA_DIR'] automatically

From the command line:

    python3 rhgt_paths.py               # print the detected directory + diagnostics
    python3 rhgt_paths.py --export      # print 'export RHGT_DATA_DIR=...'
    eval "$(python3 rhgt_paths.py --export)" && python3 code-rhgt-v2.py

If the RHGT_DATA_DIR environment variable is already set, the module respects
that value and only checks that it is valid, without overwriting it.
"""

from __future__ import annotations

import argparse
import os
import sys

__all__ = ["ARCHS", "MODALITIES", "classify", "find_data_dir", "apply", "report"]

ARCHS = ("arm", "mips", "mipsel", "x86")
MODALITIES = ("pcap", "sar", "strace")

CACHE_NAME = "cached_merged_multimodal.parquet"

# Roots to scan, in decreasing order of priority. Roots that do not exist are skipped.
SEARCH_ROOTS = (
    "/kaggle/input",                 # Kaggle: attached datasets (read-only)
    "/kaggle/working",               # Kaggle: working directory
    "./data",                        # standard layout of this repo
    ".",                             # running directly inside the data directory
    "/content/local_dataset",        # Colab: local SSD
    "/content/drive/MyDrive/dataset",  # Colab: Google Drive
    "/content",
)

MAX_DEPTH = 4  # enough for /kaggle/input/<dataset>/<owner>/<slug>/<subdir>


# --------------------------------------------------------------------------- #
# Classify a directory
# --------------------------------------------------------------------------- #
def classify(path: str) -> tuple[str | None, dict]:
    """Return (kind, info). kind ∈ {'cache', 'raw', None}."""
    if not os.path.isdir(path):
        return None, {}

    cache = os.path.join(path, CACHE_NAME)
    if os.path.isfile(cache):
        try:
            size = os.path.getsize(cache)
        except OSError:
            size = 0
        return "cache", {"cache_bytes": size}

    complete, partial, files = [], [], 0
    for arch in ARCHS:
        adir = os.path.join(path, arch)
        if not os.path.isdir(adir):
            continue
        have = [m for m in MODALITIES
                if os.path.isfile(os.path.join(adir, f"{m}.parquet"))]
        if not have:
            continue
        files += len(have)
        (complete if len(have) == len(MODALITIES) else partial).append(arch)

    if not (complete or partial):
        return None, {}
    if "pcap" not in MODALITIES:  # defensive, never happens
        return None, {}
    return "raw", {"complete": complete, "partial": partial, "files": files}


def _score(kind: str, info: dict) -> tuple:
    """Ranking score: cache beats raw; raw with more complete architectures wins."""
    if kind == "cache":
        return (2, 0, 0)
    return (1, len(info.get("complete", [])), info.get("files", 0))


# --------------------------------------------------------------------------- #
# Scan
# --------------------------------------------------------------------------- #
def _walk(root: str, max_depth: int = MAX_DEPTH):
    """Yield the subdirectories of root down to max_depth (including root itself)."""
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        return
    yield root
    base_depth = root.rstrip(os.sep).count(os.sep)
    for cur, dirs, _files in os.walk(root):
        depth = cur.rstrip(os.sep).count(os.sep) - base_depth
        if depth >= max_depth:
            dirs[:] = []
            continue
        # Do not descend into architecture directories, result directories, or hidden directories.
        dirs[:] = [d for d in dirs
                   if d not in ARCHS
                   and not d.startswith(".")
                   and d not in ("experiments", "__pycache__", "checkpoints",
                                 "prob_dumps", "_archive")]
        for d in dirs:
            yield os.path.join(cur, d)


def find_data_dir(hint: str | None = None,
                  roots: tuple[str, ...] = SEARCH_ROOTS,
                  verbose: bool = True):
    """Find the best data directory.

    Order: hint (if valid) → RHGT_DATA_DIR (if valid) → scan roots.
    Returns (path, kind, candidates), where candidates is the ranked list
    [(path, kind, info)]. path = None if nothing is found.
    """
    found: dict[str, tuple[str, dict]] = {}

    def consider(p: str):
        p = os.path.abspath(p)
        if p in found:
            return
        kind, info = classify(p)
        if kind:
            found[p] = (kind, info)

    # Explicit choices take absolute priority, but they must still be valid.
    for explicit in (hint, os.environ.get("RHGT_DATA_DIR")):
        if explicit:
            kind, info = classify(explicit)
            if kind:
                if verbose:
                    print(f"✅ Using the specified path: {explicit}  ({kind})")
                return os.path.abspath(explicit), kind, [
                    (os.path.abspath(explicit), kind, info)]
            if verbose:
                print(f"⚠️  The specified path is NOT valid, ignoring it: {explicit}")

    for root in roots:
        for d in _walk(root):
            consider(d)

    if not found:
        return None, None, []

    ranked = sorted(found.items(),
                    key=lambda kv: _score(kv[1][0], kv[1][1]), reverse=True)
    best_path, (best_kind, _best_info) = ranked[0]
    return best_path, best_kind, [(p, k, i) for p, (k, i) in ranked]


# --------------------------------------------------------------------------- #
# Report + apply
# --------------------------------------------------------------------------- #
def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def report(candidates) -> None:
    if not candidates:
        print("❌ No data directory found.")
        print("   Required: <dir>/{arm,mips,mipsel,x86}/{pcap,sar,strace}.parquet")
        print("   or <dir>/" + CACHE_NAME)
        print("\n   Roots scanned:")
        for r in SEARCH_ROOTS:
            mark = "exists" if os.path.isdir(r) else "does not exist"
            print(f"     {r:42s} {mark}")
        inp = "/kaggle/input"
        if os.path.isdir(inp):
            print(f"\n   Contents of {inp}:")
            for d in sorted(os.listdir(inp))[:20]:
                print(f"     {d}")
        return

    print(f"Found {len(candidates)} candidates (ranked by priority):")
    for i, (p, kind, info) in enumerate(candidates):
        tag = "→ PICK" if i == 0 else "      "
        if kind == "cache":
            detail = f"pre-merged parquet, {_human(info.get('cache_bytes', 0))}"
        else:
            detail = (f"all 3 modalities: {info.get('complete') or '—'}"
                      f" | partial modalities: {info.get('partial') or '—'}"
                      f" | {info.get('files', 0)} file")
        print(f"  {tag} [{kind:5s}] {p}")
        print(f"          {detail}")


def apply(hint: str | None = None, verbose: bool = True) -> str | None:
    """Detect, then set os.environ['RHGT_DATA_DIR'] (and RHGT_CACHE_PARQUET if applicable).

    Returns the chosen path, or None.
    """
    path, kind, cands = find_data_dir(hint, verbose=verbose)
    if verbose:
        report(cands)
    if path is None:
        return None
    os.environ["RHGT_DATA_DIR"] = path
    if kind == "cache":
        os.environ["RHGT_CACHE_PARQUET"] = os.path.join(path, CACHE_NAME)
    if verbose:
        print(f"\n📂 RHGT_DATA_DIR = {path}")
        if kind == "cache":
            print(f"⚡ RHGT_CACHE_PARQUET = {os.environ['RHGT_CACHE_PARQUET']}")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Detect the CIC-YNU-IoTMal data directory for code-rhgt-v2.py.")
    ap.add_argument("--hint", default=None,
                    help="A path you believe is correct; it is still validated before use.")
    ap.add_argument("--export", action="store_true",
                    help="Only print the 'export RHGT_DATA_DIR=...' line, to be eval'd in a shell.")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.export:
        path, kind, _ = find_data_dir(args.hint, verbose=False)
        if path is None:
            print("# rhgt_paths: data NOT found", file=sys.stderr)
            return 1
        print(f"export RHGT_DATA_DIR={path!r}".replace("'", '"'))
        if kind == "cache":
            print(f'export RHGT_CACHE_PARQUET="{os.path.join(path, CACHE_NAME)}"')
        return 0

    path = apply(args.hint, verbose=not args.quiet)
    return 0 if path else 1


if __name__ == "__main__":
    raise SystemExit(main())
