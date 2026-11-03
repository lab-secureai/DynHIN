#!/usr/bin/env python3
"""Check that the local copy of CIC-YNU-IoTMal matches the files used in the paper.

Expected layout (relative to this folder):

    cic-ynu-iotmal/{arm,mips,mipsel,x86}/{pcap,sar,strace}.parquet

Usage:
    python data/check_data.py                      # checks data/cic-ynu-iotmal
    python data/check_data.py --root /path/to/dir  # dir containing arm/ mips/ mipsel/ x86/
"""
import argparse
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.join(HERE, "cic-ynu-iotmal"),
                    help="directory containing arm/ mips/ mipsel/ x86/")
    args = ap.parse_args()

    expected = []
    with open(os.path.join(HERE, "SHA256SUMS.txt"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                digest, rel = line.split()
                expected.append((digest, rel.split("/", 1)[1]))  # drop the 'cic-ynu-iotmal/' prefix

    missing, mismatch = 0, 0
    for digest, rel in expected:
        path = os.path.join(args.root, rel)
        if not os.path.isfile(path):
            print(f"MISSING   {rel}")
            missing += 1
            continue
        ok = sha256(path) == digest
        print(f"{'OK      ' if ok else 'MISMATCH'}  {rel}")
        mismatch += 0 if ok else 1

    print()
    if missing or mismatch:
        print(f"{missing} missing, {mismatch} with a different checksum "
              f"(a different dataset release may give different results).")
        sys.exit(1)
    print(f"All {len(expected)} files match the versions used in the paper.")
    print(f"Run the pipeline with:  export RHGT_DATA_DIR={args.root}")


if __name__ == "__main__":
    main()
