#!/usr/bin/env python3
"""
check_review_csv_in_db.py

Read-only. For every window listed in a review CSV (for example
review_jun2019_pass1.csv), report which labels the hoplite DB holds at that
recording + offset. Use it to see whether pass-1 labels made it into a DB.

Usage:
    python3 check_review_csv_in_db.py \
        --db  /mnt/PAM_Analysis/perch-hoplite/db/MARS_20190601_20190630_32kHz_norm \
        --csv /mnt/PAM_Analysis/perch-hoplite/results/review_jun2019_pass1.csv

The review CSVs have the recording filename in column 3 and the offset in
column 4 (counting from 1). Override with --filename-col / --offset-col
(0-based) if the printed header shows otherwise.
"""
import argparse
import csv
import sqlite3
import struct
from collections import Counter
from pathlib import Path


def first_offset(blob) -> float:
    if not blob:
        return 0.0
    n = len(blob) // 8
    return struct.unpack(f"<{n}d", bytes(blob[: n * 8]))[0] if n else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="hoplite DB directory")
    ap.add_argument("--csv", required=True, help="review CSV")
    ap.add_argument("--filename-col", type=int, default=2, help="0-based (default 2)")
    ap.add_argument("--offset-col", type=int, default=3, help="0-based (default 3)")
    args = ap.parse_args()

    with open(args.csv, newline="") as f:
        rows = list(csv.reader(f))
    header, data = rows[0], rows[1:]
    print("CSV header :", header)
    print("First row  :", data[0] if data else None)
    print(f"Windows in CSV: {len(data)}\n")

    con = sqlite3.connect(f"file:{Path(args.db) / 'hoplite.sqlite'}?mode=ro", uri=True)
    index: dict[tuple[str, float], list[tuple[str, int]]] = {}
    for fn, blob, label, ltype in con.execute(
        "SELECT r.filename, a.offsets, a.label, a.label_type "
        "FROM annotations a JOIN recordings r ON r.id = a.recording_id"
    ):
        index.setdefault((fn[:20], round(first_offset(blob), 1)), []).append((label, ltype))
    con.close()

    summary: Counter = Counter()
    for r in data:
        stem = r[args.filename_col][:20]
        off = round(float(r[args.offset_col]), 1)
        hits = index.get((stem, off))
        shown = ", ".join(f"{lab}(type {lt})" for lab, lt in hits) if hits else "NOT IN DB"
        print(f"{stem}  {off:7.1f}s  ->  {shown}")
        if hits:
            for lab, lt in hits:
                summary[f"{lab} (type {lt})"] += 1
        else:
            summary["NOT IN DB"] += 1

    print("\nSummary of the CSV's windows:")
    for k, v in summary.most_common():
        print(f"  {v:4d}  {k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
