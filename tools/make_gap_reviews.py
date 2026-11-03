#!/usr/bin/env python3
"""
make_gap_reviews.py

For each month given, builds a review CSV of the windows whose v10 logit is at or above the
pass-1 threshold (default 2.31) that carry NO annotation of any type in that month's hoplite DB.
These are windows that were never reviewed (or skipped) in pass-1.

Output: <results-dir>/review_<mon><year>_gap.csv, with the same columns as the v10 inference CSV
(idx, project, filename, window_start, window_end, label, logits), highest logit first. The
file can be used directly as --detections-csv for `phase2_classify.py review`.

Safe by design: reads the DBs and v10 CSVs, creates new CSVs only, and refuses to overwrite an
existing file unless --overwrite is given.

Usage (from ~/perch-hoplite on spark-ae0e):
    python3 tools/make_gap_reviews.py --months 2016-01 2019-01 2019-03 2019-11 2022-01 2023-12
"""

import argparse
import calendar
import csv
import glob
import os
import re
import sqlite3
import struct
from pathlib import Path

STEM_RE = re.compile(r"(MARS_(\d{8})_\d{6})")
DB_RE = re.compile(r"^MARS_(\d{4})(\d{2})01_\d{8}_32kHz(_norm)?$")
AUDIO_CANDIDATES = [
    "/mnt/PAM_Analysis/GoogleMultiSpeciesWhaleModel2/resampled_32kHz/{y}/{m}",
    "/mnt/PAM_20TiB_32k/{y}/{m}",
    "/mnt/PAM_32k_Archive/{y}/{m}",
    "/mnt/SL500/{y}/{m}",
    "/mnt/SL500/resampled_32kHz/{y}/{m}",
]


def first_offset(blob):
    if not blob:
        return 0.0
    n = len(blob) // 8
    return struct.unpack(f"<{n}d", bytes(blob[: n * 8]))[0] if n else 0.0


def find_db(root, ym):
    best = None
    for d in sorted(Path(root).iterdir()):
        m = DB_RE.match(d.name)
        if m and f"{m.group(1)}-{m.group(2)}" == ym and (d / "hoplite.sqlite").exists():
            if best is None or m.group(3):
                best = d
    return best


def annotated_keys(db_dir):
    con = sqlite3.connect(f"file:{db_dir / 'hoplite.sqlite'}?mode=ro", uri=True)
    try:
        rows = con.execute("SELECT r.filename, a.offsets FROM annotations a "
                           "JOIN recordings r ON r.id = a.recording_id").fetchall()
    finally:
        con.close()
    keys = set()
    for fn, blob in rows:
        m = STEM_RE.search(fn)
        if m:
            keys.add((m.group(1), round(first_offset(blob), 1)))
    return keys


def find_audio(ym):
    y, m = ym.split("-")
    for tpl in AUDIO_CANDIDATES:
        path = tpl.format(y=y, m=m)
        if os.path.isdir(path):
            return path, len(os.listdir(path))
    return None, 0


def main():
    ap = argparse.ArgumentParser(description="Build review CSVs for unannotated above-threshold windows.")
    ap.add_argument("--months", nargs="+", required=True, help="YYYY-MM, one or more")
    ap.add_argument("--db-root", default="/mnt/PAM_Analysis/perch-hoplite/db")
    ap.add_argument("--results-dir", default="/mnt/PAM_Analysis/perch-hoplite/results")
    ap.add_argument("--threshold", type=float, default=2.31)
    ap.add_argument("--overwrite", action="store_true", help="replace an existing review_*_gap.csv")
    args = ap.parse_args()
    res = os.path.expanduser(args.results_dir)

    for ym in args.months:
        y, m = ym.split("-")
        print(f"\n== {ym}")
        db = find_db(args.db_root, ym)
        files = glob.glob(os.path.join(res, f"MARS_{y}{m}01_*_v10_orcaval.csv"))
        if db is None or not files:
            print(f"   skipped: {'no DB' if db is None else 'no v10 CSV'} for this month")
            continue
        ann = annotated_keys(db)
        with open(files[0], newline="") as f:
            rd = csv.DictReader(f)
            cols = rd.fieldnames
            todo = []
            for r in rd:
                if r.get("label", "orca_call") != "orca_call":
                    continue
                mm = STEM_RE.search(r["filename"])
                key = (mm.group(1), round(float(r["window_start"]), 1)) if mm else None
                if key and float(r["logits"]) >= args.threshold and key not in ann:
                    todo.append(r)
        todo.sort(key=lambda r: -float(r["logits"]))
        out = os.path.join(res, f"review_{calendar.month_abbr[int(m)].lower()}{y}_gap.csv")
        if not todo:
            print("   no unannotated above-threshold windows")
            continue
        if os.path.exists(out) and not args.overwrite:
            print(f"   NOT written: {out} already exists (use --overwrite to replace it)")
            continue
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(todo)
        days = sorted({r["filename"][5:13] for r in todo})
        audio, nfiles = find_audio(ym)
        print(f"   {len(todo)} windows, scores {float(todo[-1]['logits']):.3f} to {float(todo[0]['logits']):.3f}, dates {', '.join(days)}")
        print(f"   wrote {out}")
        print(f"   db:    {db}")
        print(f"   audio: {audio} ({nfiles} files)" if audio else "   audio: NOT FOUND in the usual locations")


if __name__ == "__main__":
    main()
