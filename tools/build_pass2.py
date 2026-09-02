#!/usr/bin/env python3
"""tools/build_pass2.py — build the pass-2 zoom-in review set for a month.

The two-stage review protocol (finding #34):
  PASS 1  broad and shallow — everything at the models' operating thresholds, to FIND episodes.
  PASS 2  narrow and deep   — everything above a low floor, restricted to the dates/recordings
                              where pass 1 confirmed something, to COUNT the calls in them.

Pass 2 matters because detector recall at the operating point is poor: in September 2015 only
3 of 18 confirmed orca calls cleared v10's 2.31 threshold, and 11 of the 21 pass-2 clips (all
below the pass-1 cutoff) turned out to be real orca. Without the zoom-in that month reads as
7 scattered calls instead of two encounters.

Note the pass-2 floor (default 0.20) is ABOVE the standard 0.0 inference floor, so the ordinary
`_orcaval.csv` already contains every candidate — no extra inference run is needed.

Windows already reviewed are excluded, either by passing the pass-1 CSV (--exclude) or by
reading the DB's annotations table (--exclude-db), or both.

Usage:
    python3 tools/build_pass2.py \\
        --scores results/MARS_20151001_20151031_v10_orcaval.csv \\
        --dates 20151026 20151027 \\
        --exclude results/review_oct2015_pass1.csv \\
        --min-score 0.20 \\
        --out results/review_oct2015_pass2.csv
"""
import argparse
import csv
import struct
import sys
from collections import Counter
from pathlib import Path

COL_FILE, COL_OFF, COL_LABEL, COL_SCORE = 2, 3, 5, 6


def key(fn, off):
    return (fn, f"{float(off):.1f}")


def read_scores(path, label):
    header, rows = None, []
    with open(path) as f:
        for r in csv.reader(f):
            if header is None:
                header = r
                continue
            if len(r) <= COL_SCORE or r[COL_LABEL] != label:
                continue
            try:
                float(r[COL_SCORE])
            except ValueError:
                continue
            rows.append(r)
    return header, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scores', required=True, help='the v10 (or other) _orcaval.csv')
    ap.add_argument('--dates', nargs='+', required=True,
                    help='YYYYMMDD date(s) to restrict to, or full recording prefixes')
    ap.add_argument('--exclude', nargs='*', default=[],
                    help='CSV(s) of already-reviewed windows (e.g. the pass-1 set)')
    ap.add_argument('--exclude-db', default=None,
                    help='hoplite.sqlite — also exclude anything already in annotations')
    ap.add_argument('--min-score', type=float, default=0.20)
    ap.add_argument('--label', default='orca_call')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    header, rows = read_scores(args.scores, args.label)

    done = set()
    for x in args.exclude:
        _, ex = read_scores(x, args.label)
        for r in ex:
            done.add(key(r[COL_FILE], r[COL_OFF]))
    if args.exclude_db:
        import sqlite3
        con = sqlite3.connect(args.exclude_db)
        for fn, blob in con.execute("""SELECT r.filename, a.offsets FROM annotations a
                                       JOIN recordings r ON r.id = a.recording_id"""):
            try:
                start, _ = struct.unpack('<2d', blob)
                done.add(key(fn, start))
            except Exception:
                pass
        con.close()
    print(f"{len(done)} window(s) already reviewed, will be excluded", file=sys.stderr)

    def wanted(fn):
        return any(d in fn for d in args.dates)

    out = [r for r in rows
           if wanted(r[COL_FILE])
           and float(r[COL_SCORE]) >= args.min_score
           and key(r[COL_FILE], r[COL_OFF]) not in done]
    out.sort(key=lambda r: float(r[COL_SCORE]), reverse=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(out)

    print(f"\n{len(out)} NEW clips -> {args.out}\n")
    print(f"{'score':>7}  {'recording':<34} {'offset':>8}")
    print("-" * 56)
    for r in out:
        print(f"{float(r[COL_SCORE]):7.3f}  "
              f"{r[COL_FILE].replace('_resampled_32kHz.wav',''):<34} {r[COL_OFF]:>8}")
    print("\nby date:", dict(sorted(Counter(r[COL_FILE][5:13] for r in out).items())))
    print(f"\nlaunch with --num-results {len(out)}")


if __name__ == '__main__':
    main()
