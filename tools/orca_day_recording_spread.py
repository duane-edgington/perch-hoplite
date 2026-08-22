#!/usr/bin/env python3
"""
orca_day_recording_spread.py — confound check for the by-day orca t-SNE.

For each confirmed orca day, report how many DISTINCT recordings (WAV files) its
orca_call windows span, and the per-file counts + time span. Purpose: test whether a
day's t-SNE cluster (esp. Apr 25) reflects genuine acoustic difference or is an artifact
of coming from just one or two recordings (e.g. a single boat-heavy stretch).

Read:
  - Many files spread across the day  -> separation is more likely real.
  - One or two files                  -> be skeptical; cluster may be recording-specific.

Uses the same annotations join as tools/plot_tsne.py (no usearch needed — labels only).

    python3 tools/orca_day_recording_spread.py \
        --db /mnt/PAM_Analysis/perch-hoplite/db/MARS_20180401_20180430_32kHz_norm \
        --db /mnt/PAM_Analysis/perch-hoplite/db/MARS_20180501_20180531_32kHz_norm
"""
import argparse
import os
import re
import sqlite3
from collections import defaultdict
from datetime import date

CONFIRMED_DAYS = {
    date(2018, 4, 13), date(2018, 4, 18), date(2018, 4, 21), date(2018, 4, 25), date(2018, 5, 12),
}
_DATE_RE = re.compile(r"(20\d{2})(\d{2})(\d{2})")
# MARS filename: MARS_YYYYMMDD_HHMMSS_resampled_32kHz.wav -> capture HHMMSS too
_DT_RE = re.compile(r"(20\d{2})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})")


def _day(fname):
    m = _DATE_RE.search(fname or "")
    if not m:
        return None
    try:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


def _hhmm(fname):
    m = _DT_RE.search(fname or "")
    return f"{m.group(4)}:{m.group(5)}" if m else "??:??"


def collect(db_dirs):
    # day -> filename -> count
    per = defaultdict(lambda: defaultdict(int))
    for d in db_dirs:
        p = os.path.join(d, "hoplite.sqlite")
        if not os.path.exists(p):
            print(f"  (skip, no hoplite.sqlite: {d})")
            continue
        con = sqlite3.connect(p)
        rows = con.execute("""
            SELECT a.label, a.label_type, r.filename
            FROM annotations a
            JOIN recordings r ON r.id = a.recording_id
        """).fetchall()
        con.close()
        for label, ltype, fname in rows:
            if ltype == 2 or label != "orca_call":
                continue
            day = _day(fname)
            if day in CONFIRMED_DAYS:
                per[day][fname] += 1
    return per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", action="append", required=True, help="hoplite DB dir (repeatable)")
    ap.add_argument("--max-files-shown", type=int, default=12)
    args = ap.parse_args()

    per = collect(args.db)
    if not per:
        print("No confirmed-day orca_call annotations found.")
        return

    print(f"\n{'day':<12}{'orca windows':>14}{'distinct WAVs':>15}{'read':>10}")
    print("-" * 51)
    for day in sorted(per):
        files = per[day]
        total = sum(files.values())
        nfiles = len(files)
        verdict = "REAL-ish" if nfiles >= 5 else ("skeptical" if nfiles <= 2 else "mixed")
        print(f"{day.isoformat():<12}{total:>14}{nfiles:>15}{verdict:>10}")

    print("\nPer-file breakdown (time-of-day HH:MM from filename):")
    for day in sorted(per):
        files = per[day]
        print(f"\n  {day.isoformat()}  —  {len(files)} recordings, {sum(files.values())} windows")
        shown = sorted(files.items(), key=lambda kv: kv[1], reverse=True)
        for fname, cnt in shown[:args.max_files_shown]:
            print(f"      {_hhmm(fname):>6}  {cnt:>4}  {fname}")
        if len(shown) > args.max_files_shown:
            print(f"      ... +{len(shown) - args.max_files_shown} more recordings")

    print("\nInterpretation: a day whose orca windows span MANY recordings across the day is "
          "hard to explain as a single-recording/boat artifact — its t-SNE separation is more "
          "likely real. A day concentrated in 1-2 files warrants skepticism.")
    print("\nCAVEAT (J. Ryan, Aug 21 2026): a good multi-recording spread only rules out the "
          "single-boat/single-recording artifact. It does NOT rule out range/propagation "
          "effects — animals at different distances from the hydrophone on different days can "
          "produce systematically different embeddings purely from frequency-dependent "
          "propagation loss and topography (MARS sits in Monterey Canyon at ~891m), with no "
          "difference in the actual calls or callers. Multi-recording spread is necessary but "
          "NOT sufficient to conclude 'different pod/encounter' — it narrows the explanation "
          "space, it doesn't settle it.")


if __name__ == "__main__":
    main()
