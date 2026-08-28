#!/usr/bin/env python3
"""tools/coverage_histogram.py
Build the PERMANENT per-day recording-effort record for one month of MARS audio.

WHY THIS EXISTS
---------------
Under the full-archive campaign the bulk resampled WAV is DELETED after each month is
analyzed (analyze-then-archive; see CLAUDE_perch_hoplite.md #28 and the HANDOFF storage
note). Once it is gone, there is no way to recover how many hours were actually recorded
on a given day. Raw per-day detection counts are therefore UNINTERPRETABLE without this
file: August 2015 ranges from 2.7 h to 24 h of coverage per day, so Aug 19 (16 files) and
Aug 20 (145 files) cannot be compared as counts. Every seasonal/interannual figure needs
detections per hour of effort, which needs these hours.

Run this at Stage 1.5, BEFORE embedding, and commit the CSV to the repo.

WHAT IT DOES
------------
Reads durations directly from the audio (does NOT assume 600 s/file), then writes one row
per calendar date of the month -- including dates with ZERO files, which would otherwise
vanish from a `uniq -c` histogram (Aug 16 2015 is exactly this case).

Columns:
    date              YYYY-MM-DD
    files             number of files whose name carries this date
    seconds           summed true duration
    hours             seconds / 3600, 2dp
    expected_windows  sum(ceil(duration/5)) -- the Stage 2 reconciliation target
    pct_of_day        hours / 24 * 100, 1dp
    short_files       count of files not exactly 600 s (restarts / truncations)
    note              COMPLETE | NEAR-COMPLETE | PARTIAL | ABSENT | OVER-24H

Usage:
    python3 tools/coverage_histogram.py --audio-dir /mnt/.../resampled_32kHz/2015/08
    python3 tools/coverage_histogram.py --audio-dir ... --out results/coverage/2015-08_coverage.csv
"""
import argparse
import calendar
import csv
import math
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

FNAME_RE = re.compile(r'MARS_(\d{4})(\d{2})(\d{2})_(\d{6})')
WINDOW_S = 5.0
CHUNK = 200  # files per soxi invocation


def durations_for(files):
    """Return {path: seconds}. Batches soxi calls; falls back to per-file on mismatch."""
    out = {}
    for i in range(0, len(files), CHUNK):
        chunk = files[i:i + CHUNK]
        try:
            r = subprocess.run(['soxi', '-D'] + [str(f) for f in chunk],
                               capture_output=True, text=True, check=True)
            lines = [ln for ln in r.stdout.strip().split('\n') if ln.strip()]
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            sys.exit(f"soxi failed: {e}")
        if len(lines) == len(chunk):
            for f, ln in zip(chunk, lines):
                out[f] = float(ln)
        else:
            # ragged output -- redo this chunk one file at a time
            for f in chunk:
                r = subprocess.run(['soxi', '-D', str(f)],
                                   capture_output=True, text=True)
                out[f] = float(r.stdout.strip()) if r.returncode == 0 else 0.0
        print(f"  ...{min(i + CHUNK, len(files))}/{len(files)} files read",
              file=sys.stderr, end='\r')
    print(file=sys.stderr)
    return out


def classify(hours, files):
    if files == 0:
        return "ABSENT"
    if hours > 24.05:
        return "OVER-24H"
    if hours >= 23.9:
        return "COMPLETE"
    if hours >= 22.0:
        return "NEAR-COMPLETE"
    return "PARTIAL"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--audio-dir', required=True,
                    help='resampled WAV dir for ONE month, e.g. .../resampled_32kHz/2015/08')
    ap.add_argument('--out', default=None,
                    help='output CSV (default: results/coverage/<YYYY>-<MM>_coverage.csv)')
    ap.add_argument('--pattern', default='*.wav')
    args = ap.parse_args()

    audio_dir = Path(args.audio_dir)
    if not audio_dir.is_dir():
        sys.exit(f"not a directory: {audio_dir}")

    files = sorted(audio_dir.glob(args.pattern))
    if not files:
        sys.exit(f"no files matching {args.pattern} in {audio_dir}")

    # group by date from filename
    by_date = defaultdict(list)
    unparsed = []
    for f in files:
        m = FNAME_RE.search(f.name)
        if not m:
            unparsed.append(f.name)
            continue
        y, mo, d, _ = m.groups()
        by_date[(int(y), int(mo), int(d))].append(f)

    if unparsed:
        print(f"WARNING: {len(unparsed)} file(s) had no parseable MARS_YYYYMMDD_ date; "
              f"excluded. First: {unparsed[0]}", file=sys.stderr)
    if not by_date:
        sys.exit("no parseable filenames")

    years = {k[0] for k in by_date}
    months = {k[1] for k in by_date}
    if len(years) > 1 or len(months) > 1:
        sys.exit(f"--audio-dir must hold ONE month; found years={sorted(years)} "
                 f"months={sorted(months)}")
    year, month = years.pop(), months.pop()

    print(f"Reading durations for {len(files)} file(s) in {year}-{month:02d}...",
          file=sys.stderr)
    durs = durations_for(files)

    rows = []
    ndays = calendar.monthrange(year, month)[1]
    for day in range(1, ndays + 1):
        fl = by_date.get((year, month, day), [])
        secs = sum(durs.get(f, 0.0) for f in fl)
        wins = sum(math.ceil(durs.get(f, 0.0) / WINDOW_S) for f in fl)
        short = sum(1 for f in fl if abs(durs.get(f, 0.0) - 600.0) > 1e-6)
        hours = secs / 3600.0
        rows.append({
            'date': f"{year:04d}-{month:02d}-{day:02d}",
            'files': len(fl),
            'seconds': f"{secs:.1f}",
            'hours': f"{hours:.2f}",
            'expected_windows': wins,
            'pct_of_day': f"{hours / 24.0 * 100:.1f}",
            'short_files': short,
            'note': classify(hours, len(fl)),
        })

    out = Path(args.out) if args.out else \
        Path('results/coverage') / f"{year:04d}-{month:02d}_coverage.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- console summary ----
    tot_f = sum(r['files'] for r in rows)
    tot_s = sum(float(r['seconds']) for r in rows)
    tot_w = sum(r['expected_windows'] for r in rows)
    tot_short = sum(r['short_files'] for r in rows)
    nominal = ndays * 24.0

    print(f"\n{'date':12} {'files':>6} {'hours':>7} {'%day':>6} {'windows':>9} {'short':>6}  note")
    print('-' * 68)
    for r in rows:
        print(f"{r['date']:12} {r['files']:>6} {r['hours']:>7} {r['pct_of_day']:>6} "
              f"{r['expected_windows']:>9} {r['short_files']:>6}  {r['note']}")
    print('-' * 68)
    print(f"{'TOTAL':12} {tot_f:>6} {tot_s/3600:>7.2f} "
          f"{tot_s/3600/nominal*100:>6.1f} {tot_w:>9} {tot_short:>6}")
    print(f"\nMonth nominal coverage : {nominal:.0f} h ({ndays} days)")
    print(f"Actual recorded        : {tot_s/3600:.2f} h "
          f"({tot_s/3600/nominal*100:.1f}% of nominal)")
    print(f"Missing                : {nominal - tot_s/3600:.2f} h")
    print(f"TRUE expected windows  : {tot_w}   <-- Stage 2 reconciliation target")
    print(f"  (files x 120 would be {tot_f*120}, "
          f"a {tot_f*120 - tot_w} window overcount)")
    absent = [r['date'] for r in rows if r['note'] == 'ABSENT']
    over = [r['date'] for r in rows if r['note'] == 'OVER-24H']
    if absent:
        print(f"\nDATES WITH ZERO DATA ({len(absent)}): {', '.join(absent)}")
    if over:
        print(f"\n*** DATES EXCEEDING 24 h ({len(over)}): {', '.join(over)}")
        print("    More audio than the day can hold => files OVERLAP in time and the DB")
        print("    will contain DUPLICATED audio. Investigate before embedding.")
    print(f"\nWrote {out}")


if __name__ == '__main__':
    main()
