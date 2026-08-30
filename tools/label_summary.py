#!/usr/bin/env python3
"""tools/label_summary.py — per-day, per-class label summary for one processed month.

Produces (a) a per-day histogram of CONFIRMED labels by class, effort-normalised against
the month's coverage CSV, and (b) an exact UTC timestamp list for every confirmed label so
a collaborator can navigate directly to it in an external tool (e.g. the MBARI Soundscape
Visual Browser, https://www.mbari.org/data/soundscape-visual-browser/).

Timestamps are derived as: recording start (from the MARS_YYYYMMDD_HHMMSS filename)
+ the window's offset within that recording.

Usage:
    python3 tools/label_summary.py \\
        --db /mnt/PAM_Analysis/perch-hoplite/db/MARS_20150901_20150930_32kHz_norm/hoplite.sqlite \\
        --coverage results/coverage/2015-09_coverage.csv \\
        [--markdown]
"""
import argparse
import csv
import re
import sqlite3
import struct
import sys
from collections import defaultdict
from datetime import datetime, timedelta

FN = re.compile(r'MARS_(\d{8})_(\d{6})')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', required=True)
    ap.add_argument('--coverage', default=None,
                    help='results/coverage/<YYYY>-<MM>_coverage.csv, for effort normalisation')
    ap.add_argument('--markdown', action='store_true', help='emit markdown tables')
    args = ap.parse_args()

    con = sqlite3.connect(args.db)
    rows = []
    for fn, blob, label, prov in con.execute("""
            SELECT r.filename, a.offsets, a.label, a.provenance
            FROM annotations a JOIN recordings r ON r.id = a.recording_id"""):
        m = FN.search(fn)
        if not m:
            continue
        try:
            start, _ = struct.unpack('<2d', blob)
        except Exception:
            start = 0.0
        d, t = m.groups()
        t0 = datetime.strptime(d + t, "%Y%m%d%H%M%S") + timedelta(seconds=start)
        rows.append((t0, label, fn, start, prov))
    con.close()

    if not rows:
        sys.exit("no annotations in this DB")
    rows.sort()

    classes = sorted({r[1] for r in rows})
    per_day = defaultdict(lambda: defaultdict(int))
    for t0, label, *_ in rows:
        per_day[t0.strftime("%Y-%m-%d")][label] += 1

    hours = {}
    if args.coverage:
        try:
            for row in csv.DictReader(open(args.coverage)):
                hours[row['date']] = float(row['hours'])
        except Exception as e:
            print(f"(coverage not read: {e})", file=sys.stderr)

    md = args.markdown
    sep = " | " if md else "  "
    print(f"\n## Confirmed labels by day\n" if md else "\nCONFIRMED LABELS BY DAY")
    head = ["date"] + classes + (["hours", "orca/h"] if hours else [])
    if md:
        print("| " + " | ".join(head) + " |")
        print("|" + "|".join("---" for _ in head) + "|")
    else:
        print("  " + "  ".join(f"{h:>12}" for h in head))
    for day in sorted(per_day):
        cells = [day] + [str(per_day[day].get(c, 0)) for c in classes]
        if hours:
            h = hours.get(day, 0.0)
            o = per_day[day].get('orca_call', 0)
            cells += [f"{h:.1f}", f"{o/h:.3f}" if h else "-"]
        print(("| " + " | ".join(cells) + " |") if md
              else "  " + "  ".join(f"{c:>12}" for c in cells))

    print(f"\n## Exact UTC timestamps\n" if md else "\nEXACT UTC TIMESTAMPS")
    for c in classes:
        sub = [r for r in rows if r[1] == c]
        print(f"\n**{c}** — {len(sub)}\n" if md else f"\n{c} ({len(sub)}):")
        if md:
            print("| UTC | recording | offset |")
            print("|---|---|---|")
        for t0, _l, fn, off, _p in sub:
            short = fn.replace('_resampled_32kHz.wav', '')
            print(f"| {t0:%Y-%m-%d %H:%M:%S} | `{short}` | {off:.0f} s |" if md
                  else f"  {t0:%Y-%m-%d %H:%M:%S}  {short}  @{off:.0f}s")

    print(f"\nTOTAL: {len(rows)} labels across {len(per_day)} day(s)", file=sys.stderr)


if __name__ == '__main__':
    main()
