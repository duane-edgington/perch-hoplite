#!/usr/bin/env python3
"""
diel_periods_orca.py   (READ-ONLY on the databases)

Split verified orca detections from the MARS hydrophone into daylight and
night periods, three rows per LOCAL calendar date (America/Los_Angeles,
PST/PDT handled automatically):

    row 1  local midnight -> civil dawn     daylight = 0
    row 2  civil dawn     -> civil dusk     daylight = 1
    row 3  civil dusk     -> next midnight  daylight = 0

Dawn/dusk = NOAA civil twilight (sun 6 deg below horizon), NOAA solar
algorithm (~1 min accuracy), at the MARS site 36.7125 N, 122.1868 W
(same function and coordinates as tools/plot_diel_vs_sightings.py).

Output columns:
    YYYYMMDD         local date
    start_local      ISO 8601 with UTC offset, e.g. 2018-04-13T06:12:30-07:00
    end_local        same format (period is [start, end) )
    daylight         1 = civil dawn to civil dusk, 0 = night
    orca_verified    verified orca_call windows whose START falls in the period
    data_available   1 if any recording overlaps the period, else 0
    recorded_hours   hours of recording inside the period (union of files)

Sources (from --db-root, any start day):
    MARS_YYYYMMDD_YYYYMMDD_32kHz[_norm]/hoplite.sqlite   (_norm preferred)
    detections: annotations with label 'orca_call', label_type 1 (positive),
                unique windows (recording + offset)
    timeline:   recordings table (start time from MARS_YYYYMMDD_HHMMSS in the
                filename, UTC) and windows table (duration = windows x 5 s,
                because the WAVs are no longer on disk)

Only recordings starting inside [--utc-start, --utc-end] are used, so the
first and last local dates can be partly outside the analyzed record; their
recorded_hours show that.

Validation (--daily-csv): detections are regrouped by UTC day and compared
with the daily CSV (YYYYMMDD,orca_verified,data_available). Every mismatch is
printed and written next to the output as *_validation.csv.

Usage (from ~/perch-hoplite on spark-ae0e):
    python3 tools/diel_periods_orca.py --daily-csv <path to daily_orca_verified_2015-2023.csv> --out results/diel_orca_verified_2015-2023.csv

Writes only --out and its _validation.csv. Refuses to overwrite --out unless
--overwrite is given. Never modifies or deletes anything else.
"""

import argparse
import bisect
import csv
import math
import os
import re
import sqlite3
import struct
import sys
from collections import defaultdict
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

TZ = ZoneInfo("America/Los_Angeles")
LAT, LON = 36.7125, -122.1868
WINDOW_S = 5.0
DB_RE = re.compile(r"^MARS_(\d{8})_(\d{8})_32kHz(_norm)?$")
STEM_RE = re.compile(r"MARS_(\d{8})_(\d{6})")


# ---------------------------------------------------------------- NOAA civil twilight
def civil_dawn_dusk_utc(d):
    """Civil dawn and dusk for local date d, returned as aware UTC datetimes.
    NOAA solar-position algorithm, same as tools/plot_diel_vs_sightings.py,
    but returned in UTC so no fixed offset is assumed."""
    n = d.timetuple().tm_yday
    gamma = 2 * math.pi / 365 * (n - 1 + 0.5)
    eqtime = 229.18 * (0.000075 + 0.001868 * math.cos(gamma)
                       - 0.032077 * math.sin(gamma) - 0.014615 * math.cos(2 * gamma)
                       - 0.040849 * math.sin(2 * gamma))
    decl = (0.006918 - 0.399912 * math.cos(gamma) + 0.070257 * math.sin(gamma)
            - 0.006758 * math.cos(2 * gamma) + 0.000907 * math.sin(2 * gamma)
            - 0.002697 * math.cos(3 * gamma) + 0.00148 * math.sin(3 * gamma))
    latr = math.radians(LAT)
    zen = math.radians(96.0)
    cos_ha = (math.cos(zen) - math.sin(latr) * math.sin(decl)) / (math.cos(latr) * math.cos(decl))
    ha = math.degrees(math.acos(max(-1.0, min(1.0, cos_ha))))
    dawn_min = 720 - 4 * (LON + ha) - eqtime     # minutes after 00:00 UTC on date d
    dusk_min = 720 - 4 * (LON - ha) - eqtime     # can exceed 1440 (next UTC day)
    base = datetime.combine(d, time(0), tzinfo=timezone.utc)
    return (base + timedelta(minutes=dawn_min)).replace(microsecond=0), \
           (base + timedelta(minutes=dusk_min)).replace(microsecond=0)


def build_periods(first_local, last_local):
    """List of (local_date, start_utc, end_utc, daylight), contiguous in time."""
    periods = []
    d = first_local
    while d <= last_local:
        midnight = datetime.combine(d, time(0), tzinfo=TZ).astimezone(timezone.utc)
        next_mid = datetime.combine(d + timedelta(days=1), time(0), tzinfo=TZ).astimezone(timezone.utc)
        dawn, dusk = civil_dawn_dusk_utc(d)
        if not (midnight < dawn < dusk < next_mid):
            sys.exit(f"ERROR: unexpected dawn/dusk order on {d}: {dawn} {dusk}")
        periods.append((d, midnight, dawn, 0))
        periods.append((d, dawn, dusk, 1))
        periods.append((d, dusk, next_mid, 0))
        d += timedelta(days=1)
    return periods


# ---------------------------------------------------------------- databases
def find_dbs(root):
    """All DBs as {(first_day, last_day): dir}, preferring _norm when both exist.
    Accepts any start day (e.g. MARS_20150728_20150731_32kHz_norm)."""
    chosen = {}
    for p in sorted(Path(root).iterdir()):
        m = DB_RE.match(p.name)
        if m and (p / "hoplite.sqlite").exists():
            key = (datetime.strptime(m.group(1), "%Y%m%d").date(),
                   datetime.strptime(m.group(2), "%Y%m%d").date())
            if key not in chosen or (m.group(3) and not chosen[key][1]):
                chosen[key] = (p, bool(m.group(3)))
    return {k: v[0] for k, v in sorted(chosen.items())}


def first_offset(blob):
    if not blob:
        return 0.0
    n = len(blob) // 8
    return struct.unpack(f"<{n}d", bytes(blob[: n * 8]))[0] if n else 0.0


def file_start_utc(filename):
    m = STEM_RE.search(filename)
    if not m:
        return None, None
    t = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    return m.group(0), t


def read_db(db_dir, t0, t1, seen):
    """Return (intervals, detections) for recordings starting in [t0, t1)."""
    con = sqlite3.connect(f"file:{db_dir / 'hoplite.sqlite'}?mode=ro", uri=True)
    try:
        recs = con.execute(
            "SELECT r.filename, COUNT(w.id) FROM recordings r "
            "LEFT JOIN windows w ON w.recording_id = r.id GROUP BY r.id").fetchall()
        cols = [c[1] for c in con.execute("PRAGMA table_info(annotations)")]
        sql = ("SELECT r.filename, a.offsets FROM annotations a "
               "JOIN recordings r ON r.id = a.recording_id WHERE a.label = 'orca_call'")
        if "label_type" in cols:
            sql += " AND a.label_type = 1"
        anns = con.execute(sql).fetchall()
    finally:
        con.close()

    intervals = []
    for fn, nwin in recs:
        stem, t = file_start_utc(fn)
        if t is None or not (t0 <= t < t1) or nwin == 0:
            continue
        intervals.append((t, t + timedelta(seconds=nwin * WINDOW_S)))

    dets = []
    for fn, blob in anns:
        stem, t = file_start_utc(fn)
        if t is None or not (t0 <= t < t1):
            continue
        off = round(first_offset(blob), 1)
        if (stem, off) in seen:
            continue
        seen.add((stem, off))
        dets.append(t + timedelta(seconds=off))
    return intervals, dets


def merge(intervals):
    out = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db-root", default="/mnt/PAM_Analysis/perch-hoplite/db")
    ap.add_argument("--utc-start", default="2015-07-01", help="first UTC day analyzed")
    ap.add_argument("--utc-end", default="2023-12-31", help="last UTC day analyzed (inclusive)")
    ap.add_argument("--daily-csv", help="daily UTC CSV to validate against (optional)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if os.path.exists(args.out) and not args.overwrite:
        sys.exit(f"ERROR: {args.out} exists; pass --overwrite to replace it.")

    t0 = datetime.strptime(args.utc_start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    t1 = datetime.strptime(args.utc_end, "%Y-%m-%d").replace(tzinfo=timezone.utc) + timedelta(days=1)

    dbs = find_dbs(args.db_root)
    used = {k: v for k, v in dbs.items() if k[1] >= t0.date() and k[0] < t1.date()}
    covered = set()
    for a, b in used:
        d = a
        while d <= b:
            covered.add(d)
            d += timedelta(days=1)
    missing = []
    d = t0.date()
    while d < t1.date():
        if d not in covered:
            missing.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    print(f"DBs used: {len(used)}   UTC days in range with no DB: {len(missing)}")
    if missing:
        print("No DB (treated as no data): " + " ".join(missing[:40])
              + (" ..." if len(missing) > 40 else ""))

    intervals, dets, seen = [], [], set()
    for key, path in used.items():
        iv, dt = read_db(path, t0, t1, seen)
        intervals += iv
        dets += dt
        print(f"  {key[0]}..{key[1]}  {path.name:40s} recordings {len(iv):5d}  orca windows {len(dt):5d}")
    merged = merge(intervals)

    periods = build_periods(t0.astimezone(TZ).date(), (t1 - timedelta(seconds=1)).astimezone(TZ).date())
    starts = [p[1] for p in periods]

    counts = [0] * len(periods)
    for t in dets:
        i = bisect.bisect_right(starts, t) - 1
        if i < 0 or t >= periods[i][2]:
            sys.exit(f"ERROR: detection {t} outside the period range")
        counts[i] += 1

    secs = [0.0] * len(periods)
    j = 0
    for i, (_, ps, pe, _) in enumerate(periods):
        while j < len(merged) and merged[j][1] <= ps:
            j += 1
        k = j
        while k < len(merged) and merged[k][0] < pe:
            secs[i] += (min(pe, merged[k][1]) - max(ps, merged[k][0])).total_seconds()
            k += 1

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["YYYYMMDD", "start_local", "end_local", "daylight",
                    "orca_verified", "data_available", "recorded_hours"])
        for (d, ps, pe, day), c, s in zip(periods, counts, secs):
            w.writerow([d.strftime("%Y%m%d"),
                        ps.astimezone(TZ).isoformat(timespec="seconds"),
                        pe.astimezone(TZ).isoformat(timespec="seconds"),
                        day, c, int(s > 0), f"{s / 3600:.4f}"])

    print(f"\nWrote {len(periods)} rows ({len(periods) // 3} local dates) to {args.out}")
    for day, name in [(1, "daylight"), (0, "night")]:
        n = sum(c for c, p in zip(counts, periods) if p[3] == day)
        h = sum(s for s, p in zip(secs, periods) if p[3] == day) / 3600
        print(f"  {name:9s} verified windows {n:6d}   recorded hours {h:10.1f}   per hour {n / h if h else 0:.4f}")
    print(f"  total     verified windows {len(dets):6d}")

    if args.daily_csv:
        utc_counts = defaultdict(int)
        for t in dets:
            utc_counts[t.strftime("%Y%m%d")] += 1
        utc_data = set()
        for s, e in merged:
            d = s.date()
            while d <= (e - timedelta(seconds=1)).date():
                utc_data.add(d.strftime("%Y%m%d"))
                d += timedelta(days=1)
        bad = []
        with open(args.daily_csv, newline="") as f:
            for r in csv.DictReader(f):
                k = r["YYYYMMDD"].strip()
                exp_c, exp_a = int(r["orca_verified"]), int(r["data_available"])
                got_c, got_a = utc_counts.get(k, 0), int(k in utc_data)
                if exp_c != got_c or exp_a != got_a:
                    bad.append([k, exp_c, got_c, exp_a, got_a])
        vpath = os.path.splitext(args.out)[0] + "_validation.csv"
        with open(vpath, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["YYYYMMDD_utc", "daily_orca", "db_orca", "daily_available", "db_available"])
            w.writerows(bad)
        print(f"\nValidation against {args.daily_csv}: {len(bad)} mismatched UTC days "
              f"(details in {vpath})")
        for row in bad[:20]:
            print("  ", row)


if __name__ == "__main__":
    main()
