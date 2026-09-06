#!/usr/bin/env python3
"""
may_per_day_v4_v10.py — per-day May 2018 detection counts for v4 and v10, all 31 days.

Serves DATA_REQUEST_v10_may_per_day.md: a talk slide comparing v4 vs v10 day-by-day on the
held-out month, with confirmed-orca ground truth per day.

Reads (no re-inference — these already exist):
  - v10 inference CSV: MARS_20180501_20180531_v10_orcaval.csv (floor 0.0)
  - v4  inference CSV: MARS_20180501_20180531_v4_orcaval.csv  (floor 0.0)
  - confirmed orca windows: the May DB annotations (label=orca_call, label_type=1)

Emits one row per day for all 31 days (including zeros), columns:
  date, v4_det_T0.00, v4_det_T1.16, v4_det_T2.31,
        v10_det_T0.00, v10_det_T1.16, v10_det_T2.31,
        confirmed_clips, notes

CSV cols in the inference files: idx,project,filename,window_start,window_end,label,logits
Day is parsed from the filename: MARS_YYYYMMDD_HHMMSS_resampled_32kHz.wav -> substr(6,8).
"""
import csv, sqlite3, struct
from pathlib import Path
from collections import defaultdict

DB  = "/mnt/PAM_Analysis/perch-hoplite/db/MARS_20180501_20180531_32kHz_norm/hoplite.sqlite"
V10 = "/mnt/PAM_Analysis/perch-hoplite/results/MARS_20180501_20180531_v10_orcaval.csv"
V4  = "/mnt/PAM_Analysis/perch-hoplite/results/MARS_20180501_20180531_v4_orcaval.csv"
OUT = "/mnt/PAM_Analysis/perch-hoplite/results/may2018_per_day_v4_v10.csv"

THRESHOLDS = [0.0, 1.16, 2.31]
DAYS = [f"201805{d:02d}" for d in range(1, 32)]   # 20180501 .. 20180531


def day_of(filename):
    # MARS_20180512_....wav -> 20180512
    b = Path(filename).name
    return b[5:13]


def count_by_day(csv_path):
    """Return {day: {thresh: count}} for orca_call rows."""
    counts = {d: {t: 0 for t in THRESHOLDS} for d in DAYS}
    with open(csv_path) as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("label") != "orca_call":
                continue
            day = day_of(row["filename"])
            if day not in counts:
                continue
            logit = float(row["logits"])
            for t in THRESHOLDS:
                if logit >= t:
                    counts[day][t] += 1
    return counts


def confirmed_by_day(db_path):
    """Return {day: count} of confirmed orca_call windows (label_type=1)."""
    con = sqlite3.connect(db_path)
    rows = con.execute(
        "SELECT r.filename FROM annotations a JOIN recordings r ON r.id=a.recording_id "
        "WHERE a.label='orca_call' AND a.label_type=1"
    ).fetchall()
    con.close()
    out = defaultdict(int)
    for (fn,) in rows:
        out[day_of(fn)] += 1
    return out


def main():
    for p in (DB, V10, V4):
        if not Path(p).exists():
            raise SystemExit(f"MISSING: {p}")

    v10 = count_by_day(V10)
    v4  = count_by_day(V4)
    conf = confirmed_by_day(DB)

    new_days = {"20180502", "20180503", "20180507", "20180529"}

    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "date",
            "v4_det_T0.00", "v4_det_T1.16", "v4_det_T2.31",
            "v10_det_T0.00", "v10_det_T1.16", "v10_det_T2.31",
            "confirmed_clips", "notes",
        ])
        for d in DAYS:
            iso = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
            note = ""
            if d in new_days:
                note = "NEW day found by v10 (v4 missed)"
            elif d in {"20180512", "20180513", "20180514", "20180516"}:
                note = "known orca day"
            w.writerow([
                iso,
                v4[d][0.0], v4[d][1.16], v4[d][2.31],
                v10[d][0.0], v10[d][1.16], v10[d][2.31],
                conf.get(d, 0), note,
            ])

    # console summary
    print(f"Wrote {OUT}")
    print(f"{'date':12} {'v4@1.16':>8} {'v10@1.16':>9} {'v10@2.31':>9} {'confirmed':>10}  notes")
    tot = defaultdict(int)
    for d in DAYS:
        iso = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
        c = conf.get(d, 0)
        note = ("NEW (v10)" if d in new_days else
                "known" if d in {"20180512","20180513","20180514","20180516"} else "")
        if v4[d][1.16] or v10[d][1.16] or c:
            print(f"{iso:12} {v4[d][1.16]:>8} {v10[d][1.16]:>9} {v10[d][2.31]:>9} {c:>10}  {note}")
        tot['v4_116'] += v4[d][1.16]; tot['v10_116'] += v10[d][1.16]
        tot['v10_231'] += v10[d][2.31]; tot['conf'] += c
    print(f"{'TOTAL':12} {tot['v4_116']:>8} {tot['v10_116']:>9} {tot['v10_231']:>9} {tot['conf']:>10}")
    print("\nDays with confirmed>0:", sum(1 for d in DAYS if conf.get(d,0) > 0))
    print("Item 3 check (new days each carry how many @>=1.16):",
          {f"{d[4:6]}-{d[6:8]}": v10[d][1.16] for d in sorted(new_days)})


if __name__ == "__main__":
    main()

