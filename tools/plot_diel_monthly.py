#!/usr/bin/env python3
"""tools/plot_diel_monthly.py

Generate a time-of-day (diel) scatter plot for one month showing:
  - Confirmed orca calls (green) at their exact local clock time, read from the hoplite DB
  - Unconfirmed / "other" labeled calls (red), also from the DB
  - Surface sightings (blue, sized by count) passed on the command line
  - A per-day civil-twilight night band (shaded), computed from the NOAA solar algorithm
  - A DST change line if the month straddles a DST boundary

This script is a companion to tools/plot_diel_vs_sightings.py which is the canonical
single-month tool. This version adds:
  - "other" label rendering in red
  - DST boundary line (--dst-day argument)
  - Cleaner legend placement options

USAGE EXAMPLES

  # January 2016 (PST throughout, no DST)
  python3 tools/plot_diel_monthly.py \
      --db /mnt/PAM_Analysis/perch-hoplite/db/MARS_20160101_20160131_32kHz_norm/hoplite.sqlite \
      --year 2016 --month 1 --utc-offset -8 \
      --title "January 2016" \
      --sighting 1 13.5 5 --sighting 11 14 100 --sighting 15 10 300 \
      --sighting 16 10 7 --sighting 23 10 3 \
      --out figures/panel_january2016.png

  # March 2016 (PST before Mar 13, PDT after)
  python3 tools/plot_diel_monthly.py \
      --db /mnt/PAM_Analysis/perch-hoplite/db/MARS_20160301_20160331_32kHz_norm/hoplite.sqlite \
      --year 2016 --month 3 --utc-offset -7 \
      --dst-day 13 --dst-before-offset -8 \
      --title "March 2016" \
      --sighting 1 10 5 --sighting 10 10 6 --sighting 29 10 6 \
      --gap 0 0 \
      --out figures/panel_march2016.png

  # February 2016 (no orca, but sightings — shows empty acoustic record)
  python3 tools/plot_diel_monthly.py \
      --db /mnt/PAM_Analysis/perch-hoplite/db/MARS_20160201_20160229_32kHz_norm/hoplite.sqlite \
      --year 2016 --month 2 --utc-offset -8 \
      --title "February 2016" \
      --sighting 12 14 13 --sighting 20 10 9 --sighting 28 10 6 \
      --gap 3.5 8.5 --gap 12.5 13.5 \
      --out figures/panel_february2016.png

ARGUMENTS
  --db PATH             hoplite.sqlite for the month
  --year INT            calendar year
  --month INT           calendar month (1-12)
  --utc-offset FLOAT    local offset from UTC (e.g. -8 PST, -7 PDT)
  --title STR           plot title (default: "Month YYYY")
  --sighting D H N      surface sighting: day-of-month, local hour, count (repeatable)
  --gap X0 X1           data-gap hatch span in day coords (repeatable)
  --dst-day INT         day-of-month when DST changes (draws a dashed vertical line)
  --dst-before-offset F UTC offset BEFORE the DST change (used for civil twilight before --dst-day)
  --legend-loc STR      matplotlib legend location string (default: "upper left")
  --out PATH            output PNG path

NOTES
  - Call times are read directly from the DB — no hand-transcription.
  - Sightings are passed as CLI args so no copyrighted data enters the repo.
  - Night band uses the NOAA solar algorithm (embedded, no network required).
  - Figures plotting MBWW / CKWP sighting counts are INTERNAL ONLY — do not
    commit to the public repo without written permission from Nancy Black / CKWP.
"""
import argparse
import calendar
import math
import sqlite3
import struct
import random
from datetime import date, datetime, timedelta

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ── NOAA civil twilight (embedded — no external dependency) ───────────
def civil_dawn_dusk(d, lat=36.7125, lon=-122.1868, tz_offset=-8.0):
    """Local decimal-hour civil dawn and dusk for date d.
    NOAA solar-position algorithm; accurate to ~1 minute.
    Returns (dawn, dusk). Polar edges: (0,24) continuous light, (None,None) none."""
    N = d.timetuple().tm_yday
    gamma = 2 * math.pi / 365 * (N - 1 + 0.5)
    eqtime = 229.18 * (0.000075 + 0.001868 * math.cos(gamma)
                       - 0.032077 * math.sin(gamma)
                       - 0.014615 * math.cos(2*gamma)
                       - 0.040849 * math.sin(2*gamma))
    decl = (0.006918 - 0.399912*math.cos(gamma) + 0.070257*math.sin(gamma)
            - 0.006758*math.cos(2*gamma) + 0.000907*math.sin(2*gamma)
            - 0.002697*math.cos(3*gamma) + 0.00148*math.sin(3*gamma))
    latr = math.radians(lat)
    zen = math.radians(96.0)
    cos_ha = (math.cos(zen) - math.sin(latr)*math.sin(decl)) / (math.cos(latr)*math.cos(decl))
    if cos_ha > 1:
        return None, None
    if cos_ha < -1:
        return 0.0, 24.0
    ha = math.degrees(math.acos(cos_ha))
    return (720 - 4*(lon+ha) - eqtime)/60 + tz_offset, \
           (720 - 4*(lon-ha) - eqtime)/60 + tz_offset


# ── Read confirmed calls from DB ──────────────────────────────────────
def load_calls(db_path, utc_offset, dst_day=None, dst_before_offset=None, label="orca_call"):
    con = sqlite3.connect(db_path)
    rows = con.execute(
        "SELECT r.filename, a.offsets, a.label FROM annotations a "
        "JOIN recordings r ON r.id=a.recording_id WHERE a.label=?",
        (label,)).fetchall()
    con.close()
    pts = []
    for fn, blob, _ in rows:
        try:
            start = struct.unpack("<2d", blob)[0] if blob else 0.0
        except Exception:
            start = 0.0
        day_num = int(fn[13:15])
        off = utc_offset
        if dst_day is not None and dst_before_offset is not None and day_num < dst_day:
            off = dst_before_offset
        t = datetime.strptime(fn[5:20], "%Y%m%d_%H%M%S") + timedelta(seconds=start) \
            + timedelta(hours=off)
        pts.append((t.day, t.hour + t.minute/60 + t.second/3600))
    return pts


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", required=True)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--month", type=int, required=True)
    ap.add_argument("--utc-offset", type=float, required=True)
    ap.add_argument("--title", default=None)
    ap.add_argument("--lat", type=float, default=36.7125)
    ap.add_argument("--lon", type=float, default=-122.1868)
    ap.add_argument("--sighting", nargs=3, action="append", type=float,
                    metavar=("DAY", "HOUR", "COUNT"), default=[])
    ap.add_argument("--gap", nargs=2, action="append", type=float,
                    metavar=("X0", "X1"), default=[])
    ap.add_argument("--dst-day", type=int, default=None,
                    help="Day-of-month when DST changes (draws dashed line)")
    ap.add_argument("--dst-before-offset", type=float, default=None,
                    help="UTC offset BEFORE the DST change day")
    ap.add_argument("--legend-loc", default="upper left")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    nd = calendar.monthrange(args.year, args.month)[1]
    title = args.title or f"{calendar.month_name[args.month]} {args.year}"

    orca = load_calls(args.db, args.utc_offset, args.dst_day, args.dst_before_offset,
                      label="orca_call")
    other = load_calls(args.db, args.utc_offset, args.dst_day, args.dst_before_offset,
                       label="other")

    NIGHT = "#c7d2e0"; DAYC = "#fffdf3"; GRID = "#94a3b8"
    TXT = "#0f172a"; AX = "#334155"
    GREEN = "#16a34a"; RED = "#dc2626"; BLUE = "#2563eb"; HC = "#64748b"

    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor(DAYC)

    days = list(range(1, nd+1))
    dawn, dusk = [], []
    for dd in days:
        off = args.utc_offset
        if args.dst_day and args.dst_before_offset and dd < args.dst_day:
            off = args.dst_before_offset
        dw, du = civil_dawn_dusk(date(args.year, args.month, dd),
                                 lat=args.lat, lon=args.lon, tz_offset=off)
        dawn.append(dw if dw is not None else 0.0)
        dusk.append(du if du is not None else 24.0)

    xs = np.array([d-0.5 for d in days] + [nd+0.5])
    ax.fill_between(xs, 0, np.array(dawn+[dawn[-1]]), step="post", color=NIGHT, zorder=0)
    ax.fill_between(xs, np.array(dusk+[dusk[-1]]), 24, step="post", color=NIGHT, zorder=0)

    for x0, x1 in args.gap:
        ax.axvspan(x0, x1, facecolor="none", edgecolor=HC, hatch="////", lw=0, alpha=0.7, zorder=1)

    if args.dst_day:
        ax.axvline(args.dst_day, color="#f59e0b", lw=1.5, linestyle="--", alpha=0.7, zorder=2)
        ax.text(args.dst_day+0.2, 23, "DST →", fontsize=13, color="#f59e0b", va="top")

    random.seed(1)
    if orca:
        ax.scatter([d+random.uniform(-0.3, 0.3) for d, h in orca],
                   [h for d, h in orca], s=90, color=GREEN, edgecolor="#0f172a",
                   lw=0.6, alpha=0.85, zorder=5, label=f"Orca call confirmed ({len(orca)})")
    else:
        ax.scatter([], [], s=90, color=GREEN, edgecolor="#0f172a", lw=0.6,
                   label="Orca call confirmed (0)")

    if other:
        ax.scatter([d+random.uniform(-0.3, 0.3) for d, h in other],
                   [h for d, h in other], s=90, color=RED, edgecolor="#0f172a",
                   lw=0.6, alpha=0.85, zorder=5,
                   label=f"Acoustic — unconfirmed / other ({len(other)})")

    for d, h, n in args.sighting:
        ax.scatter([d], [h], s=max(150, n*8), color=BLUE, edgecolor="#0f172a",
                   lw=0.8, alpha=0.85, zorder=4)
        if n >= 3:
            ax.text(d, h, str(int(n)), ha="center", va="center", fontsize=11,
                    color="white", fontweight="700", zorder=6)
    if args.sighting:
        ax.scatter([], [], s=200, color=BLUE, edgecolor="#0f172a", lw=0.8,
                   label="KW sighted (size = count)")

    if args.gap:
        handles, labels = ax.get_legend_handles_labels()
        handles.append(mpatches.Patch(facecolor="white", edgecolor=HC, hatch="////",
                                      label="No acoustic data (recorder dropout)"))
        ax.legend(handles=handles, loc=args.legend_loc, fontsize=15,
                  framealpha=0.96, edgecolor=GRID)
    else:
        ax.legend(loc=args.legend_loc, fontsize=15, framealpha=0.96, edgecolor=GRID)

    ax.set_xlim(0.5, nd+0.5); ax.set_ylim(0, 24)
    ax.set_xticks([1, 5, 10, 15, 20, 25, nd])
    ax.set_yticks(range(0, 25, 3))
    ax.set_yticklabels([f"{h:02d}:00" for h in range(0, 25, 3)])
    ax.tick_params(labelsize=18, colors=AX)
    ax.set_xlabel("Day of month", fontsize=21, color=AX)
    ax.set_ylabel("Time of day (local)", fontsize=21, color=AX)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4, zorder=1)
    for sp in ax.spines.values():
        sp.set_color(GRID)

    fig.suptitle(f"{title} — killer whale detections & sightings by time of day",
                 fontsize=27, color=TXT, fontweight="700", y=0.98)
    ax.set_title("Shaded = night (civil twilight, per day) · "
                 "green = MARS acoustic · red = unconfirmed · blue = surface sighting",
                 fontsize=15, color=AX, pad=10)
    fig.text(0.5, 0.005,
             f"Night = civil twilight (NOAA solar algorithm, {args.lat:.2f}N "
             f"{abs(args.lon):.2f}W). "
             "Sightings © Monterey Bay Whale Watch / Nancy Black — internal only.",
             ha="center", fontsize=12, color=AX)

    fig.savefig(args.out, dpi=150, facecolor="#ffffff", bbox_inches="tight")
    print(f"Wrote {args.out}  ({len(orca)} orca, {len(other)} other, "
          f"{len(args.sighting)} sightings)")


if __name__ == "__main__":
    main()
