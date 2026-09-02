#!/usr/bin/env python3
"""tools/plot_diel_vs_sightings.py

Time-of-day (diel) scatter of confirmed orca calls vs. surface sightings for one month,
with a per-day civil-twilight night band computed from the NOAA solar algorithm.

WHY THIS PLOT: across Sep-Nov 2015 every confirmed acoustic encounter fell at night and
every whale-watch sighting fell in daylight -- the two records barely overlap (finding #46).
A day-of-month vs. time-of-day scatter with the night region shaded makes that pattern
legible at a glance, which a per-day bar chart did not (J. Ryan could not read the bar version).

DATA SOURCES
  - Acoustic: confirmed orca_call annotations read straight from the month's hoplite DB.
    Each call is placed at recording-start + window-offset, converted to LOCAL time.
  - Sightings: passed in on the command line (--sighting DAY HOUR COUNT ...), because the
    MBWW / CKWP sighting lists are copyright and must NOT be committed to a public repo.
    The plot renders them but the numbers live only on the user's machine.

TIME ZONE: pass --utc-offset. Monterey is PDT (-7) through the DST change (Nov 1 2015),
PST (-8) after. The tool does not guess; you pass the offset for the month.

NIGHT BAND: civil twilight (sun 6 deg below horizon), computed per calendar day via the
NOAA solar-position algorithm (self-contained below -- no network, no dependencies beyond
numpy/matplotlib). Default location is the MARS node (36.7125 N, 122.1868 W).

The output figure plots copyrighted sighting counts and therefore should be treated as
INTERNAL (do not commit to the public repo). The TOOL is fine to commit.

USAGE
  python3 tools/plot_diel_vs_sightings.py \
      --db /mnt/PAM_Analysis/perch-hoplite/db/MARS_20151101_20151130_32kHz_norm/hoplite.sqlite \
      --year 2015 --month 11 --utc-offset -8 \
      --title "November 2015" \
      --sighting 5 11 25 --sighting 12 11 7 --sighting 13 11 7 \
      --gap 14.5 15.5 \
      --out figures/panel_november2015.png

  # acoustic only, no sightings:
  python3 tools/plot_diel_vs_sightings.py --db ... --year 2015 --month 10 --utc-offset -7 \
      --title "October 2015" --gap 17.5 18.5 --out figures/panel_october2015.png
"""
import argparse
import calendar
import math
import sqlite3
import struct
import sys
from datetime import date, datetime, timedelta

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ── NOAA civil twilight ────────────────────────────────────────────────
def civil_dawn_dusk(d, lat=36.7125, lon=-122.1868, tz_offset=-8.0):
    """Local decimal-hour civil dawn and dusk (sun 6 deg below horizon) for date d.
    NOAA solar-position algorithm; accurate to ~1 minute. Returns (dawn, dusk).
    Polar edge cases return (0,24) for continuous light or (None,None) for none."""
    N = d.timetuple().tm_yday
    gamma = 2 * math.pi / 365 * (N - 1 + 0.5)
    eqtime = 229.18 * (0.000075 + 0.001868 * math.cos(gamma)
                       - 0.032077 * math.sin(gamma) - 0.014615 * math.cos(2 * gamma)
                       - 0.040849 * math.sin(2 * gamma))
    decl = (0.006918 - 0.399912 * math.cos(gamma) + 0.070257 * math.sin(gamma)
            - 0.006758 * math.cos(2 * gamma) + 0.000907 * math.sin(2 * gamma)
            - 0.002697 * math.cos(3 * gamma) + 0.00148 * math.sin(3 * gamma))
    latr = math.radians(lat)
    zen = math.radians(96.0)  # civil twilight
    cos_ha = (math.cos(zen) - math.sin(latr) * math.sin(decl)) / (math.cos(latr) * math.cos(decl))
    if cos_ha > 1:
        return None, None
    if cos_ha < -1:
        return 0.0, 24.0
    ha = math.degrees(math.acos(cos_ha))
    sunrise_utc = 720 - 4 * (lon + ha) - eqtime
    sunset_utc = 720 - 4 * (lon - ha) - eqtime
    return sunrise_utc / 60 + tz_offset, sunset_utc / 60 + tz_offset


# ── read confirmed calls from the DB ───────────────────────────────────
def load_calls(db_path, utc_offset, label="orca_call"):
    con = sqlite3.connect(db_path)
    rows = con.execute(
        "SELECT r.filename, a.offsets FROM annotations a "
        "JOIN recordings r ON r.id = a.recording_id WHERE a.label = ?",
        (label,)).fetchall()
    con.close()
    pts = []
    for fn, blob in rows:
        try:
            start = struct.unpack("<2d", blob)[0] if blob else 0.0
        except Exception:
            start = 0.0
        # filename form: MARS_YYYYMMDD_HHMMSS_...
        t = datetime.strptime(fn[5:20], "%Y%m%d_%H%M%S") + timedelta(seconds=start) \
            + timedelta(hours=utc_offset)
        pts.append((t.day, t.hour + t.minute / 60 + t.second / 3600))
    return pts


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", required=True, help="month hoplite.sqlite")
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--month", type=int, required=True)
    ap.add_argument("--utc-offset", type=float, required=True,
                    help="local offset from UTC, e.g. -7 (PDT) or -8 (PST)")
    ap.add_argument("--label", default="orca_call")
    ap.add_argument("--title", default=None)
    ap.add_argument("--lat", type=float, default=36.7125)
    ap.add_argument("--lon", type=float, default=-122.1868)
    ap.add_argument("--sighting", nargs=3, action="append", type=float, metavar=("DAY", "HOUR", "COUNT"),
                    default=[], help="a surface sighting: day-of-month, local hour, count. Repeatable.")
    ap.add_argument("--gap", nargs=2, action="append", type=float, metavar=("X0", "X1"),
                    default=[], help="data-gap span (day coords) to hatch. Repeatable.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    nd = calendar.monthrange(args.year, args.month)[1]
    pts = load_calls(args.db, args.utc_offset, args.label)
    title = args.title or f"{calendar.month_name[args.month]} {args.year}"

    NIGHT = "#c7d2e0"; DAYC = "#fffdf3"; GRID = "#94a3b8"; TXT = "#0f172a"
    AX = "#334155"; GREEN = "#16a34a"; BLUE = "#2563eb"; HC = "#64748b"

    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor(DAYC)

    days = list(range(1, nd + 1))
    dawn, dusk = [], []
    for dd in days:
        dw, du = civil_dawn_dusk(date(args.year, args.month, dd),
                                 lat=args.lat, lon=args.lon, tz_offset=args.utc_offset)
        dawn.append(dw if dw is not None else 0.0)
        dusk.append(du if du is not None else 24.0)
    xs = np.array([d - 0.5 for d in days] + [nd + 0.5])
    ax.fill_between(xs, 0, np.array(dawn + [dawn[-1]]), step="post", color=NIGHT, zorder=0)
    ax.fill_between(xs, np.array(dusk + [dusk[-1]]), 24, step="post", color=NIGHT, zorder=0)

    for x0, x1 in args.gap:
        ax.axvspan(x0, x1, facecolor="none", edgecolor=HC, hatch="////", lw=0, alpha=0.7, zorder=1)

    # night/day tally for the caption
    n_night = 0
    for d, h in pts:
        dw, du = civil_dawn_dusk(date(args.year, args.month, int(d)),
                                 lat=args.lat, lon=args.lon, tz_offset=args.utc_offset)
        if not (dw <= h < du):
            n_night += 1

    import random
    random.seed(1)
    if pts:
        ax.scatter([d + random.uniform(-0.32, 0.32) for d, h in pts],
                   [h for d, h in pts], s=90, color=GREEN, edgecolor="#0f172a",
                   lw=0.6, alpha=0.85, zorder=5,
                   label=f"Orca call confirmed ({len(pts)}; {n_night} at night)")

    for d, h, n in args.sighting:
        ax.scatter([d], [h], s=max(150, n * 22), color=BLUE, edgecolor="#0f172a",
                   lw=0.8, alpha=0.85, zorder=4)
        ax.text(d, h, str(int(n)), ha="center", va="center", fontsize=13,
                color="white", fontweight="700", zorder=6)
    if args.sighting:
        ax.scatter([], [], s=200, color=BLUE, edgecolor="#0f172a", lw=0.8,
                   label="KW sighted (size = count)")

    ax.set_xlim(0.5, nd + 0.5); ax.set_ylim(0, 24)
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
    ax.set_title("Shaded = night (civil twilight, computed per day) · "
                 "green = MARS acoustic · blue = surface sighting",
                 fontsize=16, color=AX, pad=10)
    ax.legend(loc="upper left", fontsize=16, framealpha=0.96, edgecolor=GRID)

    fig.text(0.5, 0.005,
             f"Night = civil twilight (NOAA solar algorithm, {args.lat:.2f} N {abs(args.lon):.2f} W). "
             "Sighting data is third-party (e.g. MBWW / Nancy Black) — internal use only, "
             "not for redistribution.",
             ha="center", fontsize=12, color=AX)

    fig.savefig(args.out, dpi=150, facecolor="#ffffff", bbox_inches="tight")
    print(f"Wrote {args.out}  ({len(pts)} calls, {n_night} at night, {len(args.sighting)} sightings)")


if __name__ == "__main__":
    main()
