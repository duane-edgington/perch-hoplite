#!/usr/bin/env python3
"""
generate_orca_histograms.py

Daily histograms of confirmed orca_call annotations from the perch-hoplite
per-month databases, for the MARS orca campaign.

Set 1 (one figure per YEAR, 12 monthly panels each)
    Bar chart, x = day of month, y = confirmed orca clips (raw count by default).
    Output: <out-dir>/set1_by_year/orca_daily_YYYY.png

Set 2 (one figure per CALENDAR MONTH, one line per year)
    Line plot, x = day of month, y = confirmed orca clips per recorded hour
    (coverage-normalized by default). Every analyzed year appears in the legend
    with a fixed color and its total, "(n=0)" included; years with no confirmed
    orca are drawn as a flat line on zero.
    Output: <out-dir>/set2_by_month/orca_daily_MM_Month.png

Also writes <out-dir>/orca_daily_counts.csv (one row per calendar day of every
included month). Re-plot without touching the databases using --reuse-counts.

Conventions
    * Day = UTC date of the recording (same convention as the campaign tables,
      which use the date in the MARS_YYYYMMDD_HHMMSS filename).
    * One annotation = one 5-second window. Duplicate (recording, offset) rows
      are counted once.
    * Only label_type 1 (positive) rows are counted by default. Some DBs (for
      example Apr 2018) also hold label_type 2 rows for orca_call (the 54 extra
      rows in Apr 2018 beyond the 374 confirmed). Use --all-label-types to count
      every type, or --label-type N for a specific one.
    * Per-month DBs only: MARS_YYYYMM01_YYYYMMDD_32kHz[_norm]. If both exist for
      a month, the _norm DB is used. Combined/test/partial-month DBs are ignored.
    * Hours recorded come from results/coverage/YYYY-MM_coverage.csv
      (needs 'date' and 'hours' columns).

Review status. A month counts as reviewed when its pass1_orca cell is filled in
in campaign_pass_statistics.csv (--stats-csv). A reviewed month with no confirmed
orca is shown as "no confirmed orca"; a month with no recorded review and no
confirmed clips is shown as "not reviewed" (gray panel in Set 1, a legend entry
with no line in Set 2). Without the stats CSV, no review status is shown.

Counts reflect review effort (pass-1, plus pass-2 where it was done), not only
orca presence. Months after --through are excluded.

Examples
    python3 generate_orca_histograms.py
    python3 generate_orca_histograms.py --years 2023
    python3 generate_orca_histograms.py --months 4 6 --raw-set2
    python3 generate_orca_histograms.py --normalize-set1
    python3 generate_orca_histograms.py --reuse-counts
"""

from __future__ import annotations

import argparse
import calendar
import csv
import math
import re
import sqlite3
import struct
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

DB_RE = re.compile(r"^MARS_(\d{4})(\d{2})01_\d{8}_32kHz(_norm)?$")
FILE_RE = re.compile(r"MARS_(\d{8})_(\d{6})")

NAN = float("nan")


# ----------------------------------------------------------------------------
# Reading the databases
# ----------------------------------------------------------------------------

def parse_ym(text: str) -> tuple[int, int]:
    m = re.fullmatch(r"(\d{4})-(\d{2})", text)
    if not m:
        raise argparse.ArgumentTypeError(f"expected YYYY-MM, got {text!r}")
    return int(m.group(1)), int(m.group(2))


def find_month_dbs(db_root: Path) -> dict[tuple[int, int], Path]:
    """Map (year, month) -> DB directory; prefer the _norm DB when both exist."""
    chosen: dict[tuple[int, int], tuple[Path, bool]] = {}
    for d in sorted(db_root.iterdir()):
        m = DB_RE.match(d.name)
        if not m or not (d / "hoplite.sqlite").exists():
            continue
        key = (int(m.group(1)), int(m.group(2)))
        is_norm = m.group(3) is not None
        if key not in chosen or (is_norm and not chosen[key][1]):
            chosen[key] = (d, is_norm)
    return {k: v[0] for k, v in sorted(chosen.items())}


def first_offset(blob) -> float:
    """Start time (s) of the window: first double in the offsets blob."""
    if not blob:
        return 0.0
    n = len(blob) // 8
    if n == 0:
        return 0.0
    return struct.unpack(f"<{n}d", bytes(blob[: n * 8]))[0]


def read_orca_clips(db_dir: Path, label: str, label_type: int | None):
    """Return (list of clip datetimes, {label_type: rows}, raw row count)."""
    uri = f"file:{db_dir / 'hoplite.sqlite'}?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    try:
        rows = con.execute(
            "SELECT r.filename, a.offsets, a.label_type "
            "FROM annotations a JOIN recordings r ON r.id = a.recording_id "
            "WHERE a.label = ?",
            (label,),
        ).fetchall()
    finally:
        con.close()

    type_counts: dict[int, int] = defaultdict(int)
    seen: set[tuple[str, float]] = set()
    times: list[datetime] = []
    for filename, blob, ltype in rows:
        type_counts[ltype] += 1
        if label_type is not None and ltype != label_type:
            continue
        m = FILE_RE.search(filename)
        if not m:
            continue
        start = first_offset(blob)
        key = (filename, round(start, 3))
        if key in seen:
            continue
        seen.add(key)
        t0 = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        times.append(t0 + timedelta(seconds=start))
    return times, dict(type_counts), len(rows)


def read_coverage(cov_dir: Path, year: int, month: int):
    """Return {date: hours} for the month, or None if no usable coverage CSV."""
    path = cov_dir / f"{year}-{month:02d}_coverage.csv"
    if not path.exists():
        return None
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    cols = {c.strip().lower(): c for c in rows[0].keys() if c}
    dcol, hcol = cols.get("date"), cols.get("hours")
    if dcol is None or hcol is None:
        print(f"  WARNING: {path.name} has columns {list(rows[0].keys())}; "
              f"need 'date' and 'hours'. Hours unavailable for {year}-{month:02d}.")
        return None
    out: dict[date, float] = {}
    for r in rows:
        try:
            d = datetime.strptime(r[dcol].strip(), "%Y-%m-%d").date()
            out[d] = float(r[hcol])
        except (ValueError, AttributeError, TypeError):
            continue  # TOTAL / blank / summary rows
    return out


def build_daily(args):
    """Read the DBs. Returns {date: [count, hours_or_None, source_db_name]}."""
    db_root = Path(args.db_root)
    cov_dir = Path(args.coverage_dir).expanduser()
    dbs = find_month_dbs(db_root)
    daily: dict[date, list] = {}
    since, through = args.since, args.through

    print(f"{'month':8s} {'database':44s} {'rows(all)':>9s} {'counted':>8s} {'in-month':>9s}  rows by label_type")
    for (y, m), db_dir in dbs.items():
        if not (since <= (y, m) <= through):
            continue
        times, types, raw = read_orca_clips(db_dir, args.label, args.label_type)
        counts: dict[date, int] = defaultdict(int)
        for t in times:
            counts[t.date()] += 1
        ndays = calendar.monthrange(y, m)[1]
        in_month = 0
        cov = read_coverage(cov_dir, y, m)
        if cov is None:
            print(f"  note: no usable coverage CSV for {y}-{m:02d} (hours blank)")
        for day in range(1, ndays + 1):
            d = date(y, m, day)
            hours = cov.get(d, 0.0) if cov is not None else None
            daily[d] = [counts.get(d, 0), hours, db_dir.name]
            in_month += counts.get(d, 0)
        flag = ""
        if len(types) > 1:
            flag = ("   <-- more than one label_type; counting only "
                    f"{args.label_type}" if args.label_type is not None
                    else "   <-- more than one label_type; counting all")
        print(f"{y}-{m:02d}  {db_dir.name:44s} {raw:9d} {len(times):8d} {in_month:9d}  {types}{flag}")
    return daily


# ----------------------------------------------------------------------------
# Intermediate CSV
# ----------------------------------------------------------------------------

def write_counts_csv(daily, path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "orca_clips", "hours_recorded", "source_db"])
        for d in sorted(daily):
            count, hours, src = daily[d]
            w.writerow([d.isoformat(), count, "" if hours is None else f"{hours:.4f}", src])


def read_counts_csv(path: Path, since, through):
    daily: dict[date, list] = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            d = datetime.strptime(r["date"], "%Y-%m-%d").date()
            if not (since <= (d.year, d.month) <= through):
                continue
            hours = float(r["hours_recorded"]) if r["hours_recorded"] != "" else None
            daily[d] = [int(r["orca_clips"]), hours, r["source_db"]]
    return daily


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------

def read_review_status(path: Path):
    """{(year, month): reviewed?} from the campaign statistics CSV, or None.

    A month counts as reviewed when its pass1_orca cell is filled in (0 counts).
    """
    if not path.exists():
        return None
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or "month" not in rows[0] or "pass1_orca" not in rows[0]:
        return None
    status: dict[tuple[int, int], bool] = {}
    for r in rows:
        m = re.fullmatch(r"(\d{4})-(\d{2})", (r["month"] or "").strip())
        if m:
            status[(int(m.group(1)), int(m.group(2)))] = (r["pass1_orca"] or "").strip() != ""
    return status


def month_state(daily, status, y: int, m: int) -> str:
    """'unknown' (no status info), 'reviewed', or 'not_reviewed'.

    A month with confirmed clips is always treated as reviewed (a warning is
    printed in main() if the stats CSV does not agree).
    """
    if status is None:
        return "unknown"
    if status.get((y, m), False) or month_total(daily, y, m) > 0:
        return "reviewed"
    return "not_reviewed"


def day_value(rec, normalize: bool, min_hours: float) -> float:
    count, hours, _ = rec
    if not normalize:
        return float(count)
    if hours is None or hours < min_hours:
        return NAN
    return count / hours


def month_total(daily, y: int, m: int) -> int:
    return sum(daily[date(y, m, d)][0]
               for d in range(1, calendar.monthrange(y, m)[1] + 1)
               if date(y, m, d) in daily)


def plot_set1(daily, years, out_dir: Path, normalize: bool, min_hours: float, dpi: int,
              status=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    months_present = {(d.year, d.month) for d in daily}
    sub = out_dir / "set1_by_year"
    sub.mkdir(parents=True, exist_ok=True)
    ylabel = "orca clips per recorded hour" if normalize else "confirmed orca clips"
    written = 0

    for y in years:
        totals = {m: month_total(daily, y, m) for m in range(1, 13) if (y, m) in months_present}
        if sum(totals.values()) == 0:
            print(f"  Set 1: {y}: no confirmed orca in included months, skipped")
            continue
        fig, axes = plt.subplots(3, 4, figsize=(16, 9.5))
        any_unreviewed = False
        for m in range(1, 13):
            ax = axes[(m - 1) // 4][(m - 1) % 4]
            ndays = calendar.monthrange(y, m)[1]
            xs = list(range(1, ndays + 1))
            if (y, m) not in months_present:
                ax.set_title(f"{calendar.month_abbr[m]} {y}  (no data)", fontsize=10)
                ax.text(0.5, 0.5, "not analyzed", ha="center", va="center",
                        transform=ax.transAxes, color="gray")
                ax.set_xlim(0.5, ndays + 0.5)
                ax.set_yticks([])
                continue
            recs = [daily[date(y, m, d)] for d in xs]
            vals = [day_value(r, normalize, min_hours) for r in recs]
            for d, r in zip(xs, recs):
                if r[1] is not None and r[1] < min_hours:
                    ax.axvspan(d - 0.5, d + 0.5, color="lightgray", alpha=0.6, lw=0)
            ax.bar(xs, [0.0 if math.isnan(v) else v for v in vals],
                   width=0.8, color="#1f4e79")
            ax.set_xlim(0.5, ndays + 0.5)
            ax.set_xticks([t for t in (1, 5, 10, 15, 20, 25, 30) if t <= ndays])
            ax.set_title(f"{calendar.month_abbr[m]} {y}  (n={totals[m]})", fontsize=10)
            if totals[m] == 0:
                ax.set_ylim(0, 1)
                if month_state(daily, status, y, m) == "not_reviewed":
                    any_unreviewed = True
                    ax.set_facecolor("#e6e6e6")
                    ax.set_yticks([])
                    ax.set_title(f"{calendar.month_abbr[m]} {y}  (not reviewed)", fontsize=10)
                    ax.text(0.5, 0.5, "not yet reviewed", ha="center", va="center",
                            transform=ax.transAxes, color="dimgray", style="italic")
                else:
                    ax.text(0.5, 0.5, "no confirmed orca", ha="center", va="center",
                            transform=ax.transAxes, color="gray")
            if m % 4 == 1:
                ax.set_ylabel(ylabel, fontsize=9)
            if m > 8:
                ax.set_xlabel("day of month (UTC)", fontsize=9)
        what = "per recorded hour" if normalize else "per day"
        fig.suptitle(f"Confirmed orca clips {what} - {y}", fontsize=14)
        foot = f"Gray = fewer than {min_hours:g} h recorded that day. One clip = one 5-second window."
        if any_unreviewed:
            foot += " Gray panel = month not yet reviewed."
        fig.text(0.5, 0.005, foot, ha="center", fontsize=8, color="dimgray")
        fig.tight_layout(rect=(0, 0.02, 1, 0.96))
        out = sub / f"orca_daily_{y}.png"
        fig.savefig(out, dpi=dpi)
        plt.close(fig)
        written += 1
        print(f"  wrote {out}")
    return written


def year_colors(daily):
    """One fixed color per year, shared by every Set 2 figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    years = sorted({d.year for d in daily})
    cmap = plt.get_cmap("tab20" if len(years) > 10 else "tab10")
    return {y: cmap(i % cmap.N) for i, y in enumerate(years)}


def zero_value(rec, normalize: bool, min_hours: float) -> float:
    """Value for a year with no confirmed orca: 0, except days known to be (nearly) unrecorded."""
    hours = rec[1]
    if normalize and hours is not None and hours < min_hours:
        return NAN
    return 0.0


def plot_set2(daily, months, years, color_of, out_dir: Path, normalize: bool,
              min_hours: float, dpi: int, status=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = out_dir / "set2_by_month"
    sub.mkdir(parents=True, exist_ok=True)
    ylabel = "orca clips per recorded hour" if normalize else "confirmed orca clips per day"
    months_present = {(d.year, d.month) for d in daily}
    written = 0

    for m in months:
        ys = sorted(y for (y, mm) in months_present if mm == m and y in years)
        if not ys:
            print(f"  Set 2: {calendar.month_name[m]}: no analyzed years, skipped")
            continue

        fig, ax = plt.subplots(figsize=(12, 5.5))
        handles = {}
        ymax = 0.0
        # Years with no confirmed orca are drawn first (flat on zero) so lines with data sit on top.
        for y in sorted(ys, key=lambda yy: month_total(daily, yy, m) > 0):
            n = month_total(daily, y, m)
            xs = list(range(1, calendar.monthrange(y, m)[1] + 1))
            recs = [daily[date(y, m, d)] for d in xs]
            if n == 0 and month_state(daily, status, y, m) == "not_reviewed":
                # No line: we cannot claim zero orca for a month nobody has reviewed.
                (line,) = ax.plot([], [], ls=":", lw=1.6, color=color_of[y],
                                  label=f"{y} (not reviewed)")
                handles[y] = line
                continue
            if n == 0:
                vals = [zero_value(r, normalize, min_hours) for r in recs]
                lw, alpha, z = 1.2, 0.8, 2
            else:
                vals = [day_value(r, normalize, min_hours) for r in recs]
                lw, alpha, z = 1.6, 1.0, 3
            label = f"{y} (n={n})"
            if n > 0 and all(math.isnan(v) for v in vals):
                label = f"{y} (n={n}, no hours data)"
            (line,) = ax.plot(xs, vals, marker="o", ms=3, lw=lw, alpha=alpha, zorder=z,
                              color=color_of[y], label=label)
            handles[y] = line
            finite = [v for v in vals if not math.isnan(v)]
            if finite:
                ymax = max(ymax, max(finite))

        ax.set_xlim(0.5, 31.5)
        ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
        ax.set_xlabel("day of month (UTC)")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, ymax * 1.05 if ymax > 0 else 1)
        ax.grid(alpha=0.25)
        what = "per recorded hour" if normalize else "per day"
        ax.set_title(f"{calendar.month_name[m]} - confirmed orca clips {what}, all years")
        ax.legend(handles=[handles[y] for y in sorted(handles)],
                  title="Year (n = total clips)", loc="center left",
                  bbox_to_anchor=(1.01, 0.5), fontsize=9, frameon=False)
        if normalize:
            fig.text(0.5, -0.01, f"Days with < {min_hours:g} h recorded left blank.",
                     ha="center", fontsize=8, color="dimgray")
        out = sub / f"orca_daily_{m:02d}_{calendar.month_name[m]}.png"
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        written += 1
        print(f"  wrote {out}")
    return written


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description="Daily confirmed-orca histograms from perch-hoplite DBs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Examples", 1)[-1] if __doc__ else "")
    p.add_argument("--db-root", default="/mnt/PAM_Analysis/perch-hoplite/db",
                   help="Directory holding the per-month hoplite DB directories.")
    p.add_argument("--coverage-dir", default="~/perch-hoplite/results/coverage",
                   help="Directory with YYYY-MM_coverage.csv files.")
    p.add_argument("--out-dir", default="./orca_daily_plots", help="Output directory.")
    p.add_argument("--label", default="orca_call", help="Annotation label to count.")
    p.add_argument("--label-type", type=int, default=1,
                   help="Only count this label_type value (default 1 = positive).")
    p.add_argument("--all-label-types", action="store_true",
                   help="Count every label_type instead of filtering.")
    p.add_argument("--since", type=parse_ym, default=(2015, 1), metavar="YYYY-MM",
                   help="First month to include (default 2015-01).")
    p.add_argument("--through", type=parse_ym, default=(2023, 12), metavar="YYYY-MM",
                   help="Last month to include (default 2023-12; later months are "
                        "not fully reviewed yet).")
    p.add_argument("--years", type=int, nargs="+", help="Only these years.")
    p.add_argument("--months", type=int, nargs="+", choices=range(1, 13), metavar="M",
                   help="Only these calendar months for Set 2 (1-12).")
    p.add_argument("--normalize-set1", action="store_true",
                   help="Set 1 as clips per recorded hour instead of raw counts.")
    p.add_argument("--raw-set2", action="store_true",
                   help="Set 2 as raw daily counts instead of clips per recorded hour.")
    p.add_argument("--min-hours", type=float, default=1.0,
                   help="Days with fewer recorded hours are shaded (Set 1) or left "
                        "blank in normalized plots (default 1.0).")
    p.add_argument("--stats-csv", default="~/perch-hoplite-orca_science/campaign_pass_statistics.csv",
                   help="Campaign statistics CSV; a month counts as reviewed when its "
                        "pass1_orca cell is filled in. Used to label unreviewed months "
                        "'not reviewed'. If missing, no review status is shown.")
    p.add_argument("--reuse-counts", action="store_true",
                   help="Skip the DBs; re-plot from <out-dir>/orca_daily_counts.csv.")
    p.add_argument("--dpi", type=int, default=150)
    args = p.parse_args()

    if args.all_label_types:
        args.label_type = None

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    counts_csv = out_dir / "orca_daily_counts.csv"

    if args.reuse_counts:
        if not counts_csv.exists():
            print(f"ERROR: {counts_csv} not found; run once without --reuse-counts.")
            return 1
        daily = read_counts_csv(counts_csv, args.since, args.through)
        print(f"Loaded {len(daily)} days from {counts_csv}")
    else:
        if not Path(args.db_root).is_dir():
            print(f"ERROR: {args.db_root} is not a directory.")
            return 1
        daily = build_daily(args)
        write_counts_csv(daily, counts_csv)
        print(f"\nWrote {counts_csv} ({len(daily)} days)")

    if not daily:
        print("No months found in the requested range.")
        return 1

    all_years = sorted({d.year for d in daily})
    years = [y for y in all_years if not args.years or y in args.years]
    months = sorted(args.months) if args.months else list(range(1, 13))

    status = read_review_status(Path(args.stats_csv).expanduser())
    if status is None:
        print(f"\nNOTE: {args.stats_csv} not found or lacks 'month'/'pass1_orca'; "
              "review status will not be shown.")
    else:
        months_in = sorted({(d.year, d.month) for d in daily})
        unrev = [f"{y}-{m:02d}" for (y, m) in months_in
                 if month_state(daily, status, y, m) == "not_reviewed"]
        unlisted = [f"{y}-{m:02d}" for (y, m) in months_in
                    if not status.get((y, m), False) and month_total(daily, y, m) > 0]
        print(f"\nReview status from {args.stats_csv}:")
        print(f"  not reviewed (no pass-1 recorded, no confirmed clips): {unrev or 'none'}")
        if unlisted:
            print(f"  WARNING: confirmed clips but no pass-1 entry in the stats CSV: {unlisted}")

    print("\nSet 1 (one figure per year):")
    n1 = plot_set1(daily, years, out_dir, args.normalize_set1, args.min_hours, args.dpi,
                   status=status)
    print("\nSet 2 (one figure per calendar month):")
    n2 = plot_set2(daily, months, set(years), year_colors(daily), out_dir,
                   not args.raw_set2, args.min_hours, args.dpi, status=status)
    print(f"\nDone: {n1} Set 1 figure(s), {n2} Set 2 figure(s) in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
