#!/usr/bin/env python3
"""tools/plot_labels_monthly.py — per-day histogram of CONFIRMED LABELS by class.

Companion to plot_monthly.py. That tool plots MODEL DETECTIONS from an inference CSV;
this one plots what was CONFIRMED BY EAR, read from a hoplite DB's annotations table, and
normalises by hours actually recorded using the month's coverage CSV.

Why both: at the operating threshold the detector finds only a fraction of real calls
(September 2015: 3 of 18 confirmed orca calls, ~17% recall), so detection counts and
confirmed-label counts are different quantities and should not be conflated.

Three panels:
  1. Confirmed labels per day, one row per class.
  2. Orca calls per HOUR OF EFFORT per day, with recording hours shaded behind.
     Days with zero coverage are drawn as gaps, not zeros -- "wasn't listening" is not
     the same claim as "listened, heard nothing".
  3. UTC hour-of-day scatter of every confirmed label, coloured by class.

Usage:
    python3 tools/plot_labels_monthly.py \\
        --db /mnt/PAM_Analysis/perch-hoplite/db/MARS_20150901_20150930_32kHz_norm/hoplite.sqlite \\
        --coverage results/coverage/2015-09_coverage.csv \\
        --output-dir figures --title "September 2015 — MARS — confirmed labels"
"""
import argparse
import calendar
import csv
import re
import sqlite3
import struct
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Shared with plot_monthly.py, plus ROV_noise (added Aug 30 2026).
LABEL_COLORS = {
    "orca_call":      "#16a34a",
    "humpback_song":  "#d97706",
    "fin_whale_call": "#2563eb",
    "dolphin_call":   "#9333ea",
    "ROV_noise":      "#dc2626",
    "ship_noise":     "#0891b2",
    "other":          "#ea580c",
    "negative":       "#6b7280",
}
DEFAULT_COLOR = "#94a3b8"
FN = re.compile(r'MARS_(\d{8})_(\d{6})')


def load_labels(db):
    con = sqlite3.connect(db)
    rows = []
    for fn, blob, label in con.execute("""
            SELECT r.filename, a.offsets, a.label
            FROM annotations a JOIN recordings r ON r.id = a.recording_id"""):
        m = FN.search(fn)
        if not m:
            continue
        try:
            start, _ = struct.unpack('<2d', blob)
        except Exception:
            start = 0.0
        d, t = m.groups()
        rows.append((datetime.strptime(d + t, "%Y%m%d%H%M%S") + timedelta(seconds=start),
                     label, int(d[6:8])))
    con.close()
    return rows


def load_coverage(path):
    hours = {}
    if not path:
        return hours
    try:
        for r in csv.DictReader(open(path)):
            hours[int(r['date'][8:10])] = float(r['hours'])
    except Exception as e:
        print(f"(coverage not read: {e})", file=sys.stderr)
    return hours


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--coverage", default=None)
    ap.add_argument("--output-dir", default=".")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    rows = load_labels(args.db)
    if not rows:
        sys.exit("no annotations in this DB")
    hours = load_coverage(args.coverage)

    year, month = rows[0][0].year, rows[0][0].month
    ndays = calendar.monthrange(year, month)[1]
    days = list(range(1, ndays + 1))
    title = args.title or f"{calendar.month_name[month]} {year} — MARS — confirmed labels"

    classes = sorted({r[1] for r in rows},
                     key=lambda c: (c != "orca_call", c))   # orca first
    per_day = defaultdict(lambda: defaultdict(int))
    for _dt, label, dom in rows:
        per_day[label][dom] += 1

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"labels_{year}-{month:02d}"

    # ── Figure 1: per-day counts, one row per class ───────────────────────
    fig, axes = plt.subplots(len(classes), 1, figsize=(14, 2.6 * len(classes)),
                             gridspec_kw={"hspace": 0.55})
    if len(classes) == 1:
        axes = [axes]
    fig.patch.set_facecolor("#0f172a")
    fig.suptitle(f"{title}\nConfirmed labels per day", color="#e2e8f0", fontsize=12, y=0.995)
    for ax, lbl in zip(axes, classes):
        ax.set_facecolor("#1e293b")
        counts = [per_day[lbl].get(d, 0) for d in days]
        present = [(d, c) for d, c in zip(days, counts) if c > 0]
        if present:
            ax.bar([d for d, _ in present], [c for _, c in present],
                   color=LABEL_COLORS.get(lbl, DEFAULT_COLOR), alpha=0.9, width=0.7)
        ax.set_xlim(0.5, ndays + 0.5)
        ax.set_xticks(days)
        ax.set_title(f"{lbl}   (total: {sum(counts)})", color="#e2e8f0", fontsize=10)
        ax.set_ylabel("labels", color="#94a3b8", fontsize=8)
        ax.tick_params(colors="#94a3b8", labelsize=7)
        for sp in ax.spines.values():
            sp.set_color("#334155")
        ax.grid(axis="y", color="#334155", lw=0.5, alpha=0.5)
    axes[-1].set_xlabel("day of month (UTC)", color="#94a3b8", fontsize=9)
    f1 = out / f"{stem}_per_day.png"
    fig.savefig(f1, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    # ── Figure 2: orca per hour of effort ─────────────────────────────────
    if hours:
        fig, ax = plt.subplots(figsize=(14, 4.2))
        fig.patch.set_facecolor("#0f172a")
        ax.set_facecolor("#1e293b")
        ax2 = ax.twinx()
        ax2.bar(days, [hours.get(d, 0.0) for d in days], color="#334155",
                width=0.85, zorder=1, label="hours recorded")
        ax2.set_ylabel("hours recorded", color="#64748b", fontsize=9)
        ax2.set_ylim(0, 26)
        ax2.tick_params(colors="#64748b", labelsize=8)
        rate_d, rate_v = [], []
        for d in days:
            h = hours.get(d, 0.0)
            if h > 0:                       # zero coverage -> gap, never a zero
                rate_d.append(d)
                rate_v.append(per_day.get("orca_call", {}).get(d, 0) / h)
        ax.plot(rate_d, rate_v, "o-", color="#16a34a", lw=2, ms=6, zorder=3,
                label="orca calls / hour of effort")
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)
        ax.set_xlim(0.5, ndays + 0.5)
        ax.set_xticks(days)
        ax.set_ylabel("orca calls / hour", color="#16a34a", fontsize=9)
        ax.set_xlabel("day of month (UTC)", color="#94a3b8", fontsize=9)
        ax.tick_params(colors="#94a3b8", labelsize=8)
        ax.set_title(f"{title}\nOrca calls per hour of recording effort "
                     f"(grey = hours recorded; gaps = no data)",
                     color="#e2e8f0", fontsize=11)
        for sp in ax.spines.values():
            sp.set_color("#334155")
        ax.grid(axis="y", color="#334155", lw=0.5, alpha=0.5)
        f2 = out / f"{stem}_orca_per_hour.png"
        fig.savefig(f2, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)
    else:
        f2 = None

    # ── Figure 3: UTC hour-of-day scatter ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 4.6))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")
    # Horizontal jitter: labels cluster in minutes, so without it 8 calls in one
    # 10-minute recording render as a single dot and the density is invisible.
    import random
    random.seed(0)
    for lbl in classes:
        pts = [(r[2], r[0].hour + r[0].minute / 60 + r[0].second / 3600)
               for r in rows if r[1] == lbl]
        if pts:
            ax.scatter([p[0] + random.uniform(-0.22, 0.22) for p in pts],
                       [p[1] for p in pts], s=55,
                       color=LABEL_COLORS.get(lbl, DEFAULT_COLOR),
                       edgecolor="#0f172a", lw=0.6, label=lbl, alpha=0.85, zorder=3)
    ax.set_xlim(0.5, ndays + 0.5)
    ax.set_ylim(-0.5, 24.5)
    ax.set_yticks(range(0, 25, 3))
    ax.set_xticks(days)
    ax.set_xlabel("day of month (UTC)", color="#94a3b8", fontsize=9)
    ax.set_ylabel("hour of day (UTC)", color="#94a3b8", fontsize=9)
    ax.set_title(f"{title}\nEvery confirmed label, by day and hour", color="#e2e8f0", fontsize=11)
    ax.tick_params(colors="#94a3b8", labelsize=8)
    ax.grid(color="#334155", lw=0.5, alpha=0.5)
    for sp in ax.spines.values():
        sp.set_color("#334155")
    leg = ax.legend(facecolor="#1e293b", edgecolor="#334155", fontsize=8, loc="upper left")
    for t in leg.get_texts():
        t.set_color("#e2e8f0")
    f3 = out / f"{stem}_by_hour.png"
    fig.savefig(f3, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    for f in (f1, f2, f3):
        if f:
            print(f"Wrote {f}")


if __name__ == "__main__":
    main()
