#!/usr/bin/env python3
"""plot_detections.py
Plot orca detection timelines from inference CSV files.

Generates two figures:
  1. Overlaid timeline — v1_clean and v2_clean detections vs UTC time
  2. Side-by-side bar charts — detections per 10-minute recording file

Usage:
    python3 plot_detections.py \
        --v1 /mnt/PAM_Analysis/duane_scratch/perch_hoplite/results/MARS_20180413_orca_v1_clean_detections.csv \
        --v2 /mnt/PAM_Analysis/duane_scratch/perch_hoplite/results/MARS_20180413_orca_v2_clean_detections.csv \
        --output-dir /mnt/PAM_Analysis/duane_scratch/perch_hoplite/results
"""

import argparse
import os
import re
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import numpy as np


def parse_utc_time(filename, offset_s):
    """Extract UTC datetime from MARS filename + window offset."""
    # MARS_20180413_083913_resampled_32kHz.wav
    m = re.search(r'MARS_(\d{8})_(\d{6})', filename)
    if not m:
        return None
    date_str = m.group(1)   # 20180413
    time_str = m.group(2)   # 083913
    base_dt = datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")
    return base_dt + timedelta(seconds=float(offset_s))


def load_detections(csv_path, label):
    """Load CSV and return DataFrame with UTC datetime column."""
    df = pd.read_csv(csv_path)
    df["utc_dt"] = df.apply(
        lambda r: parse_utc_time(r["filename"], r["window_start"]), axis=1
    )
    df["utc_hour"] = df["utc_dt"].dt.hour + df["utc_dt"].dt.minute / 60.0
    df["label"] = label
    print(f"{label}: {len(df)} detections loaded from {os.path.basename(csv_path)}")
    return df


def plot_overlaid_timeline(df1, df2, output_path):
    """Plot detections as scatter/rug plot over 24-hour UTC timeline."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 8),
                             gridspec_kw={"height_ratios": [2, 2, 1.2],
                                         "hspace": 0.35})
    fig.patch.set_facecolor("#0f172a")

    colors = {"v1_clean": "#00e5ff", "v2_clean": "#f59e0b"}

    # ── Panel 1: v1_clean scatter ─────────────────────────────────────────
    ax1 = axes[0]
    ax1.set_facecolor("#1e293b")
    ax1.scatter(df1["utc_hour"], df1["logits"],
                c=colors["v1_clean"], alpha=0.7, s=18, label="v1_clean")
    ax1.axhline(0, color="#ef4444", linewidth=0.8, linestyle="--", label="threshold=0")
    ax1.set_xlim(0, 24)
    ax1.set_ylabel("Logit score", color="#94a3b8", fontsize=9)
    ax1.set_title("orca_v1_clean detections (ROC-AUC 0.982)", color="#e2e8f0", fontsize=10)
    ax1.tick_params(colors="#94a3b8", labelsize=8)
    ax1.legend(fontsize=8, facecolor="#1e293b", labelcolor="#e2e8f0")
    for spine in ax1.spines.values():
        spine.set_edgecolor("#334155")
    _add_orca_shading(ax1)

    # ── Panel 2: v2_clean scatter ─────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("#1e293b")
    ax2.scatter(df2["utc_hour"], df2["logits"],
                c=colors["v2_clean"], alpha=0.7, s=18, label="v2_clean")
    ax2.axhline(0, color="#ef4444", linewidth=0.8, linestyle="--", label="threshold=0")
    ax2.set_xlim(0, 24)
    ax2.set_ylabel("Logit score", color="#94a3b8", fontsize=9)
    ax2.set_title("orca_v2_clean detections (ROC-AUC 0.919)", color="#e2e8f0", fontsize=10)
    ax2.tick_params(colors="#94a3b8", labelsize=8)
    ax2.legend(fontsize=8, facecolor="#1e293b", labelcolor="#e2e8f0")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#334155")
    _add_orca_shading(ax2)

    # ── Panel 3: detection counts per hour (both models) ──────────────────
    ax3 = axes[2]
    ax3.set_facecolor("#1e293b")
    bins = np.arange(0, 25, 1)
    ax3.hist(df1["utc_hour"], bins=bins, alpha=0.7,
             color=colors["v1_clean"], label=f"v1_clean (n={len(df1)})")
    ax3.hist(df2["utc_hour"], bins=bins, alpha=0.5,
             color=colors["v2_clean"], label=f"v2_clean (n={len(df2)})")
    ax3.set_xlim(0, 24)
    ax3.set_xlabel("UTC Hour", color="#94a3b8", fontsize=9)
    ax3.set_ylabel("Detections/hr", color="#94a3b8", fontsize=9)
    ax3.set_title("Detection count per hour", color="#e2e8f0", fontsize=10)
    ax3.tick_params(colors="#94a3b8", labelsize=8)
    ax3.legend(fontsize=8, facecolor="#1e293b", labelcolor="#e2e8f0")
    for spine in ax3.spines.values():
        spine.set_edgecolor("#334155")
    _add_orca_shading(ax3)

    # X-axis labels — UTC hours with PDT annotation
    for ax in axes:
        ax.set_xticks(range(0, 25, 2))
        ax.set_xticklabels(
            [f"{h:02d}:00\n(PDT {(h-7)%24:02d}:00)" for h in range(0, 25, 2)],
            fontsize=7, color="#94a3b8"
        )

    fig.suptitle("MARS Hydrophone — April 13 2018 — Orca Detection Timeline",
                 color="#e2e8f0", fontsize=12, y=0.98)

    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#0f172a")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_per_file_bars(df1, df2, output_path):
    """Bar chart: detections per 10-minute recording file, both models."""
    # Extract file hour:minute for grouping
    def file_hhmm(filename):
        m = re.search(r'_(\d{6})_resampled', filename)
        if m:
            t = m.group(1)
            return int(t[:2]) + int(t[2:4]) / 60.0
        return None

    df1["file_hour"] = df1["filename"].apply(file_hhmm)
    df2["file_hour"] = df2["filename"].apply(file_hhmm)

    c1 = df1.groupby("file_hour").size()
    c2 = df2.groupby("file_hour").size()
    all_hours = sorted(set(c1.index) | set(c2.index))

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    x = np.array(all_hours)
    width = 0.04
    bars1 = ax.bar(x - width/2, [c1.get(h, 0) for h in all_hours],
                   width=width, color="#00e5ff", alpha=0.8,
                   label=f"v1_clean (n={len(df1)})")
    bars2 = ax.bar(x + width/2, [c2.get(h, 0) for h in all_hours],
                   width=width, color="#f59e0b", alpha=0.8,
                   label=f"v2_clean (n={len(df2)})")

    ax.set_xlabel("UTC Hour", color="#94a3b8", fontsize=9)
    ax.set_ylabel("Detections per file", color="#94a3b8", fontsize=9)
    ax.set_title("MARS April 13 2018 — Detections per 10-min file",
                 color="#e2e8f0", fontsize=11)
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 1))
    ax.set_xticklabels(
        [f"{h:02d}" for h in range(0, 25)],
        fontsize=7, color="#94a3b8"
    )
    ax.tick_params(colors="#94a3b8", labelsize=8)
    ax.legend(fontsize=9, facecolor="#1e293b", labelcolor="#e2e8f0")
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")
    _add_orca_shading(ax)

    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#0f172a")
    plt.close(fig)
    print(f"Saved: {output_path}")


def _add_orca_shading(ax):
    """Shade known orca active hours (UTC) in light green."""
    active_windows = [
        (6.8, 9.85),    # 06:49 – 09:49 UTC morning event
        (12.9, 13.1),   # 12:59 UTC midday
        (14.3, 18.5),   # 14:19 – 18:29 UTC afternoon event
        (20.8, 21.0),   # 20:49 UTC evening
    ]
    ymin, ymax = ax.get_ylim()
    for start, end in active_windows:
        ax.axvspan(start, end, alpha=0.08, color="#22c55e", zorder=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v1", required=True, help="v1_clean detections CSV")
    ap.add_argument("--v2", required=True, help="v2_clean detections CSV")
    ap.add_argument("--output-dir", default=".",
                    help="Directory to write PNG files")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df1 = load_detections(args.v1, "v1_clean")
    df2 = load_detections(args.v2, "v2_clean")

    timeline_path = os.path.join(args.output_dir,
                                 "MARS_20180413_orca_detection_timeline.png")
    perfile_path  = os.path.join(args.output_dir,
                                 "MARS_20180413_orca_detection_per_file.png")

    plot_overlaid_timeline(df1, df2, timeline_path)
    plot_per_file_bars(df1, df2, perfile_path)

    print("\nDone. Output files:")
    print(f"  {timeline_path}")
    print(f"  {perfile_path}")


if __name__ == "__main__":
    main()
