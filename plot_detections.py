#!/usr/bin/env python3
"""plot_detections.py
Plot orca detection timelines from inference CSV files.
Supports v1_clean, v2_clean, and v3_clean (multi-class) outputs.

Usage:
    python3 plot_detections.py \
        --v1 /mnt/PAM_Analysis/duane_scratch/perch_hoplite/results/MARS_20180413_orca_v1_clean_detections.csv \
        --v2 /mnt/PAM_Analysis/duane_scratch/perch_hoplite/results/MARS_20180413_orca_v2_clean_detections.csv \
        --v3 /mnt/PAM_Analysis/duane_scratch/perch_hoplite/results/MARS_20180413_orca_v3_clean_detections.csv \
        --output-dir /mnt/PAM_Analysis/duane_scratch/perch_hoplite/results
"""

import argparse
import os
import re
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta


def parse_utc_hour(filename, offset_s):
    m = re.search(r'MARS_(\d{8})_(\d{6})', filename)
    if not m:
        return None
    base_dt = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    dt = base_dt + timedelta(seconds=float(offset_s))
    return dt.hour + dt.minute / 60.0


def load_detections(csv_path, label_filter=None):
    """Load CSV, optionally filtering to a specific label."""
    df = pd.read_csv(csv_path)
    if label_filter and "label" in df.columns:
        df = df[df["label"] == label_filter].copy()
    df["utc_hour"] = df.apply(
        lambda r: parse_utc_hour(r["filename"], r["window_start"]), axis=1)
    return df


def _add_orca_shading(ax):
    for start, end in [(6.8, 9.85), (12.9, 13.1), (14.3, 18.5), (20.8, 21.0)]:
        ax.axvspan(start, end, alpha=0.08, color="#22c55e", zorder=0)


def _xticks(ax):
    ax.set_xticks(range(0, 25, 2))
    ax.set_xticklabels(
        [f"{h:02d}:00\n(PDT {(h-7)%24:02d}:00)" for h in range(0, 25, 2)],
        fontsize=7, color="#94a3b8")


def plot_all(v1_path, v2_path, v3_path, output_dir):
    # Load all three
    df1 = load_detections(v1_path)  # v1: orca only (no label column)
    df2 = load_detections(v2_path)  # v2: orca only
    df3_orca = load_detections(v3_path, label_filter="orca_call")
    df3_dolp = load_detections(v3_path, label_filter="dolphin_call")

    colors = {
        "v1": "#00e5ff",
        "v2": "#f59e0b",
        "v3_orca": "#a3e635",
        "v3_dolp": "#f472b6",
    }

    # ── Figure 1: Hourly detection counts, all models ────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 8),
                             gridspec_kw={"hspace": 0.4})
    fig.patch.set_facecolor("#0f172a")
    fig.suptitle("MARS Hydrophone — April 13 2018 — Orca Detection Comparison",
                 color="#e2e8f0", fontsize=12, y=0.98)

    bins = np.arange(0, 25, 1)

    # Top panel: v1 vs v2 vs v3 orca
    ax = axes[0]
    ax.set_facecolor("#1e293b")
    ax.hist(df1["utc_hour"], bins=bins, alpha=0.7,
            color=colors["v1"], label=f"v1_clean orca (n={len(df1)}, ROC-AUC 0.982)")
    ax.hist(df2["utc_hour"], bins=bins, alpha=0.5,
            color=colors["v2"], label=f"v2_clean orca (n={len(df2)}, ROC-AUC 0.919)")
    ax.hist(df3_orca["utc_hour"], bins=bins, alpha=0.5,
            color=colors["v3_orca"], label=f"v3_clean orca (n={len(df3_orca)}, ROC-AUC 0.990)")
    ax.set_xlim(0, 24)
    ax.set_ylabel("Detections/hr", color="#94a3b8", fontsize=9)
    ax.set_title("Orca detections — all three models", color="#e2e8f0", fontsize=10)
    ax.tick_params(colors="#94a3b8", labelsize=8)
    ax.legend(fontsize=8, facecolor="#1e293b", labelcolor="#e2e8f0")
    for spine in ax.spines.values(): spine.set_edgecolor("#334155")
    _add_orca_shading(ax)
    _xticks(ax)

    # Bottom panel: v3 orca vs v3 dolphin
    ax2 = axes[1]
    ax2.set_facecolor("#1e293b")
    ax2.hist(df3_orca["utc_hour"], bins=bins, alpha=0.7,
             color=colors["v3_orca"], label=f"v3_clean orca_call (n={len(df3_orca)})")
    ax2.hist(df3_dolp["utc_hour"], bins=bins, alpha=0.7,
             color=colors["v3_dolp"], label=f"v3_clean dolphin_call (n={len(df3_dolp)})")
    ax2.set_xlim(0, 24)
    ax2.set_ylabel("Detections/hr", color="#94a3b8", fontsize=9)
    ax2.set_title("v3_clean: orca vs dolphin detections", color="#e2e8f0", fontsize=10)
    ax2.tick_params(colors="#94a3b8", labelsize=8)
    ax2.legend(fontsize=8, facecolor="#1e293b", labelcolor="#e2e8f0")
    for spine in ax2.spines.values(): spine.set_edgecolor("#334155")
    _add_orca_shading(ax2)
    _xticks(ax2)

    out1 = os.path.join(output_dir, "MARS_20180413_orca_detection_comparison.png")
    fig.savefig(out1, dpi=150, bbox_inches="tight", facecolor="#0f172a")
    plt.close(fig)
    print(f"Saved: {out1}")

    # ── Figure 2: Logit score scatter, v3_clean orca + dolphin ───────────
    df3_all = pd.read_csv(v3_path)
    df3_all["utc_hour"] = df3_all.apply(
        lambda r: parse_utc_hour(r["filename"], r["window_start"]), axis=1)

    fig2, ax3 = plt.subplots(figsize=(14, 5))
    fig2.patch.set_facecolor("#0f172a")
    ax3.set_facecolor("#1e293b")

    for lbl, col in [("orca_call", colors["v3_orca"]),
                     ("dolphin_call", colors["v3_dolp"]),
                     ("other", "#94a3b8")]:
        sub = df3_all[df3_all["label"] == lbl]
        if len(sub):
            ax3.scatter(sub["utc_hour"], sub["logits"],
                        c=col, alpha=0.6, s=14, label=f"{lbl} (n={len(sub)})")

    ax3.axhline(0, color="#ef4444", linewidth=0.8, linestyle="--", label="threshold=0")
    ax3.set_xlim(0, 24)
    ax3.set_xlabel("UTC Hour", color="#94a3b8", fontsize=9)
    ax3.set_ylabel("Logit score", color="#94a3b8", fontsize=9)
    ax3.set_title("v3_clean — all detections by class and score",
                  color="#e2e8f0", fontsize=11)
    ax3.tick_params(colors="#94a3b8", labelsize=8)
    ax3.legend(fontsize=9, facecolor="#1e293b", labelcolor="#e2e8f0")
    for spine in ax3.spines.values(): spine.set_edgecolor("#334155")
    _add_orca_shading(ax3)
    _xticks(ax3)

    out2 = os.path.join(output_dir, "MARS_20180413_v3_clean_scatter.png")
    fig2.savefig(out2, dpi=150, bbox_inches="tight", facecolor="#0f172a")
    plt.close(fig2)
    print(f"Saved: {out2}")

    # Summary
    print(f"\nSummary:")
    print(f"  v1_clean: {len(df1)} orca detections")
    print(f"  v2_clean: {len(df2)} orca detections")
    print(f"  v3_clean: {len(df3_orca)} orca + {len(df3_dolp)} dolphin detections")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v1", required=True)
    ap.add_argument("--v2", required=True)
    ap.add_argument("--v3", required=True)
    ap.add_argument("--output-dir", default=".")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    plot_all(args.v1, args.v2, args.v3, args.output_dir)


if __name__ == "__main__":
    main()
