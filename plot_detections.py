#!/usr/bin/env python3
"""plot_detections.py
Plot orca detection timelines from inference CSV files.
Supports v1_clean through v4_clean outputs.

Usage:
    python3 plot_detections.py \
        --v1 .../MARS_20180413_orca_v1_clean_detections.csv \
        --v2 .../MARS_20180413_orca_v2_clean_detections.csv \
        --v3 .../MARS_20180413_orca_v3_clean_detections.csv \
        --v4 .../MARS_20180413_orca_v4_clean_detections.csv \
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
    df = pd.read_csv(csv_path)
    if label_filter and "label" in df.columns:
        df = df[df["label"] == label_filter].copy()
    df["utc_hour"] = df.apply(
        lambda r: parse_utc_hour(r["filename"], r["window_start"]), axis=1)
    return df


def _add_orca_shading(ax):
    for start, end in [(6.8, 9.85), (12.9, 13.1), (14.3, 18.5), (20.8, 21.0)]:
        ax.axvspan(start, end, alpha=0.08, color="#22c55e", zorder=0)


def _style(ax):
    ax.set_facecolor("#1e293b")
    ax.tick_params(colors="#94a3b8", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")
    ax.set_xticks(range(0, 25, 2))
    ax.set_xticklabels(
        [f"{h:02d}:00\n(PDT {(h-7)%24:02d}:00)" for h in range(0, 25, 2)],
        fontsize=7, color="#94a3b8")
    ax.set_xlim(0, 24)


def plot_all(v1_path, v2_path, v3_path, v4_path, output_dir):

    colors = {
        "v1":        "#00e5ff",
        "v2":        "#f59e0b",
        "v3_orca":   "#a3e635",
        "v4_orca":   "#fb7185",
        "dolphin":   "#f472b6",
        "other":     "#94a3b8",
    }

    bins = np.arange(0, 25, 1)

    # Load all models
    df1       = load_detections(v1_path)
    df2       = load_detections(v2_path)
    df3_orca  = load_detections(v3_path, label_filter="orca_call")
    df3_dolp  = load_detections(v3_path, label_filter="dolphin_call")
    df4_orca  = load_detections(v4_path, label_filter="orca_call")
    df4_dolp  = load_detections(v4_path, label_filter="dolphin_call")
    df4_other = load_detections(v4_path, label_filter="other")
    df4_all   = pd.read_csv(v4_path)
    df4_all["utc_hour"] = df4_all.apply(
        lambda r: parse_utc_hour(r["filename"], r["window_start"]), axis=1)

    # ── Figure 1: Orca detections — all four models ───────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 9),
                             gridspec_kw={"hspace": 0.45})
    fig.patch.set_facecolor("#0f172a")
    fig.suptitle("MARS Hydrophone — April 13 2018\nOrca & Dolphin Detections by Model Version",
                 color="#e2e8f0", fontsize=12, y=0.99)

    ax = axes[0]
    _style(ax)
    ax.hist(df1["utc_hour"], bins=bins, alpha=0.8,
            color=colors["v1"],
            label=f"v1_clean orca (n={len(df1)}, ROC-AUC 0.982, single-class)")
    ax.hist(df2["utc_hour"], bins=bins, alpha=0.6,
            color=colors["v2"],
            label=f"v2_clean orca (n={len(df2)}, ROC-AUC 0.919, single-class)")
    ax.hist(df3_orca["utc_hour"], bins=bins, alpha=0.6,
            color=colors["v3_orca"],
            label=f"v3_clean orca (n={len(df3_orca)}, ROC-AUC 0.990, multi-class)")
    ax.hist(df4_orca["utc_hour"], bins=bins, alpha=0.6,
            color=colors["v4_orca"],
            label=f"v4_clean orca (n={len(df4_orca)}, ROC-AUC 0.974, multi-class)")
    ax.set_ylabel("Detections/hr", color="#94a3b8", fontsize=9)
    ax.set_title("Orca detections — all four model versions", color="#e2e8f0", fontsize=10)
    ax.legend(fontsize=7.5, facecolor="#1e293b", labelcolor="#e2e8f0")
    _add_orca_shading(ax)

    # Bottom panel: v4 all classes
    ax2 = axes[1]
    _style(ax2)
    ax2.hist(df4_orca["utc_hour"], bins=bins, alpha=0.8,
             color=colors["v4_orca"],
             label=f"orca_call (n={len(df4_orca)})")
    ax2.hist(df4_dolp["utc_hour"], bins=bins, alpha=0.7,
             color=colors["dolphin"],
             label=f"dolphin_call (n={len(df4_dolp)})")
    ax2.hist(df4_other["utc_hour"], bins=bins, alpha=0.7,
             color=colors["other"],
             label=f"other/vessel (n={len(df4_other)})")
    ax2.set_ylabel("Detections/hr", color="#94a3b8", fontsize=9)
    ax2.set_title("v4_clean — all species/classes (orca + dolphin + other)",
                  color="#e2e8f0", fontsize=10)
    ax2.legend(fontsize=8, facecolor="#1e293b", labelcolor="#e2e8f0")
    _add_orca_shading(ax2)

    out1 = os.path.join(output_dir, "MARS_20180413_orca_detection_comparison.png")
    fig.savefig(out1, dpi=150, bbox_inches="tight", facecolor="#0f172a")
    plt.close(fig)
    print(f"Saved: {out1}")

    # ── Figure 2: v4_clean scatter — all classes ──────────────────────────
    fig2, ax3 = plt.subplots(figsize=(14, 5))
    fig2.patch.set_facecolor("#0f172a")
    _style(ax3)

    class_colors = {
        "orca_call":    colors["v4_orca"],
        "dolphin_call": colors["dolphin"],
        "other":        colors["other"],
    }
    for lbl, col in class_colors.items():
        sub = df4_all[df4_all["label"] == lbl]
        if len(sub):
            ax3.scatter(sub["utc_hour"], sub["logits"],
                        c=col, alpha=0.5, s=12,
                        label=f"{lbl} (n={len(sub)})")

    ax3.axhline(0, color="#ef4444", linewidth=0.8, linestyle="--", label="threshold=0")
    ax3.set_xlabel("UTC Hour", color="#94a3b8", fontsize=9)
    ax3.set_ylabel("Logit score", color="#94a3b8", fontsize=9)
    ax3.set_title("v4_clean — all detections by class and logit score",
                  color="#e2e8f0", fontsize=11)
    ax3.legend(fontsize=9, facecolor="#1e293b", labelcolor="#e2e8f0")
    _add_orca_shading(ax3)

    out2 = os.path.join(output_dir, "MARS_20180413_v4_clean_scatter.png")
    fig2.savefig(out2, dpi=150, bbox_inches="tight", facecolor="#0f172a")
    plt.close(fig2)
    print(f"Saved: {out2}")

    # ── Figure 3: dolphin v3 vs v4 comparison ─────────────────────────────
    fig3, ax4 = plt.subplots(figsize=(14, 4))
    fig3.patch.set_facecolor("#0f172a")
    _style(ax4)
    ax4.hist(df3_dolp["utc_hour"], bins=bins, alpha=0.7,
             color=colors["v3_orca"],
             label=f"v3_clean dolphin_call (n={len(df3_dolp)})")
    ax4.hist(df4_dolp["utc_hour"], bins=bins, alpha=0.6,
             color=colors["dolphin"],
             label=f"v4_clean dolphin_call (n={len(df4_dolp)})")
    ax4.set_ylabel("Detections/hr", color="#94a3b8", fontsize=9)
    ax4.set_title("Dolphin detections — v3_clean vs v4_clean",
                  color="#e2e8f0", fontsize=10)
    ax4.legend(fontsize=8, facecolor="#1e293b", labelcolor="#e2e8f0")
    _add_orca_shading(ax4)

    out3 = os.path.join(output_dir, "MARS_20180413_dolphin_v3_v4_comparison.png")
    fig3.savefig(out3, dpi=150, bbox_inches="tight", facecolor="#0f172a")
    plt.close(fig3)
    print(f"Saved: {out3}")

    print(f"\nSummary:")
    print(f"  v1_clean: {len(df1):4d} orca")
    print(f"  v2_clean: {len(df2):4d} orca")
    print(f"  v3_clean: {len(df3_orca):4d} orca  {len(df3_dolp):4d} dolphin")
    print(f"  v4_clean: {len(df4_orca):4d} orca  {len(df4_dolp):4d} dolphin  {len(df4_other):4d} other")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v1", required=True)
    ap.add_argument("--v2", required=True)
    ap.add_argument("--v3", required=True)
    ap.add_argument("--v4", required=True)
    ap.add_argument("--output-dir", default=".")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    plot_all(args.v1, args.v2, args.v3, args.v4, args.output_dir)


if __name__ == "__main__":
    main()
