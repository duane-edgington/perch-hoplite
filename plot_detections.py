#!/usr/bin/env python3
"""plot_detections.py
Plot orca/dolphin/other detection timelines from inference CSV files.
Supports v1_clean through v4_clean outputs, and multi-day comparisons.

Usage:
    python3 plot_detections.py \
        --v1 .../MARS_20180413_orca_v1_clean_detections.csv \
        --v2 .../MARS_20180413_orca_v2_clean_detections.csv \
        --v3 .../MARS_20180413_orca_v3_clean_detections.csv \
        --v4 .../MARS_20180413_orca_v4_clean_detections.csv \
        --apr01 .../MARS_20180401_orca_v4_clean_detections.csv \
        --apr20 .../MARS_20180420_v4_clean_detections.csv \
        --apr30 .../MARS_20180430_orca_v4_clean_detections.csv \
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
    elif label_filter and "label" not in df.columns:
        if label_filter != "orca_call":
            return df.iloc[0:0].copy()
    else:
        df = df.copy()
    # Use list comprehension — more robust than df.apply for mixed None/float
    df["utc_hour"] = [
        parse_utc_hour(str(fname), float(start))
        for fname, start in zip(df["filename"], df["window_start"])
    ]
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


colors = {
    "v1":        "#00e5ff",
    "v2":        "#f59e0b",
    "v3_orca":   "#a3e635",
    "v4_orca":   "#fb7185",
    "dolphin":   "#f472b6",
    "other":     "#94a3b8",
    "apr01":     "#38bdf8",
    "apr13":     "#fb7185",
    "apr20":     "#a3e635",
    "apr30":     "#f59e0b",
}


def plot_model_comparison(v1, v2, v3, v4, output_dir):
    """Figure 1: orca detections across all four model versions + v4 species breakdown."""
    bins = np.arange(0, 25, 1)
    df1      = load_detections(v1)
    df2      = load_detections(v2)
    df3_orca = load_detections(v3, "orca_call")
    df4_orca = load_detections(v4, "orca_call")
    df4_dolp = load_detections(v4, "dolphin_call")
    df4_othr = load_detections(v4, "other")

    fig, axes = plt.subplots(2, 1, figsize=(14, 9),
                             gridspec_kw={"hspace": 0.45})
    fig.patch.set_facecolor("#0f172a")
    fig.suptitle("MARS Hydrophone — April 13 2018\nOrca Classifier Evolution (v1→v4)",
                 color="#e2e8f0", fontsize=12, y=0.99)

    ax = axes[0]
    _style(ax)
    ax.hist(df1["utc_hour"], bins=bins, alpha=0.8, color=colors["v1"],
            label=f"v1_clean orca (n={len(df1)}, ROC-AUC 0.982, single-class)")
    ax.hist(df2["utc_hour"], bins=bins, alpha=0.6, color=colors["v2"],
            label=f"v2_clean orca (n={len(df2)}, ROC-AUC 0.919, single-class)")
    ax.hist(df3_orca["utc_hour"], bins=bins, alpha=0.6, color=colors["v3_orca"],
            label=f"v3_clean orca (n={len(df3_orca)}, ROC-AUC 0.990, multi-class)")
    ax.hist(df4_orca["utc_hour"], bins=bins, alpha=0.6, color=colors["v4_orca"],
            label=f"v4_clean orca (n={len(df4_orca)}, ROC-AUC 0.974, multi-class)")
    ax.set_ylabel("Detections/hr", color="#94a3b8", fontsize=9)
    ax.set_title("Orca detections — all four model versions", color="#e2e8f0", fontsize=10)
    ax.legend(fontsize=7.5, facecolor="#1e293b", labelcolor="#e2e8f0")
    for spine in ax.spines.values(): spine.set_edgecolor("#334155")
    _add_orca_shading(ax)

    ax2 = axes[1]
    _style(ax2)
    ax2.hist(df4_orca["utc_hour"], bins=bins, alpha=0.8, color=colors["v4_orca"],
             label=f"orca_call (n={len(df4_orca)})")
    ax2.hist(df4_dolp["utc_hour"], bins=bins, alpha=0.7, color=colors["dolphin"],
             label=f"dolphin_call (n={len(df4_dolp)})")
    ax2.hist(df4_othr["utc_hour"], bins=bins, alpha=0.7, color=colors["other"],
             label=f"other/vessel (n={len(df4_othr)})")
    ax2.set_ylabel("Detections/hr", color="#94a3b8", fontsize=9)
    ax2.set_title("v4_clean — all species/classes (April 13 2018)",
                  color="#e2e8f0", fontsize=10)
    ax2.legend(fontsize=8, facecolor="#1e293b", labelcolor="#e2e8f0")
    for spine in ax2.spines.values(): spine.set_edgecolor("#334155")
    _add_orca_shading(ax2)

    out = os.path.join(output_dir, "MARS_20180413_orca_detection_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0f172a")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_v4_scatter(v4, output_dir):
    """Figure 2: v4_clean scatter — logit score vs time, all classes."""
    df = pd.read_csv(v4)
    df["utc_hour"] = df.apply(
        lambda r: parse_utc_hour(r["filename"], r["window_start"]), axis=1)

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("#0f172a")
    _style(ax)

    for lbl, col in [("orca_call", colors["v4_orca"]),
                     ("dolphin_call", colors["dolphin"]),
                     ("other", colors["other"])]:
        sub = df[df["label"] == lbl]
        if len(sub):
            ax.scatter(sub["utc_hour"], sub["logits"],
                       c=col, alpha=0.5, s=12, label=f"{lbl} (n={len(sub)})")

    ax.axhline(0, color="#ef4444", linewidth=0.8, linestyle="--", label="threshold=0")
    ax.set_xlabel("UTC Hour", color="#94a3b8", fontsize=9)
    ax.set_ylabel("Logit score", color="#94a3b8", fontsize=9)
    ax.set_title("v4_clean — April 13 2018 — all detections by class and logit score",
                 color="#e2e8f0", fontsize=11)
    ax.legend(fontsize=9, facecolor="#1e293b", labelcolor="#e2e8f0")
    for spine in ax.spines.values(): spine.set_edgecolor("#334155")
    _add_orca_shading(ax)

    out = os.path.join(output_dir, "MARS_20180413_v4_clean_scatter.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0f172a")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_dolphin_comparison(v3, v4, output_dir):
    """Figure 3: dolphin detections v3 vs v4."""
    bins = np.arange(0, 25, 1)
    df3 = load_detections(v3, "dolphin_call")
    df4 = load_detections(v4, "dolphin_call")

    fig, ax = plt.subplots(figsize=(14, 4))
    fig.patch.set_facecolor("#0f172a")
    _style(ax)
    ax.hist(df3["utc_hour"], bins=bins, alpha=0.7, color=colors["v3_orca"],
            label=f"v3_clean dolphin_call (n={len(df3)})")
    ax.hist(df4["utc_hour"], bins=bins, alpha=0.6, color=colors["dolphin"],
            label=f"v4_clean dolphin_call (n={len(df4)})")
    ax.set_ylabel("Detections/hr", color="#94a3b8", fontsize=9)
    ax.set_title("Dolphin detections — v3_clean vs v4_clean (April 13 2018)",
                 color="#e2e8f0", fontsize=10)
    ax.legend(fontsize=8, facecolor="#1e293b", labelcolor="#e2e8f0")
    for spine in ax.spines.values(): spine.set_edgecolor("#334155")
    _add_orca_shading(ax)

    out = os.path.join(output_dir, "MARS_20180413_dolphin_v3_v4_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0f172a")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_multiday(apr01, apr13, apr20, apr30, output_dir):
    """Figure 4: four-day comparison — orca + dolphin + other."""
    bins = np.arange(0, 25, 1)

    days = {
        "Apr 1 (quiet)":   (apr01,  colors["apr01"]),
        "Apr 13 (orca)":   (apr13,  colors["apr13"]),
        "Apr 20 (vessel)": (apr20,  colors["apr20"]),
        "Apr 30 (humpback)":(apr30, colors["apr30"]),
    }

    fig, axes = plt.subplots(3, 1, figsize=(14, 12),
                             gridspec_kw={"hspace": 0.5})
    fig.patch.set_facecolor("#0f172a")
    fig.suptitle("MARS Hydrophone — April 2018 — Four-Day Comparison\n"
                 "v4_clean classifier (ROC-AUC 0.974)",
                 color="#e2e8f0", fontsize=12, y=0.99)

    for ax, label_class, title in [
        (axes[0], "orca_call",    "Orca detections — four days"),
        (axes[1], "dolphin_call", "Dolphin detections — four days"),
        (axes[2], "other",        "Other/vessel detections — four days"),
    ]:
        _style(ax)
        for day_label, (csv_path, col) in days.items():
            df = load_detections(csv_path, label_class)
            n = len(df)
            if n > 0:
                ax.hist(df["utc_hour"], bins=bins, alpha=0.7,
                        color=col, label=f"{day_label} (n={n})")
        ax.set_ylabel("Detections/hr", color="#94a3b8", fontsize=9)
        ax.set_title(title, color="#e2e8f0", fontsize=10)
        ax.legend(fontsize=8, facecolor="#1e293b", labelcolor="#e2e8f0")
        for spine in ax.spines.values(): spine.set_edgecolor("#334155")
        if label_class == "orca_call":
            _add_orca_shading(ax)

    out = os.path.join(output_dir, "MARS_April2018_four_day_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0f172a")
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v1",    required=True, help="v1_clean detections CSV (Apr 13)")
    ap.add_argument("--v2",    required=True, help="v2_clean detections CSV (Apr 13)")
    ap.add_argument("--v3",    required=True, help="v3_clean detections CSV (Apr 13)")
    ap.add_argument("--v4",    required=True, help="v4_clean detections CSV (Apr 13)")
    ap.add_argument("--apr01", required=True, help="v4_clean detections CSV (Apr 1)")
    ap.add_argument("--apr20", required=True, help="v4_clean detections CSV (Apr 20)")
    ap.add_argument("--apr30", required=True, help="v4_clean detections CSV (Apr 30)")
    ap.add_argument("--output-dir", default=".")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    plot_model_comparison(args.v1, args.v2, args.v3, args.v4, args.output_dir)
    plot_v4_scatter(args.v4, args.output_dir)
    plot_dolphin_comparison(args.v3, args.v4, args.output_dir)
    plot_multiday(args.apr01, args.v4, args.apr20, args.apr30, args.output_dir)

    print("\nAll plots saved to", args.output_dir)


if __name__ == "__main__":
    main()
