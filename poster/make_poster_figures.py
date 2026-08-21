#!/usr/bin/env python3
"""
Figures for the IEEE OCEANS Monterey 2026 poster
"Application of Foundation Model and Agile Modeling to Passive Acoustic
Detection of Orcas in Monterey Bay National Marine Sanctuary"
(D. R. Edgington, J. Ryan, MBARI).

Produces three figures, each sized to its final printed dimensions on the
72 x 48 in poster, in both PNG (300 dpi) and PDF (vector):

  oceans2026_fig2_agile_modeling_loop        16.05 x 5.9 in
  oceans2026_fig3_classifier_trajectory      16.05 x 7.2 in
  oceans2026_fig9_per_class_f1_v4            16.05 x 4.2 in

Data sources (all verified against the perch-hoplite repo, Aug 2026):
  README.md                    -- classifier table (ROC-AUC / top1 / cmap / macro F1)
  docs/agile_modeling_history.md -- session dates, labels per version, key insights
  CLAUDE_perch_hoplite.md      -- per-class F1 for v1/v2/v4

Usage:  python make_poster_figures.py [outdir]
"""

import sys
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

# ---------------------------------------------------------------- style ----
NAVY = "#0B2545"
DEEP = "#13315C"
TEAL = "#1C7293"
MINT = "#34C6A8"
CARD = "#EEF3F7"
MUTED = "#4A5C6A"
FAIL = "#B4553F"
WHITE = "#FFFFFF"

FONT = "Liberation Sans"   # metric-compatible with Arial, the poster body face
plt.rcParams.update({
    "font.family": FONT,
    "text.color": NAVY,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": NAVY,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "savefig.facecolor": CARD,
    "figure.facecolor": CARD,
    "axes.facecolor": CARD,
})

OUT = sys.argv[1] if len(sys.argv) > 1 else "."


def save(fig, name):
    for ext in ("png", "pdf"):
        path = os.path.join(OUT, f"{name}.{ext}")
        fig.savefig(path, dpi=300, facecolor=CARD)
        print("wrote", path)
    plt.close(fig)


# ================================================================= FIG 2 ===
# Agile-modeling loop: fixed pipeline on the left, the labeling loop on the
# right, inference at the bottom.

def fig2_loop():
    W, H = 16.05, 5.9
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    def box(x, y, w, h, title, sub=None, fc=WHITE, ec=TEAL, tc=NAVY, lw=2.0):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.12",
            facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2))
        if sub:
            ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center",
                    fontsize=20, fontweight="bold", color=tc, zorder=3)
            ax.text(x + w / 2, y + h * 0.26, sub, ha="center", va="center",
                    fontsize=16, color=MUTED if tc == NAVY else tc, zorder=3)
        else:
            ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                    fontsize=20, fontweight="bold", color=tc, zorder=3)

    def arrow(p1, p2, color=TEAL, rad=0.0, lw=2.4, ls="-", cs=None):
        ax.add_patch(FancyArrowPatch(
            p1, p2, arrowstyle="-|>", mutation_scale=26,
            connectionstyle=cs or f"arc3,rad={rad}", color=color,
            linewidth=lw, linestyle=ls, zorder=1,
            shrinkA=2, shrinkB=2))

    # --- fixed, computed once (left) ---
    ax.text(0.15, H - 0.32, "COMPUTED ONCE", fontsize=15, fontweight="bold",
            color=MUTED)
    box(0.15, 3.05, 3.05, 1.35, "MARS archive", "continuous PAM audio")
    box(0.15, 1.30, 3.05, 1.35, "Normalize", "per-window peak \u2192 0.25")
    box(3.65, 2.15, 3.15, 2.25, "Perch 2.0", "frozen encoder\n(never fine-tuned)",
        fc=DEEP, ec=DEEP, tc=WHITE)
    box(7.25, 2.15, 3.05, 2.25, "Hoplite", "embedding\nvector database")

    arrow((1.68, 3.05), (1.68, 2.72))                      # archive -> normalize
    arrow((3.22, 1.98), (3.65, 2.85), rad=0.18)            # normalize -> perch
    arrow((6.82, 3.28), (7.25, 3.28))                      # perch -> hoplite

    # --- the loop (right) ---
    ax.text(10.95, H - 0.32, "THE LOOP  \u2014  ~8\u201310 h of one expert's time",
            fontsize=15, fontweight="bold", color=MUTED)
    lx, rx = 10.95, 13.55
    ty, by = 3.35, 1.15
    bw, bh = 2.35, 1.25
    box(lx, ty, bw, bh, "Vector search", "nearest neighbours")
    box(rx, ty, bw, bh, "Expert label", "Gradio review")
    box(rx, by, bw, bh, "Refit", "linear head, ~30 s")
    box(lx, by, bw, bh, "Inspect errors", "\u2192 hard negatives", ec=MINT)

    arrow((lx + bw, ty + bh / 2), (rx, ty + bh / 2), MINT)          # search -> label
    arrow((rx + bw / 2, ty), (rx + bw / 2, by + bh), MINT)          # label -> refit
    arrow((rx, by + bh / 2), (lx + bw, by + bh / 2), MINT)          # refit -> inspect
    arrow((lx + bw / 2, by + bh), (lx + bw / 2, ty), MINT)          # inspect -> search

    arrow((10.30, 3.28), (lx, ty + bh / 2), rad=-0.12)              # hoplite -> loop

    # --- output ---
    box(3.65, 0.28, 6.65, 1.15, "Inference over months of audio",
        "orca logit \u2265 +1.16 operating threshold", fc=NAVY, ec=NAVY, tc=WHITE)
    arrow((5.2, 2.15), (5.2, 1.43), NAVY)                           # perch -> inference
    arrow((rx + bw / 2, by), (10.32, 0.86), color=NAVY,             # refit -> inference
          cs="angle,angleA=-90,angleB=0,rad=18")
    ax.text(12.55, 0.52, "deploy v4", fontsize=15, color=NAVY,
            style="italic", ha="center")

    ax.text(W - 0.1, 0.12,
            "Labels: 1,450 across the campaign  \u00b7  production model v4",
            ha="right", va="bottom", fontsize=15, color=MUTED, style="italic")
    return fig


# ================================================================= FIG 3 ===
# v0 -> v8 trajectory: metrics on top, labels per wave below.

VERSIONS = [
    # ver, date, roc, cmap, new_labels(main), new_labels(background), status
    ("v0", "Jul 9",     0.9773, 0.8810, 584, 0,   "ok"),
    ("v1", "Jul 9\u201310", 0.9533, 0.7999, 214, 0,   "ok"),
    ("v2", "Jul 10",    0.9654, 0.8930, 50,  0,   "ok"),
    ("v3", "Jul 11",    0.9467, 0.7370, 17,  0,   "ok"),
    ("v4", "Jul 11",    0.9590, 0.8297, 8,   0,   "prod"),
    ("v5", "Jul 15",    0.9303, 0.5945, 0,   0,   "fail"),
    ("v6", "Jul 16",    0.9499, 0.7763, 227, 350, "fail"),
    ("v7", "Jul 16",    0.9499, 0.7763, 0,   0,   "fail"),
    ("v8", "Jul 17",    0.9463, 0.6489, 0,   0,   "fail"),
]


def fig3_trajectory():
    """v0 -> v4 only.

    Per Duane's instruction of 20 Aug 2026, the experimental versions (context-embedding
    averaging, and the 4-season retrain that inflated ship_noise) are NOT presented as
    numbered stops on the development timeline. They survive as one muted line of poster
    text, not as data points here. The clean arc is v0 -> v1 -> v2 -> v3 -> v4, and v4 is
    the production model throughout the poster.
    """
    VS = VERSIONS[:5]
    W, H = 14.0, 5.6
    fig = plt.figure(figsize=(W, H))
    axm = fig.add_axes([0.075, 0.455, 0.900, 0.340])   # metrics
    axl = fig.add_axes([0.075, 0.175, 0.900, 0.200])   # labels

    xs = list(range(len(VS)))
    roc = [v[2] for v in VS]
    cmap_ = [v[3] for v in VS]

    for ax in (axm, axl):
        ax.set_xlim(-0.45, 5.15)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    axm.plot(xs, roc, "-o", color=TEAL, linewidth=2.8, markersize=10, zorder=3)
    axm.plot(xs, cmap_, "-o", color=NAVY, linewidth=2.8, markersize=10, zorder=3)
    axm.plot([4], [cmap_[4]], "o", color=MINT, markersize=22,
             markeredgecolor=NAVY, markeredgewidth=2.4, zorder=4)
    axm.text(4.30, roc[-1] + 0.004, "ROC-AUC", fontsize=17, color=TEAL,
             fontweight="bold", ha="left", va="bottom")
    axm.text(4.30, cmap_[-1] - 0.004, "cmap", fontsize=17, color=NAVY,
             fontweight="bold", ha="left", va="top")
    axm.set_ylim(0.66, 1.09)
    axm.set_yticks([0.7, 0.8, 0.9, 1.0])
    axm.tick_params(labelsize=16)
    axm.set_xticks([])
    axm.grid(axis="y", color="#D8E2EA", linewidth=1.1)
    axm.set_axisbelow(True)

    axm.annotate("17 hard-negative labels:\nApril-2026 false positives 6,489 \u2192 304",
                 xy=(3, cmap_[3] - 0.008), xytext=(1.75, 0.695),
                 fontsize=16, color=NAVY, ha="center", linespacing=1.4,
                 arrowprops=dict(arrowstyle="-|>", color=TEAL, linewidth=2.0,
                                 connectionstyle="arc3,rad=-0.2"))
    axm.annotate("v4 \u2014 production model\nROC-AUC 0.959 \u00b7 cmap 0.830 \u00b7 orca F1 0.947 @ +1.16",
                 xy=(4, cmap_[4] + 0.016), xytext=(3.15, 1.015),
                 fontsize=16, color=NAVY, ha="center", fontweight="bold",
                 linespacing=1.4,
                 arrowprops=dict(arrowstyle="-|>", color=MINT, linewidth=2.4))

    main = [v[4] for v in VS]
    colors = [MINT if v[6] == "prod" else TEAL for v in VS]
    axl.bar(xs, main, width=0.5, color=colors, zorder=3)
    axl.set_ylim(0, 700)
    axl.set_yticks([0, 300, 600])
    axl.tick_params(labelsize=16)
    axl.grid(axis="y", color="#D8E2EA", linewidth=1.1)
    axl.set_axisbelow(True)
    axl.set_ylabel("new labels", fontsize=17, color=NAVY, labelpad=10)
    for i, m in enumerate(main):
        axl.text(i, m + 28, f"+{m:,}", ha="center", va="bottom", fontsize=16, color=NAVY)
    axl.set_xticks(xs)
    axl.set_xticklabels([f"{v[0]}\n{v[1]}" for v in VS], fontsize=17, color=NAVY,
                        linespacing=1.4)
    axl.get_xticklabels()[4].set_fontweight("bold")

    fig.text(0.075, 0.945, "Five labeling waves over three days",
             fontsize=22, fontweight="bold", color=NAVY)
    fig.text(0.075, 0.895,
             "873 labels build the production model; the whole campaign totals ~1,450 "
             "in roughly 8\u201310 hours.",
             fontsize=15, color=MUTED)
    fig.text(0.075, 0.845,
             "ROC-AUC and cmap are 0\u20131 accuracy scores on held-out data \u2014 higher is better.",
             fontsize=14, color=MUTED)
    fig.text(0.975, 0.008,
             "Source: perch-hoplite README.md and docs/agile_modeling_history.md.",
             fontsize=13, color=MUTED, ha="right", style="italic")
    return fig


# ================================================================= FIG 9 ===
# Per-class F1 for v4, at the 0.0 inference default vs the F1-optimal threshold.

CLASSES = [
    # label, n, F1@0.0, F1 opt, opt thr, trustworthy
    ("orca_call",     45, 0.841, 0.947, "+1.16", True),
    ("other",         10, 0.645, 0.889, "+1.99", True),
    ("dolphin_call",  38, 0.531, 0.765, "+2.05", True),
    ("humpback_song", 47, 0.450, 0.548, "+0.98", True),
    ("ship_noise",     3, 1.000, 1.000, "+0.16", False),
]


def fig9_per_class_f1():
    W, H = 16.05, 4.2
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0.245, 0.24, 0.505, 0.545])

    ys = list(range(len(CLASSES)))[::-1]
    h = 0.34
    for y, (name, n, f0, fo, thr, ok) in zip(ys, CLASSES):
        ax.barh(y + h / 2 + 0.02, fo, height=h,
                color=MINT if ok else "#C9D6DF",
                hatch="" if ok else "///", edgecolor=WHITE, zorder=3)
        ax.barh(y - h / 2 - 0.02, f0, height=h,
                color=TEAL if ok else "#C9D6DF",
                alpha=1.0 if ok else 0.6,
                hatch="" if ok else "///", edgecolor=WHITE, zorder=3)
        ax.text(fo + 0.012, y + h / 2 + 0.02, f"{fo:.2f}", va="center",
                fontsize=15.5, color=NAVY, fontweight="bold")
        ax.text(f0 + 0.012, y - h / 2 - 0.02, f"{f0:.2f}", va="center",
                fontsize=15, color=MUTED)
        ax.text(-0.025, y, f"{name}   n={n}, cutoff {thr}", ha="right", va="center",
                fontsize=15.5, color=NAVY if ok else MUTED)

    ax.set_xlim(0, 1.10)
    ax.set_ylim(-0.75, len(CLASSES) - 0.25)
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(labelsize=15)
    ax.grid(axis="x", color="#D8E2EA", linewidth=1.1)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)


    handles = [
        Line2D([], [], marker="s", linestyle="", markersize=15, color=MINT,
               label="at each class's best cutoff"),
        Line2D([], [], marker="s", linestyle="", markersize=15, color=TEAL,
               label="at the software default of 0.0"),
    ]
    leg = ax.legend(handles=handles, fontsize=15.5, frameon=False,
                    loc="upper right", bbox_to_anchor=(1.02, -0.10), ncol=2)
    for t in leg.get_texts():
        t.set_color(NAVY)

    fig.text(0.245, 0.925, "How well each class is detected",
             fontsize=21, fontweight="bold", color=NAVY)
    fig.text(0.245, 0.862,
             "F1 folds missed calls and false alarms into one score: 0 = useless, 1 = perfect. "
             "296 held-out windows.",
             fontsize=14.5, color=MUTED)

    fig.text(0.775, 0.80,
             "Orca is strong \u2014 but only above\nour cutoff. At the software default,\n"
             "one detection in four is a false alarm.",
             fontsize=15.5, color=NAVY, va="top", linespacing=1.5)
    fig.text(0.775, 0.50,
             "Humpback is the weakest class we\ntrust the numbers for \u2014 and evidence\n"
             "that some of its labels are really\ngray whale.",
             fontsize=15.5, color=NAVY, va="top", linespacing=1.5)
    fig.text(0.775, 0.145,
             "ship_noise: only 3 held-out examples.\nIts 1.00 is an artifact, not skill.",
             fontsize=15, color=MUTED, va="top", style="italic", linespacing=1.5)
    return fig


# ================================================================= FIG 4 ===
# April-May 2018 detection calendar.
# Data: poster_fig4_calendar_apr_may_2018.csv (per-day v4 orca_call detections).
# Apr 21 2018 carries a real, never-reviewed signal (40 / 25 @ >=1.16); it is
# drawn hatched grey and labelled "pending review" -- neither claimed nor
# dismissed -- per the project's instruction of Aug 19 2026.

import csv
from datetime import date

CAL_CSV = "poster_fig4_calendar_apr_may_2018.csv"

# Live-DB resolution of 21 Aug 2026 (CLAUDE_poster_annotation_status_apr21_v2.md): April 21
# 2018 was reviewed -- all 25 clips at >=1.16 confirmed orca by ear (D. Edgington). The CSV
# still carries the older UNREVIEWED_ACTION_ITEM status, so it is overridden here until the
# CSV is regenerated. Confirmed orca days are now EIGHT, not seven.
STATUS_OVERRIDE = {
    "2018-04-21": "CONFIRMED",
}

STATUS_STYLE = {
    "CONFIRMED":              (MINT,      1.00, ""),
    "unreviewed":             (TEAL,      0.55, ""),
    "UNREVIEWED_ACTION_ITEM": ("#9AA9B4", 0.95, "////"),
    "too_faint_unlabeled":    ("#E3A857", 0.95, ""),
    "unreviewed_noise":       ("#C1D0DA", 0.95, ""),
}


def _load_calendar(path=CAL_CSV):
    rows = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            y, m, d = (int(v) for v in r["date"].split("-"))
            status = STATUS_OVERRIDE.get(r["date"], r["confirmation_status"])
            rows[date(y, m, d)] = (int(r["total_detections"]),
                                   int(r["detections_ge116"]),
                                   status)
    return rows


def fig4_calendar(path=CAL_CSV):
    rows = _load_calendar(path)
    W, H = 16.05, 6.6
    fig = plt.figure(figsize=(W, H))
    axes = [fig.add_axes([0.055, 0.545, 0.925, 0.305]),
            fig.add_axes([0.055, 0.145, 0.925, 0.305])]

    months = [(4, "April 2018", 30), (5, "May 2018", 31)]
    for ax, (mon, name, ndays) in zip(axes, months):
        days = list(range(1, ndays + 1))
        tot, ge, sts = [], [], []
        for d in days:
            t, g, s = rows.get(date(2018, mon, d), (0, 0, "unreviewed_noise"))
            tot.append(t); ge.append(g); sts.append(s)

        ax.bar(days, tot, width=0.72, color="#DCE6ED", zorder=2)
        for d, g, s in zip(days, ge, sts):
            c, a, hatch = STATUS_STYLE[s]
            ax.bar([d], [g], width=0.72, color=c, alpha=a, hatch=hatch,
                   edgecolor=WHITE if hatch else "none", linewidth=0, zorder=3)

        ax.set_xlim(0.3, 31.7)
        ax.set_ylim(0, 310)
        ax.set_xticks(days)
        ax.set_xticklabels([str(d) for d in days], fontsize=13.5)
        ax.set_yticks([0, 100, 200, 300])
        ax.tick_params(labelsize=14)
        ax.grid(axis="y", color="#D8E2EA", linewidth=1.0)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.text(0.35, 285, name, fontsize=19, fontweight="bold", color=NAVY,
                va="top")

        for d, t, g, s in zip(days, tot, ge, sts):
            if s == "CONFIRMED":
                ax.text(d, max(t, g) + 12, f"{g}", ha="center", fontsize=14,
                        fontweight="bold", color=NAVY)
            elif s == "too_faint_unlabeled":
                ax.annotate("listened to,\ntoo faint to call", xy=(d, t + 8),
                            xytext=(d, 95), fontsize=13.5, color="#A9762E",
                            ha="center", va="bottom", linespacing=1.3,
                            arrowprops=dict(arrowstyle="-|>", color="#E3A857",
                                            linewidth=1.8))
            elif s == "unreviewed" and g > 15:
                ax.text(d, max(t, g) + 12, f"{g}", ha="center", fontsize=13,
                        color=MUTED)

    axes[0].set_ylabel("detections", fontsize=15, color=NAVY, labelpad=8)
    axes[1].set_ylabel("detections", fontsize=15, color=NAVY, labelpad=8)

    handles = [
        Line2D([], [], marker="s", linestyle="", markersize=15, color=MINT,
               label="orca, confirmed by listening"),
        Line2D([], [], marker="s", linestyle="", markersize=15, color=TEAL,
               alpha=0.55, label="detected, not individually listened to"),
        Line2D([], [], marker="s", linestyle="", markersize=15, color="#E3A857",
               label="listened to, too faint to call"),
        Line2D([], [], marker="s", linestyle="", markersize=15, color="#DCE6ED",
               label="all detections, before the cutoff"),
    ]
    leg = fig.legend(handles=handles, fontsize=14.5, frameon=False, ncol=4,
                     loc="lower center", bbox_to_anchor=(0.52, 0.005))
    for t in leg.get_texts():
        t.set_color(NAVY)

    fig.text(0.055, 0.955, "Two months of listening, eight days of orca",
             fontsize=22, fontweight="bold", color=NAVY)
    fig.text(0.055, 0.905,
             "Detections per day above our score cutoff (solid) against everything the detector "
             "flagged (pale). Spring 2018 is episodic, not continuous.",
             fontsize=16, color=MUTED)
    return fig


# ================================================================= FIG 8 ===
# Threshold sweep: confirmed events hold, silent months collapse.
# Data: poster_fig8_threshold_sweep_v4.csv

# Note on the CSV's `retain_pct_T1.16` column: it is det_T1.16 / `ref`, where
# `ref` is the original expert-labeled reference count -- NOT det_T1.16 /
# det_T0.00. Only 3 of the 10 rows carry a `ref`, so this figure deliberately
# plots absolute detection counts on a log axis and quotes counts only.
# (Confirmed with the project maintainer, Aug 2026.)
SWEEP_CSV = "poster_fig8_threshold_sweep_v4.csv"
THRESHOLDS = [0.00, 1.00, 1.16, 1.50, 2.00]
SWEEP_KEEP = [
    ("Apr 13 2018 - Bigg's event",       "13 Apr 2018",  NAVY,  "-",  True,  None),
    ("Apr 18 2018 - strong day",         "18 Apr 2018",  TEAL,  "-",  True,  None),
    ("May 12 2018 - event",              "12 May 2018",  MINT,  "-",  True,  None),
    ("May 14 2018 - secondary",          "14 May 2018",  "#7FA8B8", "-", True, 1.55),
    ("October 2020 - full month",        "Oct 2020",     FAIL,  "--", False, 0.83),
    ("April 2026 - full month",          "Apr 2026",     "#D08C6E", "--", False, None),
]


def fig8_threshold_sweep(path=SWEEP_CSV):
    data = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            data[r["region"]] = [float(r[f"det_T{t:.2f}"]) for t in THRESHOLDS]

    W, H = 16.05, 4.6
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0.062, 0.185, 0.555, 0.60])
    xs = list(range(len(THRESHOLDS)))

    ax.axvspan(1.78, 2.22, color=WHITE, zorder=0)

    for key, label, color, ls, confirmed, label_y in SWEEP_KEEP:
        ys = data[key]
        ax.plot(xs, ys, ls, color=color, linewidth=3.0 if confirmed else 2.4,
                marker="o", markersize=8, zorder=3)
        ax.text(4.08, label_y if label_y else ys[-1], f"  {label}", fontsize=14.5,
                color=color, va="center",
                fontweight="bold" if confirmed else "normal")

    ax.set_yscale("log")
    ax.set_ylim(0.7, 620)
    ax.set_yticks([1, 10, 100])
    ax.set_yticklabels(["1", "10", "100"], fontsize=15)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{t:.2f}" if t else "0.00" for t in THRESHOLDS], fontsize=15)
    tick = ax.get_xticklabels()[2]
    tick.set_fontweight("bold")
    tick.set_color(NAVY)
    ax.text(2.0, 2.15, "our cutoff\n(default is 0.00)", ha="center", va="center",
            fontsize=13, color=MUTED, linespacing=1.3)
    ax.set_xlim(-0.12, 4.02)
    ax.set_xlabel("score cutoff  (detector output; higher = more orca-like)",
                  fontsize=15, color=NAVY, labelpad=6)
    ax.set_ylabel("detections", fontsize=15.5, color=NAVY, labelpad=6)
    ax.grid(axis="y", color="#D8E2EA", linewidth=1.0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    fig.text(0.062, 0.915, "Raising the cutoff separates events from false alarms",
             fontsize=21, fontweight="bold", color=NAVY)

    fig.text(0.745, 0.78,
             "Real events decline gently:\n13 April keeps 251 of its 285\ndetections at our cutoff.",
             fontsize=14.5, color=NAVY, va="top", linespacing=1.5)
    fig.text(0.745, 0.50,
             "Silent months collapse: October\n2020 falls 144 \u2192 10, April 2026\n323 \u2192 23 \u2014 and all ten October\nsurvivors were humpback.",
             fontsize=14.5, color=NAVY, va="top", linespacing=1.5)
    fig.text(0.745, 0.20,
             "14 May is small-n and falls like a false-\npositive curve \u2014 yet all four clips are\norca by ear. Shape alone is not proof.",
             fontsize=13.5, color=MUTED, va="top", style="italic", linespacing=1.5)
    return fig


if __name__ == "__main__":
    save(fig2_loop(), "oceans2026_fig2_agile_modeling_loop")
    save(fig3_trajectory(), "oceans2026_fig3_classifier_trajectory")
    save(fig9_per_class_f1(), "oceans2026_fig9_per_class_f1_v4")
    save(fig4_calendar(), "oceans2026_fig4_calendar_apr_may_2018")
    save(fig8_threshold_sweep(), "oceans2026_fig8_threshold_sweep_v4")


