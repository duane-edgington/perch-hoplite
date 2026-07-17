#!/usr/bin/env python3
"""tools/plot_tsne_orca_events.py
t-SNE visualization highlighting the two confirmed orca events separately.
Colors orca by source month to show whether April 2018 and May 2018 orca
calls occupy the same or different regions of embedding space.

Usage:
    python3 tools/plot_tsne_orca_events.py \
        --db-dir \
            /mnt/PAM_Analysis/perch-hoplite/db/MARS_20180401_20180430_32kHz_norm \
            /mnt/PAM_Analysis/perch-hoplite/db/MARS_20180501_20180531_32kHz_norm \
            /mnt/PAM_Analysis/perch-hoplite/db/MARS_20201001_20201031_32kHz_norm \
            /mnt/PAM_Analysis/perch-hoplite/db/MARS_20260401_20260430_32kHz_norm \
        --output /mnt/PAM_Analysis/perch-hoplite/results/tsne_orca_events.png \
        --orca-only  # optional: plot only orca + humpback for clarity
"""
import argparse
import sqlite3
import struct
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# Standard colors
LABEL_COLORS = {
    "orca_apr2018":  "#16a34a",   # green — April 13 2018
    "orca_may2018":  "#86efac",   # light green — May 12 2018
    "humpback_song": "#d97706",   # amber
    "dolphin_call":  "#9333ea",   # purple
    "ship_noise":    "#0891b2",   # teal
    "other":         "#f43f5e",   # rose
    "negative":      "#6b7280",   # gray
}

# Which DB maps to which orca event label
ORCA_EVENT_MAP = {
    "MARS_20180401_20180430_32kHz_norm": "orca_apr2018",
    "MARS_20180501_20180531_32kHz_norm": "orca_may2018",
}


def load_embeddings(db_path, orca_only=False):
    from perch_hoplite.db import sqlite_usearch_impl

    db = sqlite_usearch_impl.SQLiteUSearchDB.create(db_path, readonly=True)
    db_name = Path(db_path).name
    con = sqlite3.connect(str(Path(db_path) / "hoplite.sqlite"))

    rows = con.execute("""
        SELECT a.label, a.label_type, w.id, w.offsets
        FROM annotations a
        JOIN windows w ON w.recording_id = a.recording_id
                      AND w.offsets = a.offsets
        WHERE a.label_type = 1
    """).fetchall()
    con.close()

    embs, labels = [], []
    window_ids = [r[2] for r in rows]
    if not window_ids:
        return np.empty((0, 1536)), []

    emb_matrix = db.get_embeddings_batch(window_ids).astype(np.float32)

    for i, (label, label_type, wid, _) in enumerate(rows):
        # Remap orca by event
        if label == "orca_call" and db_name in ORCA_EVENT_MAP:
            display_label = ORCA_EVENT_MAP[db_name]
        else:
            display_label = label

        if orca_only and display_label not in (
                "orca_apr2018", "orca_may2018", "humpback_song"):
            continue

        embs.append(emb_matrix[i])
        labels.append(display_label)

    print(f"  {db_name}: {len(embs)} embeddings")
    return np.array(embs) if embs else np.empty((0, 1536)), labels


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db-dir", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--orca-only", action="store_true",
                    help="Plot only orca and humpback for clarity")
    ap.add_argument("--title",
        default="Perch V2 Embeddings — Orca Events Separated by Month")
    args = ap.parse_args()

    from sklearn.manifold import TSNE

    all_embs, all_labels = [], []
    for db_path in args.db_dir:
        embs, labels = load_embeddings(db_path, args.orca_only)
        if len(embs):
            all_embs.append(embs)
            all_labels.extend(labels)

    if not all_embs:
        print("No embeddings found.")
        return

    all_embs = np.vstack(all_embs)
    print(f"\nTotal: {len(all_embs)} embeddings")

    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000,
                random_state=42, verbose=1)
    coords = tsne.fit_transform(all_embs)

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    # Define display order and labels
    display_names = {
        "orca_apr2018":  "orca — April 2018 (Apr 13 event)",
        "orca_may2018":  "orca — May 2018 (May 12 event)",
        "humpback_song": "humpback_song",
        "dolphin_call":  "dolphin_call",
        "ship_noise":    "ship_noise",
        "other":         "other",
        "negative":      "negative",
    }

    unique_labels = sorted(set(all_labels),
                           key=lambda x: list(display_names.keys()).index(x)
                           if x in display_names else 99)

    for lbl in unique_labels:
        idx = [i for i, l in enumerate(all_labels) if l == lbl]
        color = LABEL_COLORS.get(lbl, "#94a3b8")
        name  = display_names.get(lbl, lbl)
        # Orca events get larger markers
        size = 55 if lbl.startswith("orca") else 35
        alpha = 0.9 if lbl.startswith("orca") else 0.75
        ax.scatter(coords[idx, 0], coords[idx, 1],
                   c=color, s=size, alpha=alpha, edgecolors='none',
                   label=f"{name} (n={len(idx)})")

    ax.set_title(args.title, color="#e2e8f0", fontsize=13, pad=12)
    ax.set_xlabel("t-SNE dim 1", color="#94a3b8", fontsize=9)
    ax.set_ylabel("t-SNE dim 2", color="#94a3b8", fontsize=9)
    ax.tick_params(colors="#94a3b8", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")

    ax.legend(framealpha=0.15, facecolor="#0f172a", edgecolor="#334155",
              labelcolor="#e2e8f0", fontsize=9, loc="upper right")

    suffix = " (orca + humpback only)" if args.orca_only else ""
    ax.text(0.01, 0.01,
            f"Total: {len(all_embs)} labeled windows{suffix}  |  "
            f"Perch V2 1536-dim → t-SNE 2D",
            transform=ax.transAxes, fontsize=7, color="#475569", va="bottom")

    plt.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches="tight", facecolor="#0f172a")
    plt.close(fig)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
