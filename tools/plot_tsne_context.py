#!/usr/bin/env python3
"""tools/plot_tsne_context.py
t-SNE visualization using 30-second context-averaged embeddings.

For each labeled window, fetches all available windows within ±15s
from the same recording, applies Gaussian weights centered on the
detection (sigma=1.5 windows), and averages to produce a 1536-dim
context embedding. This can sharpen clustering by capturing bout-level
acoustic context rather than a single 5-second snapshot.

Usage:
    python3 tools/plot_tsne_context.py \
        --db-dir /mnt/PAM_Analysis/perch-hoplite/db/MARS_20180401_20180430_32kHz_norm \
                 /mnt/PAM_Analysis/perch-hoplite/db/MARS_20201001_20201031_32kHz_norm \
                 /mnt/PAM_Analysis/perch-hoplite/db/MARS_20260401_20260430_32kHz_norm \
        --output /mnt/PAM_Analysis/perch-hoplite/results/tsne_context_3season.png \
        --title "Perch V2 Context Embeddings — 30s window, Gaussian weighted"
"""
import argparse
import sqlite3
import struct
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# Gaussian weighting: sigma in units of 5-second windows
SIGMA_WINDOWS = 1.5   # ~7.5 seconds — emphasises the detection window
CONTEXT_WINDOWS = 3   # ±3 windows = ±15 seconds


LABEL_COLORS = {
    "orca_call":     "#16a34a",
    "humpback_song": "#d97706",
    "dolphin_call":  "#9333ea",
    "ship_noise":    "#0891b2",
    "other":         "#ea580c",
    "negative":      "#6b7280",
}
DEFAULT_COLOR = "#94a3b8"


def gaussian_weight(delta_windows, sigma=SIGMA_WINDOWS):
    return np.exp(-0.5 * (delta_windows / sigma) ** 2)


def load_context_embeddings(db_path: str):
    """
    For every labeled window in the DB, compute a Gaussian-weighted
    average of all available embeddings within ±CONTEXT_WINDOWS.
    Returns (context_embs [N, 1536], labels [N], db_name).
    """
    from perch_hoplite.db import sqlite_usearch_impl

    db = sqlite_usearch_impl.SQLiteUSearchDB.create(db_path, readonly=True)
    db_name = Path(db_path).name
    sqlite_path = str(Path(db_path) / "hoplite.sqlite")
    con = sqlite3.connect(sqlite_path)

    # ── 1. Fetch all labeled windows ──────────────────────────────────
    labeled = con.execute("""
        SELECT a.label, a.label_type, w.id, a.offsets, a.recording_id
        FROM annotations a
        JOIN windows w ON w.recording_id = a.recording_id
                      AND w.offsets = a.offsets
        WHERE a.label_type = 1
    """).fetchall()

    if not labeled:
        print(f"  {db_name}: no positive annotations found")
        con.close()
        return np.empty((0, 1536)), [], db_name

    # ── 2. Build a recording → sorted windows index ───────────────────
    # Fetch all windows per recording so we can look up neighbors fast
    print(f"  {db_name}: indexing windows for {len(labeled)} labeled examples...")
    rec_windows = defaultdict(list)   # rec_id → [(start_s, window_id)]
    for row in con.execute("SELECT recording_id, id, offsets FROM windows"):
        rec_id, wid, blob = row
        if blob and len(blob) >= 8:
            start_s = struct.unpack_from('<d', blob)[0]
        else:
            start_s = 0.0
        rec_windows[rec_id].append((start_s, wid))

    # Sort numerically — never trust blob byte order
    for rec_id in rec_windows:
        rec_windows[rec_id].sort(key=lambda x: x[0])

    con.close()

    # ── 3. For each labeled window, compute context embedding ─────────
    context_embs = []
    labels_out   = []

    # Collect all window IDs we'll need, then batch-fetch
    needed_ids = set()
    todo = []  # (label, center_wid, neighbor_wids_with_deltas)

    for label, label_type, center_wid, off_blob, rec_id in labeled:
        if off_blob and len(off_blob) >= 8:
            center_s = struct.unpack_from('<d', off_blob)[0]
        else:
            center_s = 0.0

        windows = rec_windows[rec_id]
        # Find all windows within ±CONTEXT_WINDOWS * 5s
        radius_s = CONTEXT_WINDOWS * 5.0
        neighbors = []
        for s, wid in windows:
            delta_s = s - center_s
            if abs(delta_s) <= radius_s + 0.1:   # small epsilon for float
                delta_w = delta_s / 5.0           # convert to window units
                neighbors.append((delta_w, wid))
                needed_ids.add(wid)

        if not neighbors:
            neighbors = [(0.0, center_wid)]
            needed_ids.add(center_wid)

        todo.append((label, neighbors))

    # Batch-fetch all needed embeddings
    print(f"  {db_name}: fetching {len(needed_ids)} embeddings (batch)...")
    id_list = sorted(needed_ids)
    emb_matrix = db.get_embeddings_batch(id_list)          # [M, 1536]
    id_to_row  = {wid: i for i, wid in enumerate(id_list)}

    # Compute weighted averages
    for label, neighbors in todo:
        weights = np.array([gaussian_weight(dw) for dw, _ in neighbors])
        weights /= weights.sum()

        embs = np.stack([
            emb_matrix[id_to_row[wid]] for _, wid in neighbors
        ])                                                  # [K, 1536]

        ctx_emb = (weights[:, None] * embs).sum(axis=0)    # [1536]

        # L2-normalise so cosine distance still works in t-SNE
        norm = np.linalg.norm(ctx_emb)
        if norm > 1e-8:
            ctx_emb /= norm

        context_embs.append(ctx_emb)
        labels_out.append(label)

    print(f"  {db_name}: {len(context_embs)} context embeddings ready")
    return np.stack(context_embs), labels_out, db_name


def plot_tsne(all_embs, all_labels, all_dbs, output, title):
    from sklearn.manifold import TSNE

    print(f"\nRunning t-SNE on {len(all_embs)} context embeddings (1536 dims) → 2D ...")
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000,
                random_state=42, verbose=1)
    coords = tsne.fit_transform(all_embs)

    # ── Plot ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    unique_labels = sorted(set(all_labels))
    for lbl in unique_labels:
        idx = [i for i, l in enumerate(all_labels) if l == lbl]
        color = LABEL_COLORS.get(lbl, DEFAULT_COLOR)
        ax.scatter(coords[idx, 0], coords[idx, 1],
                   c=color, s=40, alpha=0.85, edgecolors='none',
                   label=f"{lbl} (n={len(idx)})")

    ax.set_title(title, color="#e2e8f0", fontsize=13, pad=12)
    ax.set_xlabel("t-SNE dim 1", color="#94a3b8", fontsize=9)
    ax.set_ylabel("t-SNE dim 2", color="#94a3b8", fontsize=9)
    ax.tick_params(colors="#94a3b8", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")

    legend = ax.legend(framealpha=0.15, facecolor="#0f172a",
                       edgecolor="#334155", labelcolor="#e2e8f0",
                       fontsize=9, loc="upper right")

    db_str = " | ".join(set(all_dbs))
    ax.text(0.01, 0.01,
            f"Total: {len(all_embs)} labeled windows  |  "
            f"Context: ±{CONTEXT_WINDOWS} windows (±15s), Gaussian σ={SIGMA_WINDOWS}  |  "
            f"Perch V2 1536-dim → t-SNE 2D",
            transform=ax.transAxes, fontsize=7,
            color="#475569", va="bottom")

    plt.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight",
                facecolor="#0f172a")
    plt.close(fig)
    print(f"\nSaved: {output}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db-dir", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--title",
        default="Perch V2 Context Embeddings — 30s Gaussian-weighted window")
    args = ap.parse_args()

    all_embs, all_labels, all_dbs = [], [], []

    for db_path in args.db_dir:
        embs, labels, db_name = load_context_embeddings(db_path)
        if len(embs):
            all_embs.append(embs)
            all_labels.extend(labels)
            all_dbs.extend([db_name] * len(labels))

    if not all_embs:
        print("No embeddings found — check DB paths and annotations.")
        return

    all_embs = np.vstack(all_embs)
    print(f"\nTotal: {len(all_embs)} context embeddings from {len(args.db_dir)} DB(s)")
    plot_tsne(all_embs, all_labels, all_dbs, args.output, args.title)


if __name__ == "__main__":
    main()
