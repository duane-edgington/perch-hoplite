#!/usr/bin/env python3
"""tools/filter_detections_context.py
Post-processing filter: suppress orca detections whose 30-second
acoustic context is dominated by humpback song.

For each orca detection, fetches ±CONTEXT_WINDOWS neighboring window
embeddings from the DB, applies the classifier to get all 5 logits,
and computes the ratio: mean_orca_logit / mean_humpback_logit.
Detections with ratio < --ratio-threshold are suppressed.

Usage:
    python3 tools/filter_detections_context.py \
        --detections-csv /mnt/PAM_Analysis/perch-hoplite/results/MARS_20260401_20260430_v4_detections.csv \
        --db-dir /mnt/PAM_Analysis/perch-hoplite/db/MARS_20260401_20260430_32kHz_norm \
        --classifier /mnt/PAM_Analysis/perch-hoplite/models/orca_v4.pt \
        --output-csv /mnt/PAM_Analysis/perch-hoplite/results/MARS_20260401_20260430_v4_ctx_filtered.csv \
        --ratio-threshold 1.0

NOTE: Option A (modify inference to output all 5 logits per window) is
tracked as a TODO — see end of this file.
"""
import argparse
import sqlite3
import struct
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from collections import defaultdict

CONTEXT_WINDOWS = 3   # ±3 windows = ±15 seconds


def load_classifier(classifier_path):
    # Inject full TF mock — matches phase2_classify.py exactly
    import sys, types, importlib.machinery as _imach
    if 'tensorflow' not in sys.modules:
        _tf = types.ModuleType('tensorflow')
        _tf.__spec__    = _imach.ModuleSpec('tensorflow', loader=None)
        _tf.__version__ = '0.0.0-mock'
        _tf.Tensor      = object
        _tf.keras       = types.ModuleType('tensorflow.keras')
        _tf.keras.__spec__ = _imach.ModuleSpec('tensorflow.keras', loader=None)
        _tf.keras.Model = object
        _tf.keras.layers    = types.ModuleType('tensorflow.keras.layers')
        _tf.keras.optimizers = types.ModuleType('tensorflow.keras.optimizers')
        _tf.keras.losses    = types.ModuleType('tensorflow.keras.losses')
        sys.modules['tensorflow']                  = _tf
        sys.modules['tensorflow.keras']            = _tf.keras
        sys.modules['tensorflow.keras.layers']     = _tf.keras.layers
        sys.modules['tensorflow.keras.optimizers'] = _tf.keras.optimizers
        sys.modules['tensorflow.keras.losses']     = _tf.keras.losses

    from perch_hoplite.agile import classifier as classifier_mod
    clf       = classifier_mod.LinearClassifier.load(classifier_path)
    beta      = np.array(clf.beta,      dtype=np.float32)
    beta_bias = np.array(clf.beta_bias, dtype=np.float32)
    classes   = list(clf.classes)
    return beta, beta_bias, classes


def apply_classifier(embeddings_np, beta, beta_bias):
    """embeddings_np: [N, 1536] float32 → logits [N, 5] float32"""
    return embeddings_np @ beta + beta_bias


def build_window_index(db_path, filenames):
    """
    Build a fast lookup: (filename, start_s) → window_id
    and filename → sorted list of (start_s, window_id)
    Only indexes recordings in `filenames` set.
    """
    con = sqlite3.connect(str(Path(db_path) / "hoplite.sqlite"))
    placeholders = ','.join('?' * len(filenames))
    rows = con.execute(f"""
        SELECT r.filename, w.id, w.offsets
        FROM windows w JOIN recordings r ON w.recording_id = r.id
        WHERE r.filename IN ({placeholders})
    """, list(filenames)).fetchall()
    con.close()

    by_file = defaultdict(list)
    for filename, wid, blob in rows:
        if blob and len(blob) >= 8:
            start_s = struct.unpack_from('<d', blob)[0]
        else:
            start_s = 0.0
        by_file[filename].append((start_s, wid))

    # Sort numerically
    for fname in by_file:
        by_file[fname].sort(key=lambda x: x[0])

    return by_file


def get_context_window_ids(by_file, filename, center_s, ctx=CONTEXT_WINDOWS):
    """Return list of window_ids within ±ctx*5s of center_s."""
    windows = by_file.get(filename, [])
    radius  = ctx * 5.0 + 0.1
    return [wid for s, wid in windows if abs(s - center_s) <= radius]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detections-csv", required=True,
                    help="Inference CSV (output of phase2_classify infer)")
    ap.add_argument("--db-dir", required=True,
                    help="Hoplite DB directory for this month")
    ap.add_argument("--classifier", required=True,
                    help="Trained classifier .pt file")
    ap.add_argument("--output-csv", required=True,
                    help="Filtered detections CSV")
    ap.add_argument("--ratio-threshold", type=float, default=1.0,
                    help="Min orca/humpback logit ratio to keep detection (default 1.0)")
    ap.add_argument("--target-label", default="orca_call",
                    help="Which detection class to filter (default: orca_call)")
    args = ap.parse_args()

    from perch_hoplite.db import sqlite_usearch_impl

    # ── Load data ─────────────────────────────────────────────────────
    print(f"Loading detections: {Path(args.detections_csv).name}")
    df = pd.read_csv(args.detections_csv)
    orca_df = df[df['label'] == args.target_label].copy()
    other_df = df[df['label'] != args.target_label].copy()
    print(f"  {len(orca_df):,} {args.target_label} detections to evaluate")
    print(f"  {len(other_df):,} other detections (passed through unchanged)")

    print(f"Loading classifier: {Path(args.classifier).name}")
    beta, beta_bias, classes = load_classifier(args.classifier)
    orca_idx  = classes.index('orca_call')
    hump_idx  = classes.index('humpback_song')
    print(f"  Classes: {classes}")
    print(f"  orca_call idx={orca_idx}, humpback_song idx={hump_idx}")

    print(f"Opening DB: {Path(args.db_dir).name}")
    db = sqlite_usearch_impl.SQLiteUSearchDB.create(args.db_dir, readonly=True)

    # ── Build window index for relevant recordings ────────────────────
    filenames = set(orca_df['filename'].unique())
    print(f"  Indexing {len(filenames):,} recordings...")
    by_file = build_window_index(args.db_dir, filenames)

    # ── Evaluate each orca detection ──────────────────────────────────
    print(f"\nEvaluating context for {len(orca_df):,} detections "
          f"(ratio threshold={args.ratio_threshold})...")

    kept = []
    suppressed = []
    ratios = []

    for i, (_, row) in enumerate(orca_df.iterrows()):
        context_ids = get_context_window_ids(
            by_file, row['filename'], row['window_start'])

        if not context_ids:
            # No context — keep by default
            kept.append(row)
            ratios.append(np.nan)
            continue

        # Fetch context embeddings and apply classifier
        embs   = db.get_embeddings_batch(context_ids).astype(np.float32)
        logits = apply_classifier(embs, beta, beta_bias)  # [K, 5]

        mean_orca = logits[:, orca_idx].mean()
        mean_hump = logits[:, hump_idx].mean()

        # Ratio: orca / humpback (add small epsilon to avoid /0)
        ratio = mean_orca / (mean_hump + 1e-6)
        ratios.append(ratio)

        if ratio >= args.ratio_threshold:
            kept.append(row)
        else:
            suppressed.append(row)

        if (i + 1) % 50 == 0:
            print(f"  {i+1:,}/{len(orca_df):,} evaluated — "
                  f"kept {len(kept)}, suppressed {len(suppressed)}")

    # ── Summary ───────────────────────────────────────────────────────
    ratios_arr = np.array([r for r in ratios if not np.isnan(r)])
    print(f"\n{'='*60}")
    print(f"Context filter results — {args.target_label}")
    print(f"  Input:      {len(orca_df):,} detections")
    print(f"  Kept:       {len(kept):,} ({100*len(kept)/len(orca_df):.1f}%)")
    print(f"  Suppressed: {len(suppressed):,} ({100*len(suppressed)/len(orca_df):.1f}%)")
    print(f"  Ratio stats: min={ratios_arr.min():.3f}  "
          f"median={np.median(ratios_arr):.3f}  "
          f"max={ratios_arr.max():.3f}")
    print(f"{'='*60}")

    # ── Write output CSV ──────────────────────────────────────────────
    kept_df = pd.DataFrame(kept) if kept else pd.DataFrame(columns=df.columns)
    out_df  = pd.concat([other_df, kept_df], ignore_index=True)
    out_df.sort_values(['filename', 'window_start'], inplace=True)
    out_df.to_csv(args.output_csv, index=False)
    print(f"\nOutput: {args.output_csv}")
    print(f"  Total detections: {len(out_df):,} "
          f"(was {len(df):,}, removed {len(df)-len(out_df):,})")

    # Print kept orca by day
    if kept:
        kept_df2 = pd.DataFrame(kept)
        kept_df2['day'] = kept_df2['filename'].str.extract(r'MARS_(\d{8})')
        day_counts = kept_df2.groupby('day').size().sort_values(ascending=False)
        print(f"\nKept orca detections by day:")
        for day, n in day_counts.head(10).items():
            print(f"  {day}: {n}")


# ── TODO: Option A — output all 5 logits per window in inference ──────────
# Modify src/infer.py to write one row per window with columns:
#   idx, filename, window_start, window_end,
#   logit_orca_call, logit_humpback_song, logit_dolphin_call,
#   logit_other, logit_ship_noise, top_label, top_logit
# This would allow context filtering without re-querying the DB,
# and would enable richer post-processing (e.g. temporal smoothing).
# Add --output-format full|compact flag to phase2_classify.py infer.
# ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
