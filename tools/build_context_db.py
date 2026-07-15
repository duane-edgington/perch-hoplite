#!/usr/bin/env python3
"""tools/build_context_db.py
Build a new Hoplite DB where every embedding is replaced by a
Gaussian-weighted average of its ±CONTEXT_WINDOWS neighbors within
the same recording. Annotations are copied unchanged.

Usage (combined 3-season DB):
    python3 tools/build_context_db.py \
        --source-db /mnt/PAM_Analysis/perch-hoplite/db/MARS_combined_3month_32kHz_norm_v2 \
        --output-db /mnt/PAM_Analysis/perch-hoplite/db/MARS_combined_3month_32kHz_ctx
"""
import argparse
import sqlite3
import struct
import shutil
import numpy as np
from pathlib import Path

SIGMA_WINDOWS   = 1.5   # Gaussian sigma in window units (~7.5 seconds)
CONTEXT_WINDOWS = 3     # ±3 windows = ±15 seconds


def gaussian_weights(T, center, sigma=SIGMA_WINDOWS, ctx=CONTEXT_WINDOWS):
    idx = np.arange(T, dtype=np.float32)
    w = np.exp(-0.5 * ((idx - center) / sigma) ** 2)
    w[np.abs(idx - center) > ctx] = 0.0
    total = w.sum()
    if total > 1e-8:
        w /= total
    else:
        w[:] = 0.0; w[center] = 1.0
    return w


def build_context_db(source_db_path: str, output_db_path: str):
    from perch_hoplite.db import sqlite_usearch_impl
    from usearch.index import Index

    src_path = Path(source_db_path)
    out_path = Path(output_db_path)

    if out_path.exists():
        print(f"Output DB already exists: {out_path} — delete it first.")
        return

    # ── Step 1: Copy SQLite (metadata + annotations) ─────────────────
    print(f"Source: {src_path.name}")
    out_path.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path / "hoplite.sqlite", out_path / "hoplite.sqlite")
    print("  SQLite metadata copied.")

    # ── Step 2: Open source DB for reading embeddings ─────────────────
    src_db  = sqlite_usearch_impl.SQLiteUSearchDB.create(str(src_path), readonly=True)
    emb_dim = src_db.get_embedding_dim()
    total   = src_db.count_embeddings()
    print(f"  {total:,} embeddings, dim={emb_dim}")

    # ── Step 3: Load source USearch index directly ────────────────────
    src_index = Index.restore(str(src_path / "usearch.index"), view=True)
    print(f"  USearch index loaded: {len(src_index):,} vectors")

    # ── Step 4: Create output USearch index ───────────────────────────
    out_index = Index(ndim=emb_dim, metric='cos', dtype='f16')

    # ── Step 5: Process recording by recording ────────────────────────
    con = sqlite3.connect(str(src_path / "hoplite.sqlite"))
    recordings = con.execute(
        "SELECT id FROM recordings ORDER BY id").fetchall()
    print(f"  Processing {len(recordings):,} recordings...")

    processed = 0
    report_every = 500

    for rec_idx, (rec_id,) in enumerate(recordings):
        rows = con.execute(
            "SELECT id, offsets FROM windows WHERE recording_id = ?",
            (rec_id,)).fetchall()
        if not rows:
            continue

        # Sort numerically by start_s
        rows.sort(key=lambda r: struct.unpack_from('<d', r[1])[0]
                  if r[1] and len(r[1]) >= 8 else 0.0)

        window_ids = np.array([r[0] for r in rows], dtype=np.uint64)
        T = len(window_ids)

        # Batch-fetch embeddings: [T, D] float32
        embs = src_db.get_embeddings_batch(window_ids.tolist()).astype(np.float32)

        # Build weight matrix [T, T] and compute context embeddings
        W = np.stack([gaussian_weights(T, i) for i in range(T)])  # [T, T]
        ctx_embs = W @ embs                                        # [T, D]

        # L2-normalise
        norms = np.linalg.norm(ctx_embs, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        ctx_embs = (ctx_embs / norms).astype(np.float16)

        # Add to output index
        out_index.add(window_ids, ctx_embs)

        processed += T
        if (rec_idx + 1) % report_every == 0:
            pct = 100 * processed / total
            print(f"  {rec_idx+1:,}/{len(recordings):,} recordings "
                  f"({processed:,}/{total:,} windows, {pct:.1f}%)")

    con.close()

    # ── Step 6: Save output USearch index ────────────────────────────
    print(f"\nSaving USearch index ({len(out_index):,} vectors)...")
    out_index.save(str(out_path / "usearch.index"))
    print(f"Done — context DB written to: {out_path.name}")
    print(f"\nNext: train v5")
    print(f"  time python3 phase2_classify.py train \\")
    print(f"      --db-dir {out_path} \\")
    print(f"      --classifier-out /mnt/PAM_Analysis/perch-hoplite/models/orca_v5.pt \\")
    print(f"      --num-steps 256 --train-ratio 0.8")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-db", required=True)
    ap.add_argument("--output-db", required=True)
    args = ap.parse_args()
    build_context_db(args.source_db, args.output_db)


if __name__ == "__main__":
    main()
