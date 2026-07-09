#!/usr/bin/env python3
"""merge_annotations.py
Copy annotations from a source Hoplite DB into a target Hoplite DB.

The source and target DBs must have been built with the same model
(same embedding_dim and audio source structure) but can cover different
date ranges. Only annotations are copied — embeddings stay separate.

Usage:
    python3 merge_annotations.py \
        --source-db /path/to/source/db \
        --target-db /path/to/target/db \
        --dry-run

Example — copy April 1 negatives into April 13 DB:
    python3 merge_annotations.py \
        --source-db /mnt/PAM_Analysis/duane_scratch/perch_hoplite/db/MARS_20180401_20180401_32kHz \
        --target-db /mnt/PAM_Analysis/duane_scratch/perch_hoplite/db/MARS_20180413_20180413_32kHz
"""

import argparse
import sqlite3
import struct
import os
import sys


def get_db_file(db_dir: str) -> str:
    p = os.path.join(db_dir, "hoplite.sqlite")
    if not os.path.isfile(p):
        raise FileNotFoundError(f"hoplite.sqlite not found in {db_dir}")
    return p


def decode_offsets(blob) -> tuple:
    if isinstance(blob, (bytes, bytearray)) and len(blob) >= 16:
        return struct.unpack_from("<dd", blob)
    return (0.0, 5.0)


def encode_offsets(start: float, end: float) -> bytes:
    return struct.pack("<dd", start, end)


def merge_annotations(source_db: str, target_db: str, dry_run: bool = False):
    src = sqlite3.connect(source_db)
    tgt = sqlite3.connect(target_db)

    # Read all annotations from source with their recording filenames
    rows = src.execute("""
        SELECT a.offsets, a.label, a.label_type, a.provenance,
               r.filename, d.name as deployment
        FROM annotations a
        JOIN recordings r ON r.id = a.recording_id
        JOIN deployments d ON d.id = r.deployment_id
    """).fetchall()

    print(f"Found {len(rows)} annotations in source DB.")

    if len(rows) == 0:
        print("Nothing to copy.")
        return

    # Show sample
    for i, row in enumerate(rows[:3]):
        offs = decode_offsets(row[0])
        print(f"  [{i}] {row[4]}  {offs[0]:.1f}-{offs[1]:.1f}s  "
              f"label={row[1]}  type={row[2]}")
    if len(rows) > 3:
        print(f"  ... and {len(rows)-3} more")

    if dry_run:
        print("\nDRY RUN — no changes made.")
        src.close()
        tgt.close()
        return

    # For each annotation, find or create the recording in target DB
    inserted = 0
    skipped = 0

    for offs_blob, label, label_type, provenance, filename, deployment in rows:
        start_s, end_s = decode_offsets(offs_blob)

        # Find recording in target by filename only — deployment names may differ
        # between source and target DBs (e.g. MARS_20180413 vs MARS_20180401_20180430)
        rec_row = tgt.execute(
            "SELECT MIN(id) FROM recordings WHERE filename=?",
            (filename,)
        ).fetchone()
        if rec_row is None:
            # Recording not in target — insert under source deployment name
            dep_row = tgt.execute(
                "SELECT id FROM deployments WHERE name=?", (deployment,)
            ).fetchone()
            if dep_row is None:
                tgt.execute(
                    "INSERT OR IGNORE INTO deployments (name, project) VALUES (?,?)",
                    (deployment, deployment)
                )
                dep_id = tgt.execute(
                    "SELECT id FROM deployments WHERE name=?", (deployment,)
                ).fetchone()[0]
            else:
                dep_id = dep_row[0]
            tgt.execute(
                "INSERT OR IGNORE INTO recordings (filename, deployment_id) VALUES (?,?)",
                (filename, dep_id)
            )
            rec_id = tgt.execute(
                "SELECT MIN(id) FROM recordings WHERE filename=?", (filename,)
            ).fetchone()[0]
        else:
            rec_id = rec_row[0]

        # Insert annotation
        off_enc = encode_offsets(start_s, end_s)
        try:
            tgt.execute("""
                INSERT INTO annotations
                    (recording_id, offsets, label, label_type, provenance)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT DO UPDATE SET
                    label_type=excluded.label_type,
                    provenance=excluded.provenance
            """, (rec_id, off_enc, label, label_type,
                  provenance + "_merged"))
            inserted += 1
        except Exception as e:
            print(f"  Warning: could not insert {filename} {start_s:.1f}s: {e}")
            skipped += 1

    tgt.commit()
    src.close()
    tgt.close()

    print(f"\nDone. Inserted {inserted} annotations ({skipped} skipped).")

    # Show target totals
    tgt2 = sqlite3.connect(target_db)
    pos = tgt2.execute(
        "SELECT label, COUNT(*) FROM annotations WHERE label_type=1 GROUP BY label"
    ).fetchall()
    neg = tgt2.execute(
        "SELECT label, COUNT(*) FROM annotations WHERE label_type=2 GROUP BY label"
    ).fetchall()
    tgt2.close()
    print(f"Target DB totals — positive: {dict(pos)}  negative: {dict(neg)}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source-db", required=True,
                   help="DB directory to copy annotations FROM")
    p.add_argument("--target-db", required=True,
                   help="DB directory to copy annotations INTO")
    p.add_argument("--dry-run", action="store_true",
                   help="Show what would be copied without making changes")
    args = p.parse_args()

    source_db = get_db_file(args.source_db)
    target_db = get_db_file(args.target_db)

    print(f"Source: {source_db}")
    print(f"Target: {target_db}")
    print()

    merge_annotations(source_db, target_db, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
