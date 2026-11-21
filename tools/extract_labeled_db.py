#!/usr/bin/env python3
"""
extract_labeled_db.py

Builds one training database for a fresh v11:
  - every positive labeled window (label_type 1) and its embedding, from the campaign databases
    and the sperm-whale clip database;
  - 5,000 weak negatives (label_type 0) from unlabeled campaign windows that scored below 0.0 on v10.
    The v10 inference CSVs hold only windows scored at or above 0.0, so an unlabeled window missing
    from a month's v10 CSV is low-scoring background. Samples are split across months in proportion
    to each month's candidates.

Left out on purpose:
  - MARS_20180401_20180430 and MARS_20180501_20180531: the poster's validation and test months (frozen)
  - any database named "combined*": an older merged copy whose rows duplicate the monthly databases
  - label_type 2 (UNCERTAIN): not used for training

Embeddings come from each database's usearch.index, keyed by window id. The output index uses the same
dimension, metric and dtype as the sources, which must agree.

Modes:
  --dry-run          count positives and weak-negative candidates per database; write nothing
  --output DIR       build the training database in DIR (must not already contain hoplite.sqlite)
"""

import argparse
import glob
import json
import os
import random
import re
import struct
import sqlite3
import sys
from collections import Counter

DB_ROOT = "/mnt/PAM_Analysis/perch-hoplite/db"
RESULTS = "/mnt/PAM_Analysis/perch-hoplite/results"
EXCLUDE_SUBSTRINGS = ["20180401_20180430", "20180501_20180531", "combined"]
POSITIVE, NEGATIVE = 1, 0
WEAK_NEG_TOTAL = 5000
SEED = 42
STEM_RE = re.compile(r"(MARS_\d{8}_\d{6})")
YM_RE = re.compile(r"MARS_(\d{4})(\d{2})\d{2}_")


def source_dbs(db_root):
    paths = sorted(glob.glob(os.path.join(db_root, "MARS_*_32kHz_norm")))
    paths += sorted(glob.glob(os.path.join(db_root, "SPERM_*_32kHz_norm")))
    keep, skipped = [], []
    for p in paths:
        name = os.path.basename(p)
        if any(x in name for x in EXCLUDE_SUBSTRINGS):
            skipped.append(name)
        elif os.path.exists(os.path.join(p, "hoplite.sqlite")):
            keep.append(p)
    return keep, skipped


def read_meta(con, key):
    row = con.execute("SELECT value FROM hoplite_metadata WHERE key=?", (key,)).fetchone()
    return json.loads(row[0]) if row else None


def first_offset(blob):
    return struct.unpack("<d", bytes(blob[:8]))[0] if blob else 0.0


def positives_rows(con):
    return con.execute(
        """SELECT w.id, r.id, r.filename, r.datetime, r.deployment_id,
                  a.offsets, a.label, a.provenance,
                  d.name, d.project, d.latitude, d.longitude
           FROM annotations a
           JOIN recordings r ON r.id = a.recording_id
           JOIN windows w ON w.recording_id = a.recording_id AND w.offsets = a.offsets
           LEFT JOIN deployments d ON d.id = r.deployment_id
           WHERE a.label_type = ?""", (POSITIVE,)).fetchall()


def v10_keys(db_dir, results_dir):
    """(stem, start) windows present in the month's v10 CSV, or None if no CSV exists."""
    name = os.path.basename(db_dir)
    m = YM_RE.search(name)
    if not m:
        return None
    files = sorted(glob.glob(os.path.join(results_dir, f"MARS_{m.group(1)}{m.group(2)}*_v10_orcaval.csv")))
    if not files:
        return None
    import csv
    keys = set()
    with open(files[0], newline="") as f:
        for r in csv.DictReader(f):
            mm = STEM_RE.search(r["filename"])
            if mm:
                keys.add((mm.group(1), round(float(r["window_start"]), 1)))
    return keys


def weak_candidates(db_dir, results_dir):
    """Unlabeled windows absent from the month's v10 CSV. Returns (window_id, stem, offset) tuples."""
    keys = v10_keys(db_dir, results_dir)
    if keys is None:
        return None
    con = sqlite3.connect(f"file:{db_dir}/hoplite.sqlite?mode=ro", uri=True)
    annotated = set()
    for fn, blob in con.execute("SELECT r.filename, a.offsets FROM annotations a JOIN recordings r ON r.id = a.recording_id"):
        mm = STEM_RE.search(fn)
        if mm:
            annotated.add((mm.group(1), round(first_offset(blob), 1)))
    out = []
    for wid, fn, blob in con.execute("SELECT w.id, r.filename, w.offsets FROM windows w JOIN recordings r ON r.id = w.recording_id"):
        mm = STEM_RE.search(fn)
        if not mm:
            continue
        k = (mm.group(1), round(first_offset(blob), 1))
        if k not in annotated and k not in keys:
            out.append((wid, mm.group(1), k[1]))
    con.close()
    return out


def allocate(counts, total):
    """Split total across databases in proportion to their candidate counts (largest remainder)."""
    grand = sum(counts.values())
    if grand == 0:
        return {k: 0 for k in counts}
    raw = {k: total * v / grand for k, v in counts.items()}
    base = {k: int(x) for k, x in raw.items()}
    rest = total - sum(base.values())
    for k in sorted(raw, key=lambda k: raw[k] - base[k], reverse=True)[:rest]:
        base[k] += 1
    return base


def plan_weak(paths, results_dir, total, seed):
    cands, counts = {}, {}
    for p in paths:
        c = weak_candidates(p, results_dir)
        if c is None:
            continue
        cands[p] = c
        counts[p] = len(c)
    alloc = allocate(counts, total)
    rng = random.Random(seed)
    chosen = {}
    for p, n in alloc.items():
        if n > 0:
            chosen[p] = rng.sample(cands[p], min(n, len(cands[p])))
    return counts, alloc, chosen


def dry_run(paths, results_dir):
    total = Counter()
    for p in paths:
        con = sqlite3.connect(f"file:{p}/hoplite.sqlite?mode=ro", uri=True)
        per = Counter(r[6] for r in positives_rows(con))
        con.close()
        if per:
            total.update(per)
            print(f"{os.path.basename(p):42s} " + "  ".join(f"{k}={v}" for k, v in sorted(per.items())))
    print("TOTAL positives by label:", dict(sorted(total.items())), "| all:", sum(total.values()))
    counts, alloc, _ = plan_weak(paths, results_dir, WEAK_NEG_TOTAL, SEED)
    print(f"weak-negative candidates (unlabeled, v10 < 0.0): {sum(counts.values()):,} in {len(counts)} databases")
    print(f"weak-negative allocation: {sum(alloc.values())} total; per database:")
    for p in sorted(alloc, key=lambda k: -alloc[k]):
        if alloc[p]:
            print(f"   {os.path.basename(p):42s} {alloc[p]:5d} of {counts[p]:,}")


def build(paths, output, results_dir, index_factory):
    import numpy as np

    first = sqlite3.connect(f"file:{paths[0]}/hoplite.sqlite?mode=ro", uri=True)
    ucfg = read_meta(first, "usearch_config")
    mcfg = read_meta(first, "model_config")
    first.close()
    for p in paths[1:]:
        c = sqlite3.connect(f"file:{p}/hoplite.sqlite?mode=ro", uri=True)
        other = read_meta(c, "usearch_config")
        c.close()
        if other["embedding_dim"] != ucfg["embedding_dim"] or other.get("metric_name") != ucfg.get("metric_name"):
            sys.exit(f"config mismatch in {p}: dim/metric differ from {paths[0]}")

    os.makedirs(os.path.join(output, "logs"), exist_ok=True)
    db_file = os.path.join(output, "hoplite.sqlite")
    if os.path.exists(db_file):
        sys.exit(f"{db_file} already exists; choose a new output directory")
    out = sqlite3.connect(db_file)
    out.executescript("""
        PRAGMA journal_mode = WAL;
        PRAGMA foreign_keys = ON;
        CREATE TABLE hoplite_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE deployments (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, project TEXT NOT NULL,
            latitude REAL, longitude REAL, UNIQUE (name, project));
        CREATE TABLE recordings (id INTEGER PRIMARY KEY AUTOINCREMENT, filename TEXT NOT NULL, datetime TEXT,
            deployment_id INTEGER REFERENCES deployments(id) ON DELETE CASCADE,
            UNIQUE (id, deployment_id), UNIQUE (filename, deployment_id));
        CREATE TABLE windows (id INTEGER PRIMARY KEY AUTOINCREMENT,
            recording_id INTEGER NOT NULL REFERENCES recordings(id) ON DELETE CASCADE,
            offsets FLOAT_LIST NOT NULL, UNIQUE (id, recording_id, offsets));
        CREATE TABLE annotations (id INTEGER PRIMARY KEY AUTOINCREMENT,
            recording_id INTEGER NOT NULL REFERENCES recordings(id) ON DELETE CASCADE,
            offsets FLOAT_LIST NOT NULL, label TEXT NOT NULL, label_type INTEGER NOT NULL,
            provenance TEXT NOT NULL, UNIQUE (id, recording_id, offsets));
        CREATE INDEX idx_annotations ON annotations (recording_id, offsets, label, label_type, provenance);
        CREATE INDEX idx_labels ON annotations (label, label_type, provenance);
        CREATE INDEX idx_recordings_deployment_id ON recordings (deployment_id);
        CREATE INDEX idx_windows_recording_id ON windows (recording_id);
    """)
    out.execute("INSERT INTO hoplite_metadata (key,value) VALUES ('usearch_config', ?)", (json.dumps(ucfg),))
    if mcfg is not None:
        out.execute("INSERT INTO hoplite_metadata (key,value) VALUES ('model_config', ?)", (json.dumps(mcfg),))
    counts_meta, alloc, chosen = plan_weak(paths, results_dir, WEAK_NEG_TOTAL, SEED)
    out.execute("INSERT INTO hoplite_metadata (key,value) VALUES ('audio_sources', ?)",
                (json.dumps({"extracted_from": [os.path.basename(p) for p in paths],
                             "positives": "label_type 1 only",
                             "weak_negatives": {"n": WEAK_NEG_TOTAL, "seed": SEED,
                                                "rule": "unlabeled, absent from month v10 CSV (score < 0.0)"},
                             "excluded": EXCLUDE_SUBSTRINGS}),))

    dim = ucfg["embedding_dim"]
    dtype = ucfg.get("dtype", "float16")
    metric = ucfg.get("metric_name", "ip")
    np_dtype = np.float16 if dtype == "float16" else np.float32
    new_index = index_factory(dim, metric, dtype)

    deploy_ids, rec_ids = {}, {}
    next_win = 1
    counts = Counter()

    def insert_window(new_rec, a_off):
        nonlocal next_win
        wid = next_win
        next_win += 1
        out.execute("INSERT INTO windows (id, recording_id, offsets) VALUES (?,?,?)", (wid, new_rec, a_off))
        return wid

    def ensure_recording(fname, dt, dname, dproj, dlat, dlon):
        new_dep = None
        if dname is not None:
            dkey = (dname, dproj)
            if dkey not in deploy_ids:
                out.execute("INSERT OR IGNORE INTO deployments (name, project, latitude, longitude) VALUES (?,?,?,?)",
                            (dname, dproj, dlat, dlon))
                deploy_ids[dkey] = out.execute("SELECT id FROM deployments WHERE name=? AND project=?",
                                               (dname, dproj)).fetchone()[0]
            new_dep = deploy_ids[dkey]
        rkey = (fname, new_dep)
        if rkey not in rec_ids:
            out.execute("INSERT OR IGNORE INTO recordings (filename, datetime, deployment_id) VALUES (?,?,?)",
                        (fname, dt, new_dep))
            rec_ids[rkey] = out.execute("SELECT id FROM recordings WHERE filename=? AND deployment_id IS ?",
                                        (fname, new_dep)).fetchone()[0]
        return rec_ids[rkey]

    for p in paths:
        con = sqlite3.connect(f"file:{p}/hoplite.sqlite?mode=ro", uri=True)
        rows = positives_rows(con)
        con.close()
        weak = chosen.get(p, [])
        if not rows and not weak:
            continue
        idx = index_factory(dim, metric, dtype)
        idx.load(os.path.join(p, "usearch.index"))

        if rows:
            vecs = idx.get([int(r[0]) for r in rows])
            if isinstance(vecs, tuple):
                vecs = np.stack(vecs)
            vecs = np.asarray(vecs).astype(np_dtype)
            for r, v in zip(rows, vecs):
                (wid_src, _, fname, dt, _, a_off, label, prov, dname, dproj, dlat, dlon) = r
                new_rec = ensure_recording(fname, dt, dname, dproj, dlat, dlon)
                new_win = insert_window(new_rec, a_off)
                out.execute("INSERT INTO annotations (recording_id, offsets, label, label_type, provenance) VALUES (?,?,?,?,?)",
                            (new_rec, a_off, label, POSITIVE, prov))
                new_index.add(new_win, v)
                counts[label] += 1

        if weak:
            src_ids = [w[0] for w in weak]
            vecs = idx.get(src_ids)
            if isinstance(vecs, tuple):
                vecs = np.stack(vecs)
            vecs = np.asarray(vecs).astype(np_dtype)
            con = sqlite3.connect(f"file:{p}/hoplite.sqlite?mode=ro", uri=True)
            for (wid_src, stem, start), v in zip(weak, vecs):
                fname, dt, dname, dproj, dlat, dlon = con.execute(
                    """SELECT r.filename, r.datetime, d.name, d.project, d.latitude, d.longitude
                       FROM windows w JOIN recordings r ON r.id = w.recording_id
                       LEFT JOIN deployments d ON d.id = r.deployment_id WHERE w.id = ?""", (wid_src,)).fetchone()
                a_off = struct.pack("<dd", float(start), float(start) + 5.0)   # start and end, as the training code requires
                new_rec = ensure_recording(fname, dt, dname, dproj, dlat, dlon)
                new_win = insert_window(new_rec, a_off)
                out.execute("INSERT INTO annotations (recording_id, offsets, label, label_type, provenance) VALUES (?,?,?,?,?)",
                            (new_rec, a_off, "background", NEGATIVE, "weak_negative_v10_below_0"))
                new_index.add(new_win, v)
                counts["background(weak)"] += 1
            con.close()
        print(f"{os.path.basename(p):42s} positives {len(rows):5d}   weak negatives {len(weak):5d}")

    out.commit()
    out.close()
    new_index.save(os.path.join(output, "usearch.index"))
    print("written:", output)
    print("counts:", dict(sorted(counts.items())), "| total:", sum(counts.values()))
    return sum(counts.values())


def default_index_factory(dim, metric, dtype):
    import usearch.index as ui
    return ui.Index(ndim=dim, metric=metric, dtype=dtype)


def main():
    ap = argparse.ArgumentParser(description="Extract positives and weak negatives into one training database.")
    ap.add_argument("--db-root", default=DB_ROOT)
    ap.add_argument("--results-dir", default=RESULTS)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--output")
    args = ap.parse_args()

    paths, skipped = source_dbs(args.db_root)
    print(f"source databases: {len(paths)}   left out: {skipped}")
    if args.dry_run or not args.output:
        dry_run(paths, args.results_dir)
        return 0
    build(paths, args.output, args.results_dir, default_index_factory)
    return 0


if __name__ == "__main__":
    sys.exit(main())
