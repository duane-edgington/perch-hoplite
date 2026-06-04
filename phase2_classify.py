#!/usr/bin/env python3
"""
phase2_classify.py — Perch Hoplite Phase 2: Search, Label, Classify & Inference
=================================================================================
Loads a Hoplite vector database produced by phase1_embed.py, then provides a
suite of sub-commands for the complete agile modeling workflow:

  search      Embed a query audio clip and find nearest neighbours.
  label       Import a CSV of labels (recording_id, offset_s, label, type) into the DB.
  train       Train a linear classifier on the current DB labels.
  review      Run the trained classifier over the DB and write top-scoring results.
  infer       Run full inference over all embeddings and write a results CSV.
  stats       Print DB statistics (label counts, embedding counts).

The GUI for interactive labeling (audio playback + click-to-label) is served as
a Gradio web app via --serve, accessible from any browser on the MBARI network.

Environment
-----------
MBARI NVIDIA DGX SPARC nodes: spark-ae0e (134.89.11.107)
                               spark-0626 (134.89.11.174)
NFS base : /mnt/PAM_Analysis/duane_scratch/perch_hoplite/
Audio    : /mnt/PAM_Archive/<year>/<deployment>/

Path constants are in:
  /mnt/PAM_Analysis/duane_scratch/perch_hoplite/.perch_env
  source that file to avoid typing long paths.

Usage examples
--------------
# 0. Source path constants (optional but convenient):
source /mnt/PAM_Analysis/duane_scratch/perch_hoplite/.perch_env

# 1. Search and launch Gradio labeling GUI:
#    Browser: http://134.89.11.107:7860  (spark-ae0e)
#          or http://134.89.11.174:7860  (spark-0626)
python3 phase2_classify.py search \
    --db-dir /mnt/PAM_Analysis/duane_scratch/perch_hoplite/db/MARS_2018 \
    --query-audio /mnt/PAM_Analysis/duane_scratch/perch_hoplite/queries/cetaceans/orca_call.wav \
    --query-label orca_call \
    --num-results 200 \
    --output-csv /mnt/PAM_Analysis/duane_scratch/perch_hoplite/results/MARS_2018_orca_search.csv \
    --serve --port 7860

# 2. Import labels from Raven Pro / PAMGuard CSV:
#    CSV columns: recording_id, offset_s, end_offset_s, label, label_type
#    label_type values: positive | negative | weak_negative
python3 phase2_classify.py label \
    --db-dir /mnt/PAM_Analysis/duane_scratch/perch_hoplite/db/MARS_2018 \
    --labels-csv /mnt/PAM_Analysis/duane_scratch/perch_hoplite/labels/orca_raven.csv \
    --annotator-id duane

# 3. Train a linear classifier:
python3 phase2_classify.py train \
    --db-dir /mnt/PAM_Analysis/duane_scratch/perch_hoplite/db/MARS_2018 \
    --classifier-out /mnt/PAM_Analysis/duane_scratch/perch_hoplite/models/orca_v1.pt \
    --num-steps 256 --learning-rate 0.001

# 4. Active learning — review classifier results and add more labels:
python3 phase2_classify.py review \
    --db-dir /mnt/PAM_Analysis/duane_scratch/perch_hoplite/db/MARS_2018 \
    --classifier /mnt/PAM_Analysis/duane_scratch/perch_hoplite/models/orca_v1.pt \
    --target-label orca_call \
    --num-results 100 \
    --serve --port 7860

# 5. Full inference — write detections CSV:
python3 phase2_classify.py infer \
    --db-dir /mnt/PAM_Analysis/duane_scratch/perch_hoplite/db/MARS_2018 \
    --classifier /mnt/PAM_Analysis/duane_scratch/perch_hoplite/models/orca_v1.pt \
    --output-csv /mnt/PAM_Analysis/duane_scratch/perch_hoplite/results/MARS_2018_orca_detections.csv \
    --logit-threshold 0.0

# 6. Check DB statistics:
python3 phase2_classify.py stats \
    --db-dir /mnt/PAM_Analysis/duane_scratch/perch_hoplite/db/MARS_2018

GUI
---
Gradio is already installed (v6.15.1). Add --serve --port 7860 to any
search or review command. The terminal will print the URL; open it in
any browser on the MBARI network. Press Ctrl+C to stop the server.

For multi-analyst annotation campaigns, use Label Studio (Docker):
  docker run -d -p 8080:8080 \
      -v /mnt/PAM_Analysis/duane_scratch/perch_hoplite/labelstudio:/label-studio/data \
      heartexlabs/label-studio:latest
  Access at http://134.89.11.107:8080

Known harmless warnings
-----------------------
  "Unable to register cuFFT/cuDNN/cuBLAS factory" — two TF builds
  registering CUDA plugins; GPU works correctly regardless.
  "MessageFactory has no attribute GetPrototype" — protobuf mismatch,
  cosmetic only.
  "NUMA node read from SysFS had negative value" — BIOS limitation,
  TF defaults to node zero correctly.
"""

import argparse
import csv
import json
import logging
import logging.handlers
import os
import sys
import time
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FORMAT = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
LOG_DATE   = "%Y-%m-%d %H:%M:%S"

def _setup_logging(log_dir: Path, verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    root = logging.getLogger()
    root.setLevel(level)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE))
    root.addHandler(ch)

    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "phase2_classify.log"
    fh = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=50 * 1024 * 1024, backupCount=5
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE))
    root.addHandler(fh)
    logging.info("Logging to %s", log_file)

log = logging.getLogger(__name__)


def _get_label_type_enum():
    """Return the LabelType enum from wherever perch-hoplite 1.0.1 exposes it.

    The location changed across versions:
      - perch_hoplite.db.interface          (older builds)
      - perch_hoplite.db.annotations        (1.0.x)
      - perch_hoplite.agile.source_info     (some builds)
    Falls back to a simple namespace object so labeling still works.
    """
    for mod_path, attr in (
        ("perch_hoplite.db.annotations",    "LabelType"),
        ("perch_hoplite.db.interface",       "LabelType"),
        ("perch_hoplite.agile.source_info",  "LabelType"),
        ("perch_hoplite.db.sqlite_usearch_impl", "LabelType"),
    ):
        try:
            import importlib
            mod = importlib.import_module(mod_path)
            lt = getattr(mod, attr, None)
            if lt is not None:
                return lt
        except ImportError:
            continue

    # Ultimate fallback: plain namespace with integer values
    # (perch-hoplite uses these ints internally in SQLite)
    log.warning(
        "Could not import LabelType from perch_hoplite — "
        "using integer fallback (0=positive, 1=negative, 2=weak_negative)."
    )
    class _LabelType:
        POSITIVE      = 1
        NEGATIVE      = 2
        WEAK_NEGATIVE = 3
        UNCERTAIN     = 3
    return _LabelType

# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------

def _require_perch():
    try:
        from perch_hoplite.agile import (
            audio_loader, classifier, classifier_data,
            embedding_display, source_info,
        )
        from perch_hoplite.db import (
            brutalism, interface, score_functions,
            search_results, sqlite_usearch_impl,
        )
        from perch_hoplite.zoo import model_configs
        import numpy as np
    except ImportError as exc:
        log.error(
            "perch-hoplite not installed. "
            "Run: pip install git+https://github.com/google-research/perch-hoplite.git\n"
            "Error: %s", exc,
        )
        sys.exit(1)
    return (audio_loader, classifier, classifier_data,
            embedding_display, source_info,
            brutalism, interface, score_functions,
            search_results, sqlite_usearch_impl,
            model_configs, np)


def _require_gradio():
    try:
        import gradio as gr
        return gr
    except ImportError:
        log.error(
            "Gradio is not installed. Install with: pip install gradio\n"
            "Or use --output-csv to write results without the GUI."
        )
        sys.exit(1)


def _require_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless backend
        import matplotlib.pyplot as plt
        import numpy as np
        return plt, np
    except ImportError as exc:
        log.error("matplotlib not available: %s", exc)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(
        prog="phase2_classify.py",
        description="Perch Hoplite Phase 2 — Search, Label, Train, Infer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    top.add_argument("--verbose", "-v", action="store_true", help="DEBUG logging.")
    top.add_argument("--log-dir", default=None, help="Directory for log files.")

    sub = top.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # ---- Common arguments ----
    def add_db(p):
        p.add_argument("--db-dir", "-d", required=True,
                       help="Path to Hoplite database directory.")

    def add_serve(p):
        p.add_argument("--serve", action="store_true", default=False,
                       help="Launch the Gradio labeling GUI.")
        p.add_argument("--port", type=int, default=7860,
                       help="Port for Gradio server (default: 7860).")
        p.add_argument("--host", default="0.0.0.0",
                       help="Bind address for Gradio server (default: 0.0.0.0).")
        p.add_argument("--share", action="store_true", default=False,
                       help="Create a public Gradio share link (requires internet).")

    # ---- search ----
    ps = sub.add_parser("search", help="Embed a query clip and find nearest neighbours.")
    add_db(ps)
    ps.add_argument("--query-audio", "-q", required=True,
                    help="Path or GCS URI to query audio clip (.wav/.flac).")
    ps.add_argument("--query-label", "-l", required=True,
                    help="Label string for this query class (e.g. 'orca_call').")
    ps.add_argument("--offset-s", type=float, default=0.0,
                    help="Start offset within the query audio (seconds, default: 0).")
    ps.add_argument("--window-s", type=float, default=5.0,
                    help="Window duration for query audio (seconds, default: 5).")
    ps.add_argument("--num-results", type=int, default=100,
                    help="Number of nearest neighbours to retrieve (default: 100).")
    ps.add_argument("--score-fn", choices=["dot", "cos", "neg_euclidean"], default="dot",
                    help="Similarity function (default: dot).")
    ps.add_argument("--exact", action="store_true", default=True,
                    help="Use exact brute-force search (default: True).")
    ps.add_argument("--approx", dest="exact", action="store_false",
                    help="Use approximate nearest-neighbour search (faster, less accurate).")
    ps.add_argument("--target-score", type=float, default=None,
                    help="If set, search for examples near this score (margin sampling).")
    ps.add_argument("--sample-rate-hz", type=int, default=None,
                    help="Audio loader sample rate override.")
    ps.add_argument("--output-csv", default=None,
                    help="Write search results to this CSV (recording_id, offset_s, score).")
    ps.add_argument("--plot-scores", default=None,
                    help="Save score histogram PNG to this path.")
    ps.add_argument("--annotator-id", default="analyst",
                    help="Annotator identifier attached to saved labels.")
    add_serve(ps)

    # ---- label ----
    pl = sub.add_parser("label", help="Import labels from a CSV into the DB.")
    add_db(pl)
    pl.add_argument("--labels-csv", required=True,
                    help=(
                        "CSV file with columns: "
                        "recording_id, offset_s, end_offset_s, label, label_type "
                        "(label_type: positive|negative|weak_negative)."
                    ))
    pl.add_argument("--annotator-id", default="analyst",
                    help="Annotator ID to attach to imported labels.")
    pl.add_argument("--dry-run", action="store_true",
                    help="Validate CSV without writing to DB.")

    # ---- train ----
    pt = sub.add_parser("train", help="Train a linear classifier on DB labels.")
    add_db(pt)
    pt.add_argument("--classifier-out", "-o", required=True,
                    help="Output path for the trained classifier (.pt file).")
    pt.add_argument("--target-labels", nargs="+", default=None,
                    help="Restrict training to these label classes (default: all).")
    pt.add_argument("--learning-rate", type=float, default=1e-3)
    pt.add_argument("--num-steps", type=int, default=128)
    pt.add_argument("--batch-size", type=int, default=128)
    pt.add_argument("--weak-neg-batch-size", type=int, default=128)
    pt.add_argument("--weak-neg-weight", type=float, default=0.05)
    pt.add_argument("--l2-mu", type=float, default=0.0)
    pt.add_argument("--train-ratio", type=float, default=0.9)
    pt.add_argument("--loss-fn", choices=["bce", "hinge"], default="bce")
    pt.add_argument("--seed", type=int, default=42)

    # ---- review ----
    pr = sub.add_parser(
        "review",
        help="Run classifier over DB, display top results for active-learning labeling.",
    )
    add_db(pr)
    pr.add_argument("--classifier", "-c", required=True,
                    help="Path to trained classifier .pt file.")
    pr.add_argument("--target-label", required=True,
                    help="Label class to review.")
    pr.add_argument("--num-results", type=int, default=100)
    pr.add_argument("--sample-size", type=int, default=10_000,
                    help="Randomly sample this many DB entries to search over.")
    pr.add_argument("--margin-target-score", type=float, default=None,
                    help="If set, use margin sampling around this logit.")
    pr.add_argument("--output-csv", default=None,
                    help="Write review results to this CSV.")
    pr.add_argument("--plot-scores", default=None,
                    help="Save score histogram PNG.")
    pr.add_argument("--annotator-id", default="analyst")
    pr.add_argument("--sample-rate-hz", type=int, default=None)
    pr.add_argument("--audio-dir", default=None,
        help="Override audio base path stored in DB (needed when DB was built "
             "on a different machine e.g. Colab). "
             "Example: /mnt/PAM_Analysis/GoogleMultiSpeciesWhaleModel2/resampled_32kHz/2018/04")
    add_serve(pr)

    # ---- infer ----
    pi = sub.add_parser("infer", help="Run inference over all embeddings and save CSV.")
    add_db(pi)
    pi.add_argument("--classifier", "-c", required=True,
                    help="Path to trained classifier .pt file.")
    pi.add_argument("--output-csv", "-o", required=True,
                    help="Output CSV path for detections.")
    pi.add_argument("--logit-threshold", type=float, default=0.0,
                    help="Minimum logit to include in output (default: 0.0).")
    pi.add_argument("--labels", nargs="+", default=None,
                    help="Restrict inference to these label classes (default: all).")
    pi.add_argument("--plot-distribution", default=None,
                    help="Save logit distribution histogram PNG.")

    # ---- stats ----
    pst = sub.add_parser("stats", help="Print DB statistics.")
    add_db(pst)

    return top


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _patch_usearch_get_embeddings_batch(db) -> None:
    """Monkey-patch get_embeddings_batch to handle USearch API version differences.

    perch-hoplite 1.0.1 was written against an older USearch where index.get(keys)
    returned a single np.ndarray.  Newer USearch (>=2.9) changed get() to always
    return a tuple of arrays (one per key), even for a single key.  The mismatch
    causes a RuntimeError inside threaded_brute_search.

    This patch wraps get_embeddings_batch so it works with both API versions.
    """
    import numpy as np
    import types

    original_fn = db.get_embeddings_batch.__func__  # unbound method

    def patched_get_embeddings_batch(self, emb_ids):
        # Try the original implementation first
        try:
            result = original_fn(self, emb_ids)
            # If it returned without error it's the old API — pass through
            return result
        except RuntimeError as exc:
            if "Expected np.ndarray" not in str(exc):
                raise  # unrelated error, re-raise

        # New USearch API: index.get(keys) returns a tuple of 1-D arrays.
        # Re-implement the batch fetch manually.
        keys = [int(eid) for eid in emb_ids]
        results = self.ui.get(keys)           # returns tuple of arrays
        if isinstance(results, np.ndarray):
            # Shouldn't happen here (old API) but be safe
            return results
        # Stack the tuple of 1-D vectors into a 2-D array
        arrays = []
        for arr in results:
            if isinstance(arr, np.ndarray):
                arrays.append(arr)
            else:
                raise RuntimeError(
                    f"Unexpected type in USearch get() result: {type(arr)}"
                )
        return np.stack(arrays, axis=0)

    # Bind the patched method to the instance
    db.get_embeddings_batch = types.MethodType(patched_get_embeddings_batch, db)
    log.debug("Applied USearch get_embeddings_batch compatibility patch.")


def load_db(db_dir: str):
    from perch_hoplite.db import sqlite_usearch_impl
    db_dir = str(db_dir)
    log.info("Loading database from %s", db_dir)
    db = sqlite_usearch_impl.SQLiteUSearchDB.create(db_dir)
    count = db.count_embeddings()
    log.info("Database loaded — %d embeddings", count)
    if count == 0:
        log.warning("Database contains zero embeddings. Run phase1_embed.py first.")
    _patch_usearch_get_embeddings_batch(db)
    return db


def _get_source(db, window_id):
    """Retrieve embedding source metadata (filename + offsets) for a window.

    Tries the known perch-hoplite method names in order, then falls back to
    a direct SQLite query using the db_config path stored on the DB object.
    """
    # Try every known method name
    for method_name in (
        "get_source_by_id",       # perch-hoplite 1.0.x
        "get_embedding_source",   # older builds
        "get_window_source",
        "get_source",
    ):
        method = getattr(db, method_name, None)
        if method is not None:
            try:
                return method(window_id)
            except Exception:
                pass

    # Last resort: direct SQLite query.
    # SQLiteUSearchDB stores its path in db_config.db_path
    import sqlite3 as _sqlite3, os as _os
    db_path_str = None
    # Try db_config path (the real location in SQLiteUSearchDB)
    db_cfg = getattr(db, "db_config", None)
    if db_cfg is not None:
        db_path_str = str(getattr(db_cfg, "db_path", None) or "")
    # Fallback: other attribute names
    if not db_path_str:
        for attr in ("db_path", "_db_path", "path", "sqlite_path"):
            v = getattr(db, attr, None)
            if v:
                db_path_str = str(v)
                break

    if not db_path_str:
        available = [a for a in dir(db)
                     if not a.startswith("__")
                     and ("path" in a.lower() or "dir" in a.lower()
                          or "source" in a.lower() or "config" in a.lower())]
        raise RuntimeError(
            f"Cannot find SQLite path on {type(db).__name__}. "
            f"Relevant attrs: {available}"
        )

    # db_path_str may be the directory; find the sqlite file inside it
    sqlite_file = db_path_str
    if _os.path.isdir(sqlite_file):
        for fname in ("hoplite.sqlite", "hoplite.db", "db.sqlite"):
            candidate = _os.path.join(sqlite_file, fname)
            if _os.path.isfile(candidate):
                sqlite_file = candidate
                break

    con = _sqlite3.connect(sqlite_file)
    # Schema: windows(id, recording_id, offsets)
    #         recordings(id, filename, datetime, deployment_id)
    #         deployments(id, name, project, ...)
    # offsets is stored as a FLOAT_LIST blob — two little-endian float64s.
    row = con.execute("""
        SELECT r.filename, d.project, w.offsets
        FROM windows w
        JOIN recordings r ON r.id = w.recording_id
        LEFT JOIN deployments d ON d.id = r.deployment_id
        WHERE w.id = ?
    """, (int(window_id),)).fetchone()
    con.close()

    if row is None:
        raise KeyError(f"window_id {window_id} not found in DB at {sqlite_file}")

    filename, project, offsets_blob = row

    # Decode FLOAT_LIST blob: two little-endian float64 values (start_s, end_s)
    import struct as _struct
    if isinstance(offsets_blob, (bytes, bytearray)) and len(offsets_blob) >= 16:
        start_s, end_s = _struct.unpack_from("<dd", offsets_blob, 0)
    elif isinstance(offsets_blob, (bytes, bytearray)) and len(offsets_blob) >= 8:
        start_s = _struct.unpack_from("<d", offsets_blob, 0)[0]
        end_s = start_s + 5.0
    else:
        # Fallback: offsets_blob might be a string like "[0.0, 5.0]"
        try:
            import json as _json
            vals = _json.loads(str(offsets_blob))
            start_s, end_s = float(vals[0]), float(vals[1])
        except Exception:
            start_s, end_s = 0.0, 5.0

    class _EmbSource:
        source_id    = filename
        dataset      = project or ""
        offsets      = (start_s, end_s)
        recording_id = None
    return _EmbSource()


def load_model_from_db(db):
    from perch_hoplite.zoo import model_configs
    from perch_hoplite.agile import source_info
    db_model_config = db.get_metadata("model_config")
    embed_config = db.get_metadata("audio_sources")
    model_class = model_configs.get_model_class(db_model_config.model_key)
    embedding_model = model_class.from_config(db_model_config.model_config)
    audio_sources = source_info.AudioSources.from_config_dict(embed_config)
    log.info("Loaded embedding model: %s", db_model_config.model_key)
    return embedding_model, audio_sources


def make_audio_loader(db, embedding_model, audio_sources, sample_rate_hz=None):
    """Return (loader_fn, sample_rate_hz, window_size_s) using the real
    perch-hoplite audio_loader.make_filepath_loader() API.

    The returned loader normalizes audio to -3 dBFS peak so clips are
    audible in the browser (MARS 32 kHz files have very low amplitude).
    """
    import numpy as _np
    from perch_hoplite.agile import audio_loader as _al

    if sample_rate_hz is None:
        sample_rate_hz = embedding_model.sample_rate
    window_size_s = getattr(embedding_model, "window_size_s", 5.0)

    _raw_loader = _al.make_filepath_loader(
        audio_sources=audio_sources,
        window_size_s=window_size_s,
        sample_rate_hz=sample_rate_hz,
    )

    _target_peak = 10 ** (-3.0 / 20)   # -3 dBFS ≈ 0.708

    def loader(window_source):
        audio, sr = _raw_loader(window_source)
        # Normalize to -3 dBFS
        peak = _np.abs(audio).max()
        if peak > 1e-6:
            audio = audio * (_target_peak / peak)
        return audio, sr

    return loader, sample_rate_hz, window_size_s


def _format_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}h {m:02d}m {s:05.2f}s"


# ---------------------------------------------------------------------------
# Sub-command: stats
# ---------------------------------------------------------------------------

def cmd_stats(args) -> int:
    db = load_db(args.db_dir)
    from perch_hoplite.db import interface as iface
    from ml_collections import config_dict
    projects = db.get_all_projects()
    total = db.count_embeddings()
    log.info("=" * 60)
    log.info("DATABASE: %s", args.db_dir)
    log.info("Total embeddings : %d", total)
    for proj in projects:
        ids = db.match_window_ids(
            deployments_filter=config_dict.create(eq=dict(project=proj))
        )
        log.info("  Project %-36s  %d embeddings", proj, len(ids))
    ann_count = len(db.get_all_annotations())
    log.info("Total annotations: %d", ann_count)
    if ann_count > 0:
        try:
            pos = db.count_each_label(label_type=interface.LabelType.POSITIVE)
            neg = db.count_each_label(label_type=interface.LabelType.NEGATIVE)
            log.info("Positive labels: %s", dict(pos))
            log.info("Negative labels: %s", dict(neg))
        except Exception as exc:
            log.debug("Label count error: %s", exc)
    log.info("=" * 60)
    return 0


# ---------------------------------------------------------------------------
# Sub-command: label (CSV import)
# ---------------------------------------------------------------------------

LABEL_TYPE_MAP = {
    "positive": None,  # resolved at runtime from interface module
    "negative": None,
    "weak_negative": None,
}

def cmd_label(args) -> int:
    db = load_db(args.db_dir)
    from perch_hoplite.db import interface as iface

    label_type_map = {
        "positive": interface.LabelType.POSITIVE,
        "negative": interface.LabelType.NEGATIVE,
        "weak_negative": getattr(interface.LabelType, "WEAK_NEGATIVE",
                         interface.LabelType.NEGATIVE),
    }

    csv_path = Path(args.labels_csv)
    if not csv_path.exists():
        log.error("Labels CSV not found: %s", csv_path)
        return 1

    inserted = 0
    skipped = 0
    errors = 0

    # Build filename -> integer recording_id map from DB embedding sources.
    # insert_annotation requires the integer primary key, not a filename string.
    # We scan all window IDs and collect (filename, recording_id) pairs.
    # This is the only reliable way in perch-hoplite 1.0.1 — there is no
    # get_all_recordings() method on the SQLiteUSearchDB.
    log.info("Building filename -> recording_id lookup from DB (scanning embeddings)...")
    filename_to_rec_id: dict[str, int] = {}
    try:
        all_ids = db.match_window_ids(limit=None)
        for wid in all_ids:
            src = _get_source(db, wid)
            fname = getattr(src, "source_id", None) or ""
            rec_id = getattr(src, "recording_id", None)
            if fname and rec_id is not None:
                filename_to_rec_id[fname] = int(rec_id)
                # Also index by stem (without extension) for flexible matching
                stem = fname.rsplit(".", 1)[0] if "." in fname else fname
                filename_to_rec_id[stem] = int(rec_id)
        log.info("  Found %d unique recordings in DB.", len({v for v in filename_to_rec_id.values()}))
    except Exception as exc:
        log.warning("Could not build filename lookup: %s — will try direct int cast.", exc)

    log.info("Reading labels from %s", csv_path)
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_cols = {"recording_id", "offset_s", "label", "label_type"}
        if not required_cols.issubset(set(reader.fieldnames or [])):
            log.error(
                "CSV missing required columns. Expected: %s  Got: %s",
                required_cols, reader.fieldnames,
            )
            return 1

        for i, row in enumerate(reader):
            try:
                lt_str = row["label_type"].strip().lower()
                lt = label_type_map.get(lt_str)
                if lt is None:
                    log.warning("Row %d: unknown label_type '%s', skipping.", i, lt_str)
                    skipped += 1
                    continue

                csv_filename = row["recording_id"].strip()
                # Resolve to integer recording_id
                rec_int_id = filename_to_rec_id.get(csv_filename)
                if rec_int_id is None:
                    stem = csv_filename.rsplit(".", 1)[0] if "." in csv_filename else csv_filename
                    rec_int_id = filename_to_rec_id.get(stem)
                if rec_int_id is None:
                    # File not in this DB — expected when label CSV covers more
                    # dates than the current DB (e.g. full-month labels on a
                    # single-day DB).
                    skipped += 1
                    continue

                offset_s = float(row["offset_s"])
                end_offset_s = float(row.get("end_offset_s") or offset_s + 5.0)
                offsets = (offset_s, end_offset_s)

                if args.dry_run:
                    log.debug(
                        "DRY RUN row %d: %s (rec_id=%d) offset=%.2f label=%s type=%s",
                        i, csv_filename, rec_int_id, offset_s, row["label"], lt_str,
                    )
                    inserted += 1
                    continue

                db.insert_annotation(
                    recording_id=rec_int_id,
                    offsets=offsets,
                    label=row["label"].strip(),
                    label_type=lt,
                    provenance=f"csv_import:{args.annotator_id}",
                    handle_duplicates="update",
                )
                inserted += 1

            except Exception as exc:
                log.warning("Row %d error: %s  Row data: %s", i, exc, row)
                errors += 1

    action = "DRY RUN — would insert" if args.dry_run else "Inserted"
    log.info("%s %d labels (%d skipped, %d errors).", action, inserted, skipped, errors)
    return 0


# ---------------------------------------------------------------------------
# Sub-command: search
# ---------------------------------------------------------------------------

def _run_search(db, embedding_model, args_query_audio, args_offset_s,
                args_window_s, args_sample_rate_hz, args_num_results,
                args_score_fn, args_exact, args_target_score, np):
    """Core search logic; returns (results, all_scores, sample_rate_hz, window_size_s)."""
    from perch_hoplite.db import brutalism, score_functions, search_results
    from perch_hoplite.agile import embedding_display

    sr = args_sample_rate_hz or embedding_model.sample_rate
    window_size_s = getattr(embedding_model, "window_size_s", 5.0)

    log.info("Loading query audio: %s", args_query_audio)
    query_display = embedding_display.QueryDisplay(
        uri=args_query_audio,
        offset_s=args_offset_s,
        window_size_s=args_window_s or window_size_s,
        sample_rate_hz=sr,
    )

    log.info("Embedding query audio...")
    audio_window = query_display.get_audio_window()
    query_embedding = embedding_model.embed(audio_window).embeddings[0, 0]
    log.info("Query embedding shape: %s", query_embedding.shape)

    score_fn = score_functions_mod.get_score_fn(args_score_fn, target_score=args_target_score)

    log.info(
        "Searching DB (exact=%s, num_results=%d, score_fn=%s)...",
        args_exact, args_num_results, args_score_fn,
    )
    t0 = time.monotonic()
    if args_exact:
        results_obj, all_scores = brutalism.threaded_brute_search(
            db, query_embedding, args_num_results, score_fn=score_fn
        )
    else:
        ann_matches = db.ui.search(query_embedding, count=args_num_results)
        results_obj = search_results.TopKSearchResults(top_k=args_num_results)
        for k, dist in zip(ann_matches.keys, ann_matches.distances):
            results_obj.update(search_results.SearchResult(k, dist))
        all_scores = np.array([r.sort_score for r in results_obj.search_results])

    elapsed = time.monotonic() - t0
    log.info(
        "Search complete in %.2fs — %d results, score range [%.4f, %.4f]",
        elapsed,
        len(results_obj.search_results),
        float(all_scores.min()) if len(all_scores) else 0,
        float(all_scores.max()) if len(all_scores) else 0,
    )
    return results_obj, all_scores, sr, window_size_s


def _write_search_csv(results_obj, db, output_csv: str) -> None:
    """Write search results (recording_id, offset_s, score) to CSV."""
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["window_id", "recording_id", "offset_s", "end_offset_s", "score"])
        for r in results_obj.search_results:
            wid = r.window_id
            source = _get_source(db, wid)
            writer.writerow([
                int(wid),
                source.source_id if hasattr(source, "source_id") else str(source),
                getattr(source, "offsets", (None, None))[0],
                getattr(source, "offsets", (None, None))[1],
                f"{r.sort_score:.6f}",
            ])
            count += 1
    log.info("Wrote %d search results to %s", count, out_path)


def _save_histogram(all_scores, hit_scores, output_path: str, title: str) -> None:
    """Save a score distribution histogram to a PNG file."""
    plt, np = _require_matplotlib()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(all_scores, bins=100, color="#1a6fa0", alpha=0.7, label="All scores")
    ax.scatter(
        hit_scores, np.zeros_like(hit_scores),
        marker="|", color="red", alpha=0.7, s=200, label="Top hits",
    )
    ax.set_title(title)
    ax.set_xlabel("Score")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=150)
    plt.close(fig)
    log.info("Score histogram saved to %s", out)


def cmd_search(args) -> int:
    (audio_loader_mod, classifier_mod, classifier_data_mod,
     embedding_display_mod, source_info_mod,
     brutalism_mod, interface_mod, score_functions_mod,
     search_results_mod, sqlite_usearch_impl_mod,
     model_configs_mod, np) = _require_perch()

    db = load_db(args.db_dir)
    embedding_model, audio_sources = load_model_from_db(db)

    if getattr(args, "audio_dir", None):
        from perch_hoplite.agile import source_info as _si
        patched_globs = []
        _globs = getattr(audio_sources, 'audio_globs', getattr(audio_sources, 'audio_sources', []))
        for ag in _globs:
            patched = _si.AudioSourceConfig(
                dataset_name=ag.dataset_name,
                base_path=args.audio_dir,
                file_glob=ag.file_glob,
                min_audio_len_s=ag.min_audio_len_s,
                target_sample_rate_hz=ag.target_sample_rate_hz,
                shard_len_s=ag.shard_len_s,
            )
            patched_globs.append(patched)
        audio_sources = _si.AudioSources(tuple(patched_globs))
        log.info("Audio base path overridden to: %s", args.audio_dir)

    audio_filepath_loader, sr, window_size_s = make_audio_loader(
        db, embedding_model, audio_sources, args.sample_rate_hz
    )

    results_obj, all_scores, sr, window_size_s = _run_search(
        db, embedding_model,
        args.query_audio, args.offset_s, args.window_s,
        sr, args.num_results, args.score_fn,
        args.exact, args.target_score, np,
    )

    if args.output_csv:
        _write_search_csv(results_obj, db, args.output_csv)

    if args.plot_scores:
        hit_scores = [r.sort_score for r in results_obj.search_results]
        _save_histogram(all_scores, hit_scores, args.plot_scores,
                        f"Search results — {args.query_label}")

    if args.serve:
        _launch_labeling_gui(
            db=db,
            results_obj=results_obj,
            audio_filepath_loader=audio_filepath_loader,
            sample_rate_hz=sr,
            query_label=args.query_label,
            annotator_id=args.annotator_id,
            host=args.host,
            port=args.port,
            share=args.share,
        )
    else:
        log.info(
            "Search complete. Use --serve to launch the labeling GUI, "
            "or --output-csv to export results."
        )

    return 0


# ---------------------------------------------------------------------------
# Sub-command: train
# ---------------------------------------------------------------------------

def cmd_train(args) -> int:
    (audio_loader_mod, classifier_mod, classifier_data_mod,
     *_rest) = _require_perch()
    import numpy as np

    db = load_db(args.db_dir)

    data_manager = classifier_data_mod.AgileDataManager(
        target_labels=args.target_labels,
        db=db,
        train_ratio=args.train_ratio,
        min_eval_examples=1,
        batch_size=args.batch_size,
        weak_negatives_batch_size=args.weak_neg_batch_size,
        rng=np.random.default_rng(seed=args.seed),
    )

    target_labels = data_manager.get_target_labels()
    log.info("Training classifier for labels: %s", target_labels)
    log.info(
        "  steps=%d  lr=%.1e  weak_neg_weight=%.3f  loss=%s  seed=%d",
        args.num_steps, args.learning_rate, args.weak_neg_weight,
        args.loss_fn, args.seed,
    )

    t0 = time.monotonic()
    linear_classifier, eval_scores = classifier_mod.train_linear_classifier(
        data_manager=data_manager,
        learning_rate=args.learning_rate,
        weak_neg_weight=args.weak_neg_weight,
        num_train_steps=args.num_steps,
    )
    elapsed = time.monotonic() - t0

    log.info("Training complete in %s", _format_duration(elapsed))
    log.info("  top1_acc : %.4f", eval_scores.get("top1_acc", float("nan")))
    log.info("  roc_auc  : %.4f", eval_scores.get("roc_auc", float("nan")))
    log.info("  cmap     : %.4f", eval_scores.get("cmap", float("nan")))

    out_path = Path(args.classifier_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    linear_classifier.save(str(out_path))
    log.info("Classifier saved to %s", out_path)

    # Save eval metrics JSON alongside the model
    metrics_path = out_path.with_suffix(".metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "labels": target_labels,
                "eval_scores": {k: float(v) for k, v in eval_scores.items()},
                "train_args": {
                    "num_steps": args.num_steps,
                    "learning_rate": args.learning_rate,
                    "weak_neg_weight": args.weak_neg_weight,
                    "batch_size": args.batch_size,
                    "train_ratio": args.train_ratio,
                    "seed": args.seed,
                },
            },
            f, indent=2,
        )
    log.info("Eval metrics written to %s", metrics_path)
    return 0


# ---------------------------------------------------------------------------
# Sub-command: review
# ---------------------------------------------------------------------------

def cmd_review(args) -> int:
    (audio_loader_mod, classifier_mod, classifier_data_mod,
     embedding_display_mod, source_info_mod,
     brutalism_mod, interface_mod, score_functions_mod,
     search_results_mod, sqlite_usearch_impl_mod,
     model_configs_mod, np) = _require_perch()

    db = load_db(args.db_dir)
    embedding_model, audio_sources = load_model_from_db(db)

    # If --audio-dir is given, override every base_path in audio_sources.
    # This is needed when the DB was built on a different machine (e.g. Colab)
    # and the audio files are now at a different path on this server.
    if getattr(args, "audio_dir", None):
        from perch_hoplite.agile import source_info as _si
        patched_globs = []
        _globs = getattr(audio_sources, 'audio_globs', getattr(audio_sources, 'audio_sources', []))
        for ag in _globs:
            patched = _si.AudioSourceConfig(
                dataset_name=ag.dataset_name,
                base_path=args.audio_dir,
                file_glob=ag.file_glob,
                min_audio_len_s=ag.min_audio_len_s,
                target_sample_rate_hz=ag.target_sample_rate_hz,
                shard_len_s=ag.shard_len_s,
            )
            patched_globs.append(patched)
        audio_sources = _si.AudioSources(tuple(patched_globs))
        log.info("Audio base path overridden to: %s", args.audio_dir)

    audio_filepath_loader, sr, _ = make_audio_loader(
        db, embedding_model, audio_sources, args.sample_rate_hz
    )

    log.info("Loading classifier from %s", args.classifier)
    linear_classifier = classifier_mod.LinearClassifier.load(args.classifier)

    # Retrieve the weight vector for the target label
    try:
        target_labels = linear_classifier.get_labels()
    except Exception:
        # Fallback: try reading from companion metrics JSON
        metrics_path = Path(args.classifier).with_suffix(".metrics.json")
        if metrics_path.exists():
            with open(metrics_path) as f:
                target_labels = json.load(f).get("labels", [])
        else:
            log.error(
                "Cannot determine classifier labels. "
                "Ensure the .metrics.json companion file exists."
            )
            return 1

    if args.target_label not in target_labels:
        log.error(
            "Target label '%s' not found in classifier labels: %s",
            args.target_label, target_labels,
        )
        return 1

    idx = target_labels.index(args.target_label)

    class_query = linear_classifier.beta[:, idx]
    bias_val = float(linear_classifier.beta_bias[idx])
    log.info(
        "Using classifier weight vector for label '%s' (index %d)",
        args.target_label, idx,
    )

    score_fn = score_functions_mod.get_score_fn(
        "dot", bias=bias_val, target_score=args.margin_target_score,
    )
    results_obj, all_scores = brutalism_mod.threaded_brute_search(
        db, class_query, args.num_results,
        score_fn=score_fn,
        sample_size=args.sample_size if args.sample_size else None,
    )


    hit_scores = [r.sort_score for r in results_obj.search_results]
    log.info(
        "Review search: %d results, score range [%.4f, %.4f]",
        len(results_obj.search_results),
        min(hit_scores) if hit_scores else 0,
        max(hit_scores) if hit_scores else 0,
    )

    if args.output_csv:
        _write_search_csv(results_obj, db, args.output_csv)

    if args.plot_scores:
        _save_histogram(all_scores, hit_scores, args.plot_scores,
                        f"Classifier review — {args.target_label}")

    if args.serve:
        _launch_labeling_gui(
            db=db,
            results_obj=results_obj,
            audio_filepath_loader=audio_filepath_loader,
            sample_rate_hz=sr,
            query_label=args.target_label,
            annotator_id=args.annotator_id,
            host=args.host,
            port=args.port,
            share=args.share,
        )

    return 0


# ---------------------------------------------------------------------------
# Sub-command: infer
# ---------------------------------------------------------------------------

def cmd_infer(args) -> int:
    (audio_loader_mod, classifier_mod, *_rest) = _require_perch()

    db = load_db(args.db_dir)

    log.info("Loading classifier from %s", args.classifier)
    linear_classifier = classifier_mod.LinearClassifier.load(args.classifier)

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log.info(
        "Running inference (logit_threshold=%.3f, labels=%s)...",
        args.logit_threshold, args.labels or "all",
    )
    t0 = time.monotonic()

    classifier_mod.write_inference_csv(
        linear_classifier, db, str(out_path),
        args.logit_threshold,
        labels=args.labels,
    )

    elapsed = time.monotonic() - t0
    log.info("Inference complete in %s", _format_duration(elapsed))

    # Quick summary
    try:
        import csv as csv_mod
        with open(out_path, newline="") as f:
            rows = list(csv_mod.DictReader(f))
        log.info("Total detections written: %d", len(rows))
        from collections import Counter
        by_label = Counter(r.get("label", "?") for r in rows)
        for lbl, cnt in sorted(by_label.items()):
            log.info("  %-40s  %d", lbl, cnt)
    except Exception:
        pass

    if args.plot_distribution:
        try:
            import pandas as pd
            import seaborn as sns
            plt_mod, np_mod = _require_matplotlib()
            df = pd.read_csv(out_path)
            fig, ax = plt_mod.subplots(figsize=(12, 5))
            for lbl in df["label"].unique():
                subset = df[df["label"] == lbl]["logits"]
                ax.hist(subset, bins=50, alpha=0.6, label=lbl)
            ax.set_title("Inference logit distribution by class")
            ax.set_xlabel("Logit")
            ax.set_ylabel("Count")
            ax.legend()
            fig.tight_layout()
            dist_path = Path(args.plot_distribution)
            dist_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(dist_path), dpi=150)
            plt_mod.close(fig)
            log.info("Distribution plot saved to %s", dist_path)
        except ImportError as exc:
            log.warning("Could not generate distribution plot: %s", exc)

    log.info("Results written to %s", out_path)
    return 0


# ---------------------------------------------------------------------------
# Gradio labeling GUI
# ---------------------------------------------------------------------------

def _launch_labeling_gui(
    db,
    results_obj,
    audio_filepath_loader,
    sample_rate_hz: int,
    query_label: str,
    annotator_id: str,
    host: str,
    port: int,
    share: bool,
) -> None:
    """
    Launch a Gradio web app for interactive audio labeling.

    The app displays up to N search results; each result shows:
      - Waveform plot of the audio segment
      - An HTML5 audio player for playback
      - Positive / Negative / Unlabeled radio buttons

    On clicking "Save Labels", all annotations are written back to the DB.

    Access via http://<server-ip>:<port> from any browser on the LAN.
    """
    gr = _require_gradio()
    plt_mod, np_mod = _require_matplotlib()
    import io, base64, soundfile as sf

    from perch_hoplite.db import interface as iface

    log.info("Building Gradio labeling interface...")

    # Pre-load all result segments into memory
    segments = []
    for r in results_obj.search_results:
        wid = r.window_id
        try:
            audio, sr_actual = audio_filepath_loader(wid)
            source = _get_source(db, wid)
            recording_id = getattr(source, "source_id", str(wid))
            offsets = getattr(source, "offsets", (0.0, 5.0))
        except Exception as exc:
            import traceback as _tb
            log.warning("Could not load audio for window %s: %s\n%s",
                        wid, exc, _tb.format_exc())
            continue
        segments.append({
            "window_id": int(wid),
            "recording_id": recording_id,
            "offset_s": offsets[0] if offsets else 0.0,
            "end_offset_s": offsets[1] if offsets else 5.0,
            "score": r.sort_score,
            "audio": audio,
            "sample_rate": sr_actual or sample_rate_hz,
        })

    log.info("Loaded %d audio segments for labeling.", len(segments))

    def _make_spectrogram_image(audio_array: "np.ndarray", sr: int) -> str:
        """Return a base64-encoded PNG spectrogram (linear-frequency STFT).

        Frequency axis covers 0 – 16 kHz (useful range for cetaceans at 32 kHz
        sample rate).  Color scale is log-power (dB), clipped to a 60 dB dynamic
        range so low-energy calls remain visible against broadband background.
        """
        import numpy as _np2
        from scipy.signal import spectrogram as _spec

        # Compute spectrogram: 512-sample window, 75% overlap
        nperseg = min(512, len(audio_array) // 4)
        f, t, Sxx = _spec(
            audio_array, fs=sr,
            nperseg=nperseg, noverlap=nperseg * 3 // 4,
            scaling="density",
        )
        # Convert to log power (dB), clip to 60 dB dynamic range
        Sxx_db = 10 * _np2.log10(Sxx + 1e-10)
        vmax = Sxx_db.max()
        vmin = vmax - 60.0

        # Only show up to 16 kHz
        f_max = min(16000, sr // 2)
        f_mask = f <= f_max
        f_plot = f[f_mask]
        S_plot = Sxx_db[f_mask, :]

        fig, axes = plt_mod.subplots(
            2, 1, figsize=(7, 3),
            gridspec_kw={"height_ratios": [2.5, 1], "hspace": 0.05},
        )
        fig.patch.set_facecolor("#111827")

        # Spectrogram (top panel)
        ax_spec = axes[0]
        ax_spec.pcolormesh(
            t, f_plot, S_plot,
            vmin=vmin, vmax=vmax,
            cmap="inferno", shading="gouraud",
        )
        ax_spec.set_ylabel("Hz", color="#94a3b8", fontsize=8)
        ax_spec.tick_params(colors="#94a3b8", labelsize=7)
        ax_spec.set_facecolor("#111827")
        for spine in ax_spec.spines.values():
            spine.set_edgecolor("#334155")
        ax_spec.tick_params(bottom=False, labelbottom=False)

        # Waveform (bottom panel)
        ax_wave = axes[1]
        t_wave = _np2.linspace(0, len(audio_array) / sr, len(audio_array))
        ax_wave.plot(t_wave, audio_array, color="#38bdf8", linewidth=0.4)
        ax_wave.set_xlim([0, t_wave[-1]])
        ax_wave.set_xlabel("Time (s)", color="#94a3b8", fontsize=8)
        ax_wave.set_facecolor("#111827")
        ax_wave.tick_params(colors="#94a3b8", labelsize=7)
        for spine in ax_wave.spines.values():
            spine.set_edgecolor("#334155")
        ax_wave.set_yticks([])

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                    facecolor="#111827")
        plt_mod.close(fig)
        buf.seek(0)
        return "data:image/png;base64," + base64.b64encode(buf.read()).decode()

    def _make_audio_b64(audio_array: "np.ndarray", sr: int) -> str:
        """Return a base64-encoded WAV for HTML5 audio element."""
        buf = io.BytesIO()
        sf.write(buf, audio_array, sr, format="WAV")
        buf.seek(0)
        return "data:audio/wav;base64," + base64.b64encode(buf.read()).decode()

    # Build per-segment HTML card
    def _segment_card(seg: dict, idx: int) -> str:
        wav_b64  = _make_audio_b64(seg["audio"], seg["sample_rate"])
        spec_b64 = _make_spectrogram_image(seg["audio"], seg["sample_rate"])
        fname = seg["recording_id"].split("/")[-1]   # basename only for display
        return (
            f"<div style='background:#1e293b;border-radius:8px;padding:12px;"
            f"margin-bottom:8px;color:#e2e8f0;font-family:monospace;font-size:11px;'>"
            f"<b>#{idx+1}</b> &nbsp; <span style='color:#7dd3fc'>{fname}</span>"
            f" &nbsp; <span style='color:#94a3b8'>"
            f"{seg['offset_s']:.1f}s – {seg['end_offset_s']:.1f}s</span>"
            f" &nbsp; <span style='color:#fbbf24'>score={seg['score']:.3f}</span><br>"
            f"<img src='{spec_b64}' style='width:100%;margin-top:6px;"
            f"border-radius:4px;display:block;'/>"
            f"<audio controls style='width:100%;margin-top:6px;' src='{wav_b64}'></audio>"
            f"</div>"
        )

    # State: labels assigned in the GUI
    label_state: dict[int, str] = {}  # window_id -> "positive"|"negative"|"unlabeled"

    with gr.Blocks(
        title="Perch Hoplite — Audio Labeling",
        css=(
            "body { background: #0f172a; color: #e2e8f0; font-family: 'Courier New', monospace; }"
            ".gr-button-primary { background: #0ea5e9 !important; }"
            ".gr-button { border-radius: 6px !important; }"
        ),
    ) as demo:
        gr.Markdown(
            f"""
# 🐋 Perch Hoplite — Audio Labeling Interface
**Query label:** `{query_label}` &nbsp;&nbsp; **Annotator:** `{annotator_id}`  
**Results loaded:** {len(segments)}  
Click **Positive** (🟢) or **Negative** (🔴) for each segment, then **Save Labels to DB**.
"""
        )

        save_btn = gr.Button("💾 Save Labels to DB", variant="primary")
        status_box = gr.Textbox(label="Status", interactive=False, lines=3)

        radio_components = []

        with gr.Column():
            for i, seg in enumerate(segments):
                with gr.Row():
                    with gr.Column(scale=4):
                        gr.HTML(_segment_card(seg, i))
                    with gr.Column(scale=1):
                        radio = gr.Radio(
                            choices=["positive", "negative", "unlabeled"],
                            value="unlabeled",
                            label=f"Label #{i+1}",
                        )
                        radio_components.append((seg["window_id"], radio))

        def save_labels(*radio_values):
            # iface is already imported at the top of _launch_labeling_gui
            saved = 0
            skipped = 0
            for (wid, _), choice in zip(radio_components, radio_values):
                if choice == "unlabeled":
                    skipped += 1
                    continue
                lt = (
                    interface.LabelType.POSITIVE
                    if choice == "positive"
                    else interface.LabelType.NEGATIVE
                )
                try:
                    source = _get_source(db, wid)
                    # insert_annotation requires the integer recording_id stored on
                    # the EmbeddingSource, NOT the filename string (source_id).
                    rec_int_id = getattr(source, "recording_id", None)
                    if rec_int_id is None:
                        rec_int_id = int(wid)
                    offsets = getattr(source, "offsets", (0.0, 5.0))
                    db.insert_annotation(
                        recording_id=rec_int_id,
                        offsets=offsets,
                        label=query_label,
                        label_type=lt,
                        provenance=f"gradio_gui:{annotator_id}",
                        handle_duplicates="update",
                    )
                    saved += 1
                except Exception as exc:
                    log.warning("Failed to save label for window %s: %s", wid, exc)

            try:
                pos_counts = db.count_each_label(label_type=interface.LabelType.POSITIVE)
                neg_counts = db.count_each_label(label_type=interface.LabelType.NEGATIVE)
                counts_str = (
                    f"Total positive: {dict(pos_counts)}\n"
                    f"Total negative: {dict(neg_counts)}"
                )
            except Exception as exc:
                counts_str = f"(Could not read label counts: {exc})"

            msg = f"Saved {saved} labels ({skipped} unlabeled skipped).\n{counts_str}"
            log.info(msg)
            return msg

        save_btn.click(
            fn=save_labels,
            inputs=[r for _, r in radio_components],
            outputs=status_box,
        )

    log.info("=" * 60)
    log.info("Launching Gradio labeling GUI")
    log.info("  Access at: http://%s:%d", host if host != "0.0.0.0" else "<server-ip>", port)
    log.info("  Press Ctrl+C to stop the server.")
    log.info("=" * 60)

    demo.launch(
        server_name=host,
        server_port=port,
        share=share,
        show_error=True,
        quiet=False,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

COMMANDS = {
    "search": cmd_search,
    "label": cmd_label,
    "train": cmd_train,
    "review": cmd_review,
    "infer": cmd_infer,
    "stats": cmd_stats,
}


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    db_dir = Path(args.db_dir)
    log_dir = Path(args.log_dir) if args.log_dir else db_dir / "logs"
    _setup_logging(log_dir, args.verbose)

    log.info("Perch Hoplite Phase 2 — Search / Classify / Infer")
    log.info("Python %s  Command: %s", sys.version.split()[0], args.command)

    fn = COMMANDS.get(args.command)
    if fn is None:
        log.error("Unknown command: %s", args.command)
        return 1

    return fn(args)


if __name__ == "__main__":
    sys.exit(main())
