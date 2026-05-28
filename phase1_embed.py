#!/usr/bin/env python3
"""
phase1_embed.py — Perch Hoplite Phase 1: Audio Embedding Pipeline
==================================================================
Initializes a Hoplite SQLite+USearch database and embeds audio files
using a pre-trained Perch/multispecies-whale model.

Environment
-----------
MBARI NVIDIA DGX SPARC nodes: spark-ae0e (134.89.11.107)
                               spark-0626 (134.89.11.174)
Python 3.12, venv at ~/perch-hoplite/
NFS audio source : /mnt/PAM_Archive/   (thalassa.shore.mbari.org)
NFS results      : /mnt/PAM_Analysis/duane_scratch/perch_hoplite/
Local fast DB    : /home/duane/perch_work/db/   (NVMe, 3.3 TB free)

Storage strategy
----------------
Always write --db-dir to local NVMe during Phase 1 for fast sequential
writes. After embedding completes, sync to NFS with:
    ~/perch_work/sync_db_to_nfs.sh <dataset-name>
Phase 2 then reads the DB from NFS.

Usage examples
--------------
# Dry run first — validate config without writing anything:
python3 phase1_embed.py \
    --dataset-name MARS_2018 \
    --audio-dir /mnt/PAM_Archive/2018 \
    --file-glob "*.flac" \
    --db-dir /home/duane/perch_work/db/MARS_2018 \
    --model perch_v2 \
    --dry-run

# Real embedding run — MARS 2018 hydrophone recordings:
python3 phase1_embed.py \
    --dataset-name MARS_2018 \
    --audio-dir /mnt/PAM_Archive/2018 \
    --file-glob "*.flac" \
    --db-dir /home/duane/perch_work/db/MARS_2018 \
    --model perch_v2 \
    --shard-len 75

# MOBB deployment, multispecies whale model:
python3 phase1_embed.py \
    --dataset-name MOBB_2022 \
    --audio-dir /mnt/PAM_Archive/2022 \
    --file-glob "*.flac" \
    --db-dir /home/duane/perch_work/db/MOBB_2022 \
    --model multispecies_whale \
    --shard-len 75

# Embed from a GCS public bucket (requires gcloud auth):
python3 phase1_embed.py \
    --dataset-name saipan_A_06 \
    --audio-dir gs://noaa-passive-bioacoustic/pifsc/audio/pipan_10/saipan/pipan_saipan_06/audio \
    --file-glob "Saipan_A_06_151006_091215.df20.*.flac" \
    --db-dir /home/duane/perch_work/db/saipan_A_06 \
    --model multispecies_whale \
    --shard-len 75

# Drop an existing database and re-embed:
python3 phase1_embed.py ... --drop-existing

# Print DB statistics without embedding:
python3 phase1_embed.py ... --stats-only

Supported models
----------------
  perch_v2           Google Perch V2 (recommended; GPU required)
                     Broadest coverage, best few-shot accuracy.
  multispecies_whale Google Multispecies Whale model
                     Pre-trained on cetaceans; try if perch_v2
                     underperforms on baleen whale calls.
  humpback           Humpback whale detector (narrow, accurate)
  surfperch          SurfPerch (coral reef — not ideal for MBARI sites)
  perch_8            Google Perch V1 (bird-focused, not useful here)
  birdnet_V2.3       BirdNET V2.3 (bird-focused, not useful here)

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
import logging
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging setup — rich console output + rotating file log
# ---------------------------------------------------------------------------
LOG_FORMAT = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
LOG_DATE   = "%Y-%m-%d %H:%M:%S"

def _setup_logging(log_dir: Path, verbose: bool) -> None:
    """Configure root logger with console + rotating file handler."""
    import logging.handlers

    level = logging.DEBUG if verbose else logging.INFO
    root = logging.getLogger()
    root.setLevel(level)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE))
    root.addHandler(ch)

    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "phase1_embed.log"
    fh = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=50 * 1024 * 1024, backupCount=5
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE))
    root.addHandler(fh)
    logging.info("Logging to %s", log_file)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
SUPPORTED_MODELS = [
    "perch_v2",
    "perch_8",
    "multispecies_whale",
    "surfperch",
    "humpback",
    "birdnet_V2.3",
]

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="phase1_embed.py",
        description="Perch Hoplite Phase 1 — embed audio files into a Hoplite vector database.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # --- Audio source ---
    src = p.add_argument_group("Audio source")
    src.add_argument(
        "--dataset-name", "-n",
        required=True,
        help="Logical name for this dataset (used as the DB project key).",
    )
    src.add_argument(
        "--audio-dir", "-i",
        required=True,
        help=(
            "Base directory (local path or GCS URI like gs://...) "
            "containing the audio files."
        ),
    )
    src.add_argument(
        "--file-glob", "-g",
        default="*.flac",
        help="Glob pattern for audio files within --audio-dir (default: *.flac).",
    )
    src.add_argument(
        "--min-audio-len", type=float, default=1.0,
        help="Minimum audio duration in seconds to include a file (default: 1.0).",
    )

    # --- Database ---
    db_grp = p.add_argument_group("Database")
    db_grp.add_argument(
        "--db-dir", "-d",
        required=True,
        help=(
            "Local directory where the Hoplite database will be stored "
            "(hoplite.sqlite + usearch.index). Created if it does not exist. "
            "On spark-ae0e/spark-0626 use local NVMe for fast writes: "
            "/home/duane/perch_work/db/<dataset-name>. "
            "After embedding, sync to NFS with sync_db_to_nfs.sh."
        ),
    )
    db_grp.add_argument(
        "--drop-existing",
        action="store_true",
        default=False,
        help=(
            "If the database already exists and contains embeddings, "
            "delete them and start fresh. Default: append / skip existing."
        ),
    )

    # --- Model ---
    mdl = p.add_argument_group("Embedding model")
    mdl.add_argument(
        "--model", "-m",
        choices=SUPPORTED_MODELS,
        default="perch_v2",
        help=f"Pre-trained model to use for embedding (default: perch_v2).",
    )
    mdl.add_argument(
        "--target-sample-rate", type=int, default=-1,
        metavar="HZ",
        help=(
            "Target sample rate in Hz. "
            "-2 = model default, -1 = source file rate, >0 = explicit rate "
            "(default: -1)."
        ),
    )

    # --- Sharding ---
    shard = p.add_argument_group("File sharding")
    shard.add_argument(
        "--shard-len", type=float, default=75.0,
        metavar="SECONDS",
        help=(
            "Shard long audio files into chunks of this length before embedding. "
            "Reduces GPU memory pressure. Set to 0 to disable sharding (default: 75)."
        ),
    )

    # --- Output / logging ---
    out = p.add_argument_group("Output and logging")
    out.add_argument(
        "--log-dir",
        default=None,
        help="Directory for log files (default: --db-dir/logs).",
    )
    out.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Enable DEBUG-level logging.",
    )
    out.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Validate configuration and print a summary without embedding.",
    )
    out.add_argument(
        "--stats-only",
        action="store_true",
        default=False,
        help="Print database statistics and exit without embedding.",
    )

    return p


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_imports():
    """Import perch-hoplite lazily so the --help flag works without it installed."""
    try:
        from perch_hoplite.agile import colab_utils, embed, source_info
        from perch_hoplite.db import brutalism, interface
        from perch_hoplite.zoo import taxonomy_model_tf
        from etils import epath
        from ml_collections import config_dict
        import numpy as np
    except ImportError as exc:
        log.error(
            "perch-hoplite is not installed. "
            "Run: pip install git+https://github.com/google-research/perch-hoplite.git\n"
            "Error: %s", exc,
        )
        sys.exit(1)
    return (colab_utils, embed, source_info, brutalism, interface,
            taxonomy_model_tf, epath, config_dict, np)


def _check_gpu():
    """Log GPU status and enable memory growth (avoids nvidia-smi version mismatch on DGX SPARC).

    Memory growth means TensorFlow allocates GPU RAM incrementally as needed
    rather than reserving all available memory at startup. Important on the
    GB10 which has only ~3.7 GB total GPU memory.
    Must be called before any TF operations that allocate tensors.
    """
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            for g in gpus:
                try:
                    tf.config.experimental.set_memory_growth(g, True)
                except RuntimeError as e:
                    log.debug("Memory growth already set or GPU initialized: %s", e)
                details = tf.config.experimental.get_device_details(g)
                name = details.get("device_name", g.name)
                log.info("GPU ready: %s (memory growth enabled)", name)
        else:
            log.warning(
                "No GPU visible to TensorFlow. "
                "Embedding will run on CPU and be very slow. "
                "Check that nvidia-tensorflow is installed and CUDA is available."
            )
    except Exception as exc:
        log.warning("Could not query GPU via TensorFlow: %s", exc)


def _format_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}h {m:02d}m {s:05.2f}s"


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def run_stats(db) -> None:
    """Print database statistics to stdout."""
    from perch_hoplite.db import interface as iface
    projects = db.get_all_projects()
    total = db.count_embeddings()
    log.info("=" * 60)
    log.info("DATABASE STATISTICS")
    log.info("=" * 60)
    log.info("Total embeddings : %d", total)
    log.info("Projects         : %d", len(projects))
    for proj in projects:
        from ml_collections import config_dict
        ids = db.match_window_ids(
            deployments_filter=config_dict.create(eq=dict(project=proj))
        )
        log.info("  %-40s  %d embeddings", proj, len(ids))
    ann_count = len(db.get_all_annotations())
    log.info("Total annotations: %d", ann_count)
    if ann_count > 0:
        try:
            pos = db.count_each_label(label_type=iface.LabelType.POSITIVE)
            neg = db.count_each_label(label_type=iface.LabelType.NEGATIVE)
            log.info("  Positive labels: %s", pos)
            log.info("  Negative labels: %s", neg)
        except Exception:
            pass
    log.info("=" * 60)


def initialize_db(db_dir: Path, drop_existing: bool, configs):
    """Open or create the Hoplite DB, optionally dropping existing data."""
    from etils import epath

    db = configs.db_config.load_db()
    num_existing = db.count_embeddings()
    log.info("Database location : %s", configs.db_config.db_config.db_path)
    log.info("Existing embeddings: %d", num_existing)

    if num_existing > 0 and drop_existing:
        log.warning(
            "drop_existing=True — deleting %d existing embeddings at %s",
            num_existing, configs.db_config.db_config.db_path,
        )
        db_path_obj = epath.Path(configs.db_config.db_config.db_path)
        for fp in db_path_obj.glob("hoplite.sqlite*"):
            fp.unlink()
        usearch_idx = db_path_obj / "usearch.index"
        if usearch_idx.exists():
            usearch_idx.unlink()
        log.info("Dropped existing database. Re-initializing...")
        db = configs.db_config.load_db()
        log.info("Fresh database initialized.")
    elif num_existing > 0:
        log.info(
            "Appending to existing database (%d embeddings already present). "
            "Use --drop-existing to start fresh.",
            num_existing,
        )

    return db


def embed_audio(args, configs, db) -> None:
    """Run the embedding loop."""
    from perch_hoplite.agile import embed as embed_mod

    audio_glob = configs.audio_sources_config.audio_globs[0]

    log.info("=" * 60)
    log.info("EMBEDDING START")
    log.info("  Dataset    : %s", audio_glob.dataset_name)
    log.info("  Audio dir  : %s", audio_glob.base_path)
    log.info("  File glob  : %s", audio_glob.file_glob)
    log.info("  Model      : %s", args.model)
    log.info("  Shard len  : %s s", args.shard_len if args.shard_len > 0 else "disabled")
    log.info("  Sample rate: %s", args.target_sample_rate)
    log.info("=" * 60)

    worker = embed_mod.EmbedWorker(
        audio_sources=configs.audio_sources_config,
        db=db,
        model_config=configs.model_config,
    )

    t0 = time.monotonic()
    worker.process_all(target_dataset_name=audio_glob.dataset_name)
    elapsed = time.monotonic() - t0

    total = db.count_embeddings()
    log.info("=" * 60)
    log.info("EMBEDDING COMPLETE")
    log.info("  Total embeddings : %d", total)
    log.info("  Elapsed time     : %s", _format_duration(elapsed))
    log.info("=" * 60)


def sanity_search(db, np) -> None:
    """Run a quick nearest-neighbour sanity check after embedding."""
    from perch_hoplite.db import brutalism
    log.info("Running sanity nearest-neighbour search...")
    ids = db.match_window_ids(limit=1)
    if not ids:
        log.warning("No embeddings found — cannot run sanity search.")
        return
    q = db.get_embedding(ids[0])
    results, scores = brutalism.brute_search(
        db, query_embedding=q, search_list_size=16, score_fn=np.dot
    )
    log.info(
        "Sanity search passed. Top-16 window IDs: %s",
        [int(r.window_id) for r in results],
    )
    log.info("Score range: min=%.4f  max=%.4f", float(scores.min()), float(scores.max()))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Resolve log dir
    db_dir = Path(args.db_dir)
    log_dir = Path(args.log_dir) if args.log_dir else db_dir / "logs"
    _setup_logging(log_dir, args.verbose)

    log.info("Perch Hoplite Phase 1 — Audio Embedding Pipeline")
    log.info("Python %s", sys.version.split()[0])

    _check_gpu()

    # Lazy imports
    (colab_utils, embed_mod, source_info,
     brutalism, interface, taxonomy_model_tf,
     epath, config_dict, np) = _require_imports()

    # Ensure DB directory exists
    db_dir.mkdir(parents=True, exist_ok=True)

    # Build audio source config
    shard_len = float(args.shard_len) if args.shard_len > 0 else None
    audio_glob = source_info.AudioSourceConfig(
        dataset_name=args.dataset_name,
        base_path=args.audio_dir,
        file_glob=args.file_glob,
        min_audio_len_s=args.min_audio_len,
        target_sample_rate_hz=args.target_sample_rate,
        shard_len_s=shard_len,
    )

    # Build full config bundle
    configs = colab_utils.load_configs(
        source_info.AudioSources((audio_glob,)),
        str(db_dir),
        model_config_key=args.model,
        db_key="sqlite_usearch",
    )

    if args.dry_run:
        log.info("[DRY RUN] Configuration validated successfully.")
        log.info("  Dataset name  : %s", args.dataset_name)
        log.info("  Audio dir     : %s", args.audio_dir)
        log.info("  File glob     : %s", args.file_glob)
        log.info("  DB dir        : %s", db_dir)
        log.info("  Model         : %s", args.model)
        log.info("  Shard len     : %s", shard_len)
        log.info("  Sample rate   : %s", args.target_sample_rate)
        log.info("  Drop existing : %s", args.drop_existing)
        log.info("[DRY RUN] Exiting without embedding.")
        return 0

    # Open / create database
    db = initialize_db(db_dir, args.drop_existing, configs)

    if args.stats_only:
        run_stats(db)
        return 0

    # Run embedding
    embed_audio(args, configs, db)

    # Post-embedding statistics
    run_stats(db)

    # Sanity search
    try:
        sanity_search(db, np)
    except Exception as exc:
        log.warning("Sanity search failed (non-fatal): %s", exc)

    log.info("Phase 1 complete. Database ready for Phase 2 (phase2_classify.py).")
    log.info("DB path: %s", db_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
