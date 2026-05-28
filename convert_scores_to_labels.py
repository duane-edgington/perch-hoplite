#!/usr/bin/env python3
"""
convert_scores_to_labels.py — Convert Google Multispecies Whale Model scores
to Hoplite-compatible label CSVs for bootstrapping Phase 2 classification.

Reads the _expanded.csv files from scores_gpu/ and applies tiered labeling
rules to produce a CSV ready for import via:
    python3 phase2_classify.py label --labels-csv <output> --db-dir <db>

Tiered labeling strategy for orca (Oo) classification
------------------------------------------------------
The Google model confuses orca calls with dolphin calls and generic
vocalizations. The key discriminator is what appears alongside Oo in the
top-ranked classes:

  POSITIVE (high confidence orca):
    - Oo score >= --oo-pos-threshold (default 0.5)
    - AND top-3 classes do NOT include Call, Whistle, or Echolocation
    - (Clean orca: Oo dominant, other whales as runners-up)

  POSITIVE (moderate, for review):
    - Oo score >= --oo-pos-threshold
    - AND Call/Whistle/Echolocation present in top-3
    - Written with label_type=weak_positive (review before training)

  NEGATIVE (reliable non-orca from other whale detections):
    - Oo score <= --oo-neg-threshold (default 0.05)
    - AND any of Bm, Bp, Mn score >= --whale-pos-threshold (default 0.3)
    - (Strong whale detection but definitely not orca)

  SKIP (confusion zone — queue for manual Gradio review):
    - Everything else (oo score 0.05-0.5, or high oo with confusion classes)
    - Not written to output; listed in a separate review queue CSV

All classes supported
---------------------
  Oo  = Orca (Orcinus orca)
  Mn  = Humpback whale (Megaptera novaeangliae)
  Bp  = Fin whale (Balaenoptera physalus)
  Bm  = Blue whale (Balaenoptera musculus)
  Eg  = North Atlantic right whale (Eubalaena glacialis) — rare at MARS
  Ba  = Minke whale (Balaenoptera acutorostrata)
  Be  = Bowhead whale (Balaenoptera edeni / Bryde's)
  Call, Whistle, Echolocation = generic odontocete/dolphin indicators
  Upcall = generic upcall
  Gunshot = impulsive noise

Usage examples
--------------
# Convert November 2021 scores, orca class only:
python3 convert_scores_to_labels.py \\
    --scores-dir /mnt/PAM_Analysis/GoogleMultiSpeciesWhaleModel2/scores_gpu/2021/11 \\
    --output-csv /mnt/PAM_Analysis/duane_scratch/perch_hoplite/labels/MARS_2021_11_orca_bootstrap.csv \\
    --target-class oo \\
    --annotator-id google_multispecies_whale_model

# Convert all years/months, all four main classes:
python3 convert_scores_to_labels.py \\
    --scores-dir /mnt/PAM_Analysis/GoogleMultiSpeciesWhaleModel2/scores_gpu \\
    --output-csv /mnt/PAM_Analysis/duane_scratch/perch_hoplite/labels/MARS_all_bootstrap.csv \\
    --target-class all \\
    --recursive

# Stricter orca threshold, also write review queue:
python3 convert_scores_to_labels.py \\
    --scores-dir /mnt/PAM_Analysis/GoogleMultiSpeciesWhaleModel2/scores_gpu/2021/11 \\
    --output-csv /mnt/PAM_Analysis/duane_scratch/perch_hoplite/labels/MARS_2021_11_orca_strict.csv \\
    --target-class oo \\
    --oo-pos-threshold 0.7 \\
    --oo-neg-threshold 0.05 \\
    --whale-pos-threshold 0.4 \\
    --review-queue /mnt/PAM_Analysis/duane_scratch/perch_hoplite/labels/MARS_2021_11_orca_review_queue.csv

Output CSV columns (Hoplite label import format)
-------------------------------------------------
  recording_id   — source audio filename without path or extension
  offset_s       — start time in seconds from file start
  end_offset_s   — end time (offset_s + 5.0)
  label          — class label string (e.g. orca_call, humpback_song)
  label_type     — positive | negative | weak_negative
  epoch_seconds  — original epoch timestamp
  oo_score       — Oo class score from Google model
  top3_classes   — top 3 ranked classes (for auditing confusion)
  source_file    — path to the expanded CSV this row came from
"""

import argparse
import csv
import logging
import logging.handlers
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

LOG_FORMAT = "%(asctime)s  %(levelname)-8s  %(message)s"
LOG_DATE   = "%Y-%m-%d %H:%M:%S"

def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    root = logging.getLogger()
    root.setLevel(level)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE))
    root.addHandler(ch)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Class name mappings
# ---------------------------------------------------------------------------

# Google model class codes -> human-readable label strings for Hoplite
CLASS_LABEL_MAP = {
    "oo":            "orca_call",
    "mn":            "humpback_song",
    "bp":            "fin_whale_call",
    "bm":            "blue_whale_call",
    "eg":            "right_whale_upcall",
    "ba":            "minke_whale_call",
    "be":            "brydes_whale_call",
    "call":          "dolphin_call",
    "whistle":       "dolphin_whistle",
    "echolocation":  "dolphin_click",
    "upcall":        "generic_upcall",
    "gunshot":       "impulsive_noise",
}

# Classes that indicate dolphin/generic vocalization confusion with orca
CONFUSION_CLASSES = {"call", "whistle", "echolocation"}

# Whale species classes (reliable non-orca when scoring high)
WHALE_CLASSES = {"mn", "bp", "bm", "eg", "ba", "be"}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="convert_scores_to_labels.py",
        description="Convert Google Multispecies Whale Model scores to Hoplite label CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--scores-dir", "-i", required=True,
                   help="Directory containing _expanded.csv files (or tree root with --recursive).")
    p.add_argument("--output-csv", "-o", required=True,
                   help="Output label CSV for Hoplite import.")
    p.add_argument("--target-class", "-c", default="oo",
                   choices=list(CLASS_LABEL_MAP.keys()) + ["all"],
                   help="Which class to generate labels for (default: oo = orca).")
    p.add_argument("--recursive", "-r", action="store_true",
                   help="Recursively search --scores-dir for _expanded.csv files.")
    p.add_argument("--annotator-id", default="google_multispecies_whale_v2",
                   help="Annotator ID written into the label CSV.")

    thresh = p.add_argument_group("Thresholds")
    thresh.add_argument("--oo-pos-threshold", type=float, default=0.5,
                        help="Minimum Oo score for a positive orca label (default: 0.5).")
    thresh.add_argument("--oo-neg-threshold", type=float, default=0.05,
                        help="Maximum Oo score for a negative orca label (default: 0.05).")
    thresh.add_argument("--whale-pos-threshold", type=float, default=0.3,
                        help="Minimum whale class score to generate a reliable negative (default: 0.3).")
    thresh.add_argument("--generic-class-threshold", type=float, default=0.5,
                        help="Min score for non-orca classes (mn/bp/bm etc) positives (default: 0.5).")
    thresh.add_argument("--logit-threshold", type=float, default=3.0,
                        help=(
                            "Minimum top-class logit for a high-confidence positive, "
                            "bypassing the confusion-class filter. "
                            "Empirically calibrated on MARS 2018 orca events: "
                            "logit > 3.0 = unambiguous orca (default: 3.0)."
                        ))
    thresh.add_argument("--no-confusion-filter", action="store_true",
                        help="Disable the Call/Whistle/Echolocation confusion filter for orca positives.")

    p.add_argument("--review-queue", default=None,
                   help="Optional: write confusion-zone windows to this CSV for manual Gradio review.")
    p.add_argument("--dry-run", action="store_true",
                   help="Count rows without writing files.")
    p.add_argument("--verbose", "-v", action="store_true")
    return p

# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def find_expanded_csvs(scores_dir: Path, recursive: bool) -> list[Path]:
    """Find all _expanded.csv files under scores_dir."""
    if recursive:
        files = sorted(scores_dir.rglob("*_expanded.csv"))
    else:
        files = sorted(scores_dir.glob("*_expanded.csv"))
    log.info("Found %d _expanded.csv files in %s", len(files), scores_dir)
    return files


def parse_expanded_csv(csv_path: Path) -> list[dict]:
    """Parse one _expanded.csv into a list of row dicts."""
    rows = []
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    except Exception as exc:
        log.warning("Could not parse %s: %s", csv_path, exc)
    return rows


def extract_class_scores(row: dict) -> dict[str, float]:
    """
    Extract a class_name -> score mapping from an expanded CSV row.
    The expanded CSV has ranked columns (class_names_1..12, scores_1..12).
    Returns a flat dict of lowercase class name -> score.
    """
    scores = {}
    for i in range(1, 13):
        name_key  = f"class_names_{i}"
        score_key = f"scores_{i}"
        if name_key in row and score_key in row:
            try:
                name  = row[name_key].strip().lower()
                score = float(row[score_key])
                scores[name] = score
            except (ValueError, KeyError):
                pass
    return scores


def get_top3(row: dict) -> list[str]:
    """Return the top-3 ranked class names from a row."""
    top3 = []
    for i in range(1, 4):
        key = f"class_names_{i}"
        if key in row:
            top3.append(row[key].strip().lower())
    return top3


def recording_id_from_filename(filename_field: str) -> str:
    """
    Extract a clean recording_id from the filename field in the expanded CSV.
    The filename field looks like:
      /mnt/.../resampled_24kHz_chunks/2021/11/MARS_20211101_000938_resampled_24kHz/
          MARS_20211101_000938_resampled_24kHz_chunk_001.wav
    We want the base recording name without chunk suffix and without extension:
      MARS_20211101_000938_resampled_24kHz
    """
    fname = Path(filename_field.strip()).stem  # remove .wav
    # Remove _chunk_NNN suffix
    parts = fname.rsplit("_chunk_", 1)
    return parts[0]


def offset_from_chunk_filename(filename_field: str, window_size_s: float = 5.0) -> tuple[float, float]:
    """
    Extract time offset from chunk filename.
    Chunk filenames end in _chunk_001, _chunk_002 etc (1-indexed).
    offset_s = (chunk_number - 1) * window_size_s
    """
    fname = Path(filename_field.strip()).stem
    parts = fname.rsplit("_chunk_", 1)
    if len(parts) == 2:
        try:
            chunk_num = int(parts[1])
            offset_s = (chunk_num - 1) * window_size_s
            return offset_s, offset_s + window_size_s
        except ValueError:
            pass
    return 0.0, window_size_s


def classify_orca_row(
    scores: dict[str, float],
    top3: list[str],
    args,
    logit_1: float = 0.0,
) -> tuple[str | None, str | None]:
    """
    Apply tiered orca labeling rules based on empirical analysis of
    MARS April 2018 confirmed orca events.

    Key finding: during confirmed orca passages, Call/Whistle/Echolocation
    appear as runners-up even at very high logits (8.5+). The logit value
    of the top-ranked Oo class is the reliable discriminator, NOT the
    runner-up class names. High logit = true orca regardless of confusion
    classes present. Low-to-moderate logit with confusion runners-up =
    dolphin or generic vocalization.

    Tiers:
      POSITIVE (clean):    Oo score >= threshold AND logit_1 >= logit_threshold
                           These are high-confidence orca even with confusion
                           classes as runners-up.
      POSITIVE (moderate): Oo score >= threshold AND logit_1 < logit_threshold
                           AND no confusion runners-up. Moderate confidence.
      REVIEW:              Oo score >= threshold AND logit_1 < logit_threshold
                           AND confusion classes present. Likely dolphin —
                           send to Gradio for manual verification.
      NEGATIVE:            Oo score <= neg_threshold AND strong whale present.
      SKIP:                Everything else.

    Returns (label_string, label_type) or (None, None) to skip.
    label_type: 'positive', 'negative', 'weak_negative', 'review'
    """
    oo_score = scores.get("oo", 0.0)
    label    = CLASS_LABEL_MAP["oo"]

    # Check for confusion classes in top-3 runners-up (positions 2 and 3)
    # Note: top3[0] is always Oo when oo_score is high; check positions 1,2
    has_confusion = any(c in CONFUSION_CLASSES for c in top3[1:3])

    # Best non-orca whale score
    best_whale_score = max((scores.get(w, 0.0) for w in WHALE_CLASSES), default=0.0)

    # --- HIGH-CONFIDENCE POSITIVE ---
    # High logit means Oo is strongly dominant regardless of runner-up classes.
    # Empirically: logit > 3.0 corresponds to oo_score > 0.95 and represents
    # unambiguous orca detections in confirmed MARS 2018 orca events.
    if oo_score >= args.oo_pos_threshold and logit_1 >= args.logit_threshold:
        return label, "positive"

    # --- MODERATE POSITIVE (no confusion runners-up) ---
    if oo_score >= args.oo_pos_threshold and not has_confusion:
        return label, "positive"

    # --- REVIEW (moderate score with confusion runners-up) ---
    # Likely dolphin or generic call — needs human verification via Gradio.
    if oo_score >= args.oo_pos_threshold and has_confusion:
        if args.no_confusion_filter:
            return label, "positive"
        return label, "review"

    # --- NEGATIVE (reliable: low oo, strong whale) ---
    if oo_score <= args.oo_neg_threshold and best_whale_score >= args.whale_pos_threshold:
        return label, "negative"

    # --- SKIP (confusion zone) ---
    return None, None


def classify_generic_row(
    target_class: str,
    scores: dict[str, float],
    args,
) -> tuple[str | None, str | None]:
    """Apply simple threshold labeling for non-orca classes."""
    score = scores.get(target_class, 0.0)
    label = CLASS_LABEL_MAP.get(target_class, target_class)
    if score >= args.generic_class_threshold:
        return label, "positive"
    # Low score = weak negative
    if score <= 0.01:
        return label, "weak_negative"
    return None, None


def process_file(
    csv_path: Path,
    target_classes: list[str],
    args,
) -> tuple[list[dict], list[dict]]:
    """
    Process one _expanded.csv file.
    Returns (label_rows, review_rows).
    """
    rows = parse_expanded_csv(csv_path)
    label_rows  = []
    review_rows = []

    for row in rows:
        scores = extract_class_scores(row)
        top3   = get_top3(row)

        filename_field = row.get("filename", "")
        if not filename_field:
            continue

        recording_id           = recording_id_from_filename(filename_field)
        offset_s, end_offset_s = offset_from_chunk_filename(filename_field)

        # Try to get epoch time from the row index and the oo_scores CSV
        # (expanded CSV uses 5_sec_time_offset as chunk index, not epoch)
        chunk_idx = int(row.get("5_sec_time_offset", 0))

        # Extract logit of top-ranked class (column logits_1 in expanded CSV)
        try:
            logit_1 = float(row.get("logits_1", 0.0))
        except (ValueError, TypeError):
            logit_1 = 0.0

        for tc in target_classes:
            if tc == "oo":
                label, label_type = classify_orca_row(scores, top3, args, logit_1=logit_1)
            else:
                label, label_type = classify_generic_row(tc, scores, args)

            if label is None:
                continue

            out_row = {
                "recording_id":  recording_id,
                "offset_s":      f"{offset_s:.1f}",
                "end_offset_s":  f"{end_offset_s:.1f}",
                "label":         label,
                "label_type":    label_type if label_type != "review" else "positive",
                "chunk_index":   chunk_idx,
                "oo_score":      f"{scores.get('oo', 0.0):.6f}",
                "top3_classes":  "|".join(top3),
                "annotator_id":  args.annotator_id,
                "source_file":   str(csv_path),
            }

            if label_type == "review":
                review_rows.append(out_row)
            else:
                label_rows.append(out_row)

    return label_rows, review_rows


def write_csv(rows: list[dict], output_path: Path) -> None:
    """Write label rows to CSV."""
    if not rows:
        log.warning("No rows to write to %s", output_path)
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "recording_id", "offset_s", "end_offset_s", "label", "label_type",
        "chunk_index", "oo_score", "top3_classes", "annotator_id", "source_file",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    log.info("Wrote %d rows to %s", len(rows), output_path)


def print_summary(label_rows: list[dict], review_rows: list[dict]) -> None:
    """Print a summary of label counts by class and type."""
    from collections import Counter
    log.info("=" * 60)
    log.info("LABEL SUMMARY")
    log.info("=" * 60)
    by_type = Counter((r["label"], r["label_type"]) for r in label_rows)
    for (label, ltype), count in sorted(by_type.items()):
        log.info("  %-30s  %-15s  %6d", label, ltype, count)
    if review_rows:
        review_count = Counter(r["label"] for r in review_rows)
        log.info("  --- Review queue (confusion zone) ---")
        for label, count in sorted(review_count.items()):
            log.info("  %-30s  %-15s  %6d", label, "review", count)
    log.info("  TOTAL labels : %d", len(label_rows))
    log.info("  TOTAL review : %d", len(review_rows))
    log.info("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = build_parser()
    args   = parser.parse_args(argv)
    _setup_logging(args.verbose)

    scores_dir = Path(args.scores_dir)
    if not scores_dir.exists():
        log.error("scores-dir does not exist: %s", scores_dir)
        return 1

    target_classes = (
        list(CLASS_LABEL_MAP.keys()) if args.target_class == "all"
        else [args.target_class]
    )
    log.info("Target classes : %s", target_classes)
    log.info("Scores dir     : %s", scores_dir)
    log.info("Output CSV     : %s", args.output_csv)
    if args.target_class == "oo":
        log.info("Oo pos threshold    : %.2f", args.oo_pos_threshold)
        log.info("Oo neg threshold    : %.2f", args.oo_neg_threshold)
        log.info("Whale pos threshold : %.2f", args.whale_pos_threshold)
        log.info("Confusion filter    : %s", not args.no_confusion_filter)

    csv_files = find_expanded_csvs(scores_dir, args.recursive)
    if not csv_files:
        log.error("No _expanded.csv files found. Check --scores-dir and --recursive.")
        return 1

    all_labels  = []
    all_reviews = []

    for i, csv_path in enumerate(csv_files):
        if i % 100 == 0:
            log.info("Processing file %d / %d ...", i + 1, len(csv_files))
        label_rows, review_rows = process_file(csv_path, target_classes, args)
        all_labels.extend(label_rows)
        all_reviews.extend(review_rows)

    print_summary(all_labels, all_reviews)

    if args.dry_run:
        log.info("[DRY RUN] No files written.")
        return 0

    write_csv(all_labels, Path(args.output_csv))

    if args.review_queue and all_reviews:
        write_csv(all_reviews, Path(args.review_queue))

    log.info("Done. Import labels with:")
    log.info("  python3 phase2_classify.py label \\")
    log.info("      --db-dir <your-db-dir> \\")
    log.info("      --labels-csv %s \\", args.output_csv)
    log.info("      --annotator-id %s", args.annotator_id)

    return 0


if __name__ == "__main__":
    sys.exit(main())


