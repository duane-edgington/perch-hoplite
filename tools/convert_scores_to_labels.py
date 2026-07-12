#!/usr/bin/env python3
"""
convert_scores_to_labels.py
============================
Convert Google Multispecies Whale Model _expanded.csv score files into
bootstrap labels for Perch Hoplite agile modeling.

LABEL TIERS
-----------
POSITIVE  : top-class logit >= --logit-threshold (default 3.0)
            OR (Oo score >= --oo-pos-threshold AND no confusion classes in top-3)
REVIEW    : Oo score >= --oo-pos-threshold AND confusion classes in top-3
            AND logit < --logit-threshold
NEGATIVE  : Oo score <= --oo-neg-threshold AND a whale class scores >= --whale-pos-threshold

KEY INSIGHT (from MARS April 13, 2018 orca event):
  Even at logit=8.5 (certainty >0.999), confusion classes (Call, Whistle, Echolocation)
  appear as runners-up alongside Oo. The logit value is the reliable discriminator —
  the confusion filter only applies to ambiguous mid-range scores.

INPUT FORMAT (_expanded.csv columns):
  5_sec_time_offset   - 5-second chunk index (0-based integer)
  all_logits_1..12    - raw logits in ranked order
  all_scores_1..12    - softmax probabilities in ranked order
  class_names_1..12   - class names in ranked order
  filename            - path to source 5-second WAV chunk

OUTPUT FORMAT (Hoplite label CSV):
  recording_id   - source audio filename stem (e.g. MARS_20180413_075913_resampled_24kHz)
  offset_s       - time offset in seconds (5_sec_time_offset * 5)
  label          - class label string (e.g. orca_call)
  label_type     - POSITIVE or NEGATIVE
  source_score   - Oo_class_score from the model
  source_logit   - top-class logit value

USAGE EXAMPLES
--------------
Dry run on known orca event (April 13, 2018):
  python convert_scores_to_labels.py \\
      --scores-dir /tmp/orca_test \\
      --output-csv /tmp/orca_test_labels.csv \\
      --target-class oo \\
      --logit-threshold 3.0 \\
      --oo-pos-threshold 0.5 \\
      --review-queue /tmp/orca_test_review.csv \\
      --dry-run --verbose

Full run on one month:
  python convert_scores_to_labels.py \\
      --scores-dir /mnt/PAM_Analysis/GoogleMultiSpeciesWhaleModel2/scores_gpu/2018/04 \\
      --output-csv /mnt/PAM_Analysis/duane_scratch/perch_hoplite/labels/MARS_2018_04_oo_labels.csv \\
      --target-class oo \\
      --logit-threshold 3.0 \\
      --review-queue /mnt/PAM_Analysis/duane_scratch/perch_hoplite/labels/MARS_2018_04_oo_review.csv

Full run on all available years for orca:
  for year in 2018 2020 2021; do
    for month_dir in /mnt/PAM_Analysis/GoogleMultiSpeciesWhaleModel2/scores_gpu/$year/*/; do
      month=$(basename $month_dir)
      python convert_scores_to_labels.py \\
          --scores-dir $month_dir \\
          --output-csv /mnt/PAM_Analysis/duane_scratch/perch_hoplite/labels/oo_labels_${year}_${month}.csv \\
          --target-class oo \\
          --logit-threshold 3.0 \\
          --review-queue /mnt/PAM_Analysis/duane_scratch/perch_hoplite/labels/oo_review_${year}_${month}.csv
    done
  done
"""

import argparse
import csv
import logging
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Classes whose presence as runners-up indicates dolphin/generic confusion
# NOTE: this filter is ONLY applied when logit < logit_threshold
CONFUSION_CLASSES = {"Call", "Whistle", "Echolocation"}

# Whale species classes — high score here makes a good negative for orca
WHALE_CLASSES = {"Mn", "Bp", "Bm", "Ba", "Be", "Eg", "Upcall"}

# Map target class codes to output label strings
CLASS_CODE_TO_LABEL = {
    "oo": "orca_call",
    "mn": "humpback_call",
    "bp": "fin_whale_call",
    "bm": "blue_whale_call",
}

# ---------------------------------------------------------------------------
# CSV parsing helpers
# ---------------------------------------------------------------------------

def parse_expanded_csv(filepath: Path) -> list[dict]:
    """
    Parse one _expanded.csv file. Returns list of row dicts with keys:
      chunk_idx, offset_s, top_logit, top_score, class_names (list),
      logits (list), scores (list), filename, oo_score
    """
    rows = []
    with open(filepath, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                chunk_idx = int(float(row["5_sec_time_offset"]))
            except (ValueError, KeyError):
                continue

            # Parse ranked class names, logits, scores (up to 12 columns)
            class_names = []
            logits = []
            scores = []
            for i in range(1, 13):
                cn_key = f"class_names_{i}"
                lg_key = f"all_logits_{i}"
                sc_key = f"all_scores_{i}"  # may also be scores_1..12
                # Try both column name variants
                if cn_key not in row:
                    break
                class_names.append(row[cn_key].strip())
                logits.append(float(row.get(lg_key, 0)))
                scores.append(float(row.get(sc_key, row.get(f"scores_{i}", 0))))

            if not class_names:
                continue

            # Find Oo score (may not be rank-1)
            oo_score = 0.0
            for cn, sc in zip(class_names, scores):
                if cn.strip() == "Oo":
                    oo_score = sc
                    break

            rows.append({
                "chunk_idx": chunk_idx,
                "offset_s": chunk_idx * 5,
                "top_logit": logits[0] if logits else 0.0,
                "top_score": scores[0] if scores else 0.0,
                "class_names": class_names,
                "logits": logits,
                "scores": scores,
                "filename": row.get("filename", ""),
                "oo_score": oo_score,
            })

    return rows


def get_oo_score_from_simple_csv(filepath: Path) -> dict[int, float]:
    """
    Parse _epoch_oo_scores.csv (epoch_seconds, Oo_class_score) as fallback.
    Returns dict of {chunk_index: score}.  Not used by default but available.
    """
    result = {}
    with open(filepath, newline="") as fh:
        reader = csv.DictReader(fh)
        for i, row in enumerate(reader):
            try:
                score = float(list(row.values())[1])
                result[i] = score
            except (ValueError, IndexError):
                continue
    return result


# ---------------------------------------------------------------------------
# Classification logic
# ---------------------------------------------------------------------------

def classify_row(
    row: dict,
    target_code: str,
    logit_threshold: float,
    oo_pos_threshold: float,
    oo_neg_threshold: float,
    whale_pos_threshold: float,
    use_confusion_filter: bool,
) -> str:
    """
    Returns 'positive', 'negative', 'review', or 'skip'.

    Decision logic:
    1. If top-class logit >= logit_threshold  →  POSITIVE  (high-confidence, skip confusion filter)
    2. If oo_score >= oo_pos_threshold:
       a. No confusion classes in top-3     →  POSITIVE
       b. Confusion classes in top-3        →  REVIEW
    3. If oo_score <= oo_neg_threshold AND any whale class score >= whale_pos_threshold → NEGATIVE
    4. Otherwise → SKIP
    """
    oo_score = row["oo_score"]
    top_logit = row["top_logit"]
    top3_classes = set(cn.strip() for cn in row["class_names"][:3])

    # Rule 1: high-confidence logit overrides everything
    if top_logit >= logit_threshold and row["class_names"][0].strip() == "Oo":
        return "positive"

    # Rule 2: score-based positive / review
    if oo_score >= oo_pos_threshold:
        if use_confusion_filter and (top3_classes & CONFUSION_CLASSES):
            return "review"
        return "positive"

    # Rule 3: reliable negative — strong whale detection, very low orca score
    if oo_score <= oo_neg_threshold:
        # Check if any whale class scores high
        whale_score = 0.0
        for cn, sc in zip(row["class_names"], row["scores"]):
            if cn.strip() in WHALE_CLASSES:
                whale_score = max(whale_score, sc)
        if whale_score >= whale_pos_threshold:
            return "negative"

    return "skip"


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

def process_file(
    expanded_csv: Path,
    target_code: str,
    logit_threshold: float,
    oo_pos_threshold: float,
    oo_neg_threshold: float,
    whale_pos_threshold: float,
    use_confusion_filter: bool,
    verbose: bool,
) -> tuple[list[dict], list[dict]]:
    """
    Process one _expanded.csv. Returns (labels, review_items).
    Each label dict has: recording_id, offset_s, label, label_type, source_score, source_logit
    """
    label_str = CLASS_CODE_TO_LABEL.get(target_code, f"{target_code}_call")

    # Derive recording_id from filename stem, stripping _expanded suffix
    stem = expanded_csv.stem  # e.g. MARS_20180413_075913_resampled_24kHz_expanded
    recording_id = stem.replace("_expanded", "")

    rows = parse_expanded_csv(expanded_csv)
    if not rows:
        log.warning("No data rows parsed from %s", expanded_csv.name)
        return [], []

    labels = []
    review_items = []

    for row in rows:
        decision = classify_row(
            row,
            target_code=target_code,
            logit_threshold=logit_threshold,
            oo_pos_threshold=oo_pos_threshold,
            oo_neg_threshold=oo_neg_threshold,
            whale_pos_threshold=whale_pos_threshold,
            use_confusion_filter=use_confusion_filter,
        )

        if decision in ("positive", "negative"):
            labels.append({
                "recording_id": recording_id,
                "offset_s": row["offset_s"],
                "label": label_str,
                "label_type": decision.upper(),
                "source_score": f"{row['oo_score']:.4f}",
                "source_logit": f"{row['top_logit']:.4f}",
                "top3_classes": "|".join(row["class_names"][:3]),
            })
            if verbose:
                log.debug(
                    "  chunk %4d  offset %4ds  oo=%.3f  logit=%6.2f  top3=[%s]  → %s",
                    row["chunk_idx"], row["offset_s"], row["oo_score"],
                    row["top_logit"], ",".join(row["class_names"][:3]), decision.upper()
                )
        elif decision == "review":
            review_items.append({
                "recording_id": recording_id,
                "offset_s": row["offset_s"],
                "label": label_str,
                "source_score": f"{row['oo_score']:.4f}",
                "source_logit": f"{row['top_logit']:.4f}",
                "top3_classes": "|".join(row["class_names"][:3]),
            })

    return labels, review_items


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def find_expanded_csvs(scores_dir: Path) -> list[Path]:
    """
    Find all _expanded.csv files under scores_dir, recursing one level into
    subdirectories (handles both flat and year/month/subdir layouts).
    """
    found = []
    # Direct files in scores_dir
    found.extend(sorted(scores_dir.glob("*_expanded.csv")))
    # One level of subdirectories
    for subdir in sorted(scores_dir.iterdir()):
        if subdir.is_dir():
            found.extend(sorted(subdir.glob("*_expanded.csv")))
    return found


def write_csv(rows: list[dict], filepath: Path, fieldnames: list[str]) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def print_summary(all_labels: list[dict], all_review: list[dict], label_str: str) -> None:
    pos = [r for r in all_labels if r["label_type"] == "POSITIVE"]
    neg = [r for r in all_labels if r["label_type"] == "NEGATIVE"]
    log.info("=" * 60)
    log.info("LABEL SUMMARY")
    log.info("=" * 60)
    log.info("  %-35s %-10s %6d", label_str, "POSITIVE", len(pos))
    log.info("  %-35s %-10s %6d", label_str, "NEGATIVE", len(neg))
    log.info("  %-35s %-10s %6d", "--- Review queue (confusion zone) ---", "", len(all_review))
    log.info("  TOTAL labels : %d", len(all_labels))
    log.info("  TOTAL review : %d", len(all_review))
    log.info("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Google Multispecies Whale Model scores to Hoplite bootstrap labels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--scores-dir", required=True, type=Path,
                        help="Directory containing *_expanded.csv files (searched recursively one level).")
    parser.add_argument("--output-csv", required=True, type=Path,
                        help="Output CSV of POSITIVE/NEGATIVE labels.")
    parser.add_argument("--review-queue", type=Path, default=None,
                        help="Output CSV of REVIEW items (confusion zone) for manual inspection.")
    parser.add_argument("--target-class", default="oo",
                        choices=list(CLASS_CODE_TO_LABEL.keys()),
                        help="Which species class to generate labels for.")
    parser.add_argument("--logit-threshold", type=float, default=3.0,
                        help="Top-class logit >= this → POSITIVE regardless of confusion filter. "
                             "Use 3.0 for orca (confirmed via April 2018 MARS event analysis).")
    parser.add_argument("--oo-pos-threshold", type=float, default=0.5,
                        help="Oo softmax score >= this triggers positive/review evaluation.")
    parser.add_argument("--oo-neg-threshold", type=float, default=0.05,
                        help="Oo softmax score <= this enables negative evaluation.")
    parser.add_argument("--whale-pos-threshold", type=float, default=0.30,
                        help="Whale class score >= this (combined with low oo) → NEGATIVE.")
    parser.add_argument("--no-confusion-filter", action="store_true",
                        help="Disable confusion class filter entirely (label all Oo >= threshold as positive).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Process files and print summary but write no output files.")
    parser.add_argument("--verbose", action="store_true",
                        help="Log each labeled chunk (debug level).")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    use_confusion_filter = not args.no_confusion_filter
    label_str = CLASS_CODE_TO_LABEL.get(args.target_class, f"{args.target_class}_call")

    log.info("Target classes : ['%s']  →  label '%s'", args.target_class, label_str)
    log.info("Scores dir     : %s", args.scores_dir)
    log.info("Output CSV     : %s", args.output_csv)
    log.info("Logit threshold     : %.2f  (overrides confusion filter when top logit >= this)", args.logit_threshold)
    log.info("Oo pos threshold    : %.2f", args.oo_pos_threshold)
    log.info("Oo neg threshold    : %.2f", args.oo_neg_threshold)
    log.info("Whale pos threshold : %.2f", args.whale_pos_threshold)
    log.info("Confusion filter    : %s", use_confusion_filter)

    expanded_csvs = find_expanded_csvs(args.scores_dir)
    if not expanded_csvs:
        log.error("No *_expanded.csv files found under %s", args.scores_dir)
        sys.exit(1)
    log.info("Found %d _expanded.csv files in %s", len(expanded_csvs), args.scores_dir)

    all_labels: list[dict] = []
    all_review: list[dict] = []

    for i, csv_path in enumerate(expanded_csvs, 1):
        log.info("Processing file %d / %d : %s", i, len(expanded_csvs), csv_path.name)
        labels, review = process_file(
            expanded_csv=csv_path,
            target_code=args.target_class,
            logit_threshold=args.logit_threshold,
            oo_pos_threshold=args.oo_pos_threshold,
            oo_neg_threshold=args.oo_neg_threshold,
            whale_pos_threshold=args.whale_pos_threshold,
            use_confusion_filter=use_confusion_filter,
            verbose=args.verbose,
        )
        all_labels.extend(labels)
        all_review.extend(review)
        pos = sum(1 for r in labels if r["label_type"] == "POSITIVE")
        neg = sum(1 for r in labels if r["label_type"] == "NEGATIVE")
        log.info("  → %d positive, %d negative, %d review", pos, neg, len(review))

    print_summary(all_labels, all_review, label_str)

    if args.dry_run:
        log.info("[DRY RUN] No files written.")
        return

    label_fields = ["recording_id", "offset_s", "label", "label_type",
                    "source_score", "source_logit", "top3_classes"]
    write_csv(all_labels, args.output_csv, label_fields)
    log.info("Wrote %d labels to %s", len(all_labels), args.output_csv)

    if args.review_queue and all_review:
        review_fields = ["recording_id", "offset_s", "label",
                         "source_score", "source_logit", "top3_classes"]
        write_csv(all_review, args.review_queue, review_fields)
        log.info("Wrote %d review items to %s", len(all_review), args.review_queue)


if __name__ == "__main__":
    main()
