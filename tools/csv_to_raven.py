#!/usr/bin/env python3
"""csv_to_raven.py
Convert perch-hoplite inference CSV to Raven Pro selection table format.

Raven Pro selection tables are tab-delimited text files with these columns:
  Selection  View  Channel  Begin Time (s)  End Time (s)
  Low Freq (Hz)  High Freq (Hz)  Begin File  Score  Label

Usage:
    python3 csv_to_raven.py \
        --input  /path/to/detections.csv \
        --output /path/to/raven_selections.txt \
        [--low-freq 500] [--high-freq 6000] [--min-score 0.0]

Example — v1_clean:
    python3 csv_to_raven.py \
        --input  /mnt/PAM_Analysis/duane_scratch/perch_hoplite/results/MARS_20180413_orca_v1_clean_detections.csv \
        --output /mnt/PAM_Analysis/duane_scratch/perch_hoplite/results/MARS_20180413_orca_v1_clean_raven.txt

Example — v2_clean:
    python3 csv_to_raven.py \
        --input  /mnt/PAM_Analysis/duane_scratch/perch_hoplite/results/MARS_20180413_orca_v2_clean_detections.csv \
        --output /mnt/PAM_Analysis/duane_scratch/perch_hoplite/results/MARS_20180413_orca_v2_clean_raven.txt
"""

import argparse
import csv
import os


# Orca call frequency range for the selection box in Raven
DEFAULT_LOW_FREQ  = 500    # Hz
DEFAULT_HIGH_FREQ = 6000   # Hz


def convert(input_csv: str, output_txt: str,
            low_freq: float, high_freq: float,
            min_score: float, audio_dir: str | None):

    rows = []
    with open(input_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            score = float(row["logits"])
            if score < min_score:
                continue
            rows.append(row)

    print(f"Loaded {len(rows)} detections from {os.path.basename(input_csv)}")

    # Raven Pro selection table columns
    headers = [
        "Selection",
        "View",
        "Channel",
        "Begin Time (s)",
        "End Time (s)",
        "Low Freq (Hz)",
        "High Freq (Hz)",
        "Begin File",
        "Score",
        "Label",
    ]

    with open(output_txt, "w", newline="") as f:
        f.write("\t".join(headers) + "\n")

        for i, row in enumerate(rows, start=1):
            filename = row["filename"]
            begin_s  = float(row["window_start"])
            end_s    = float(row["window_end"])
            score    = float(row["logits"])
            label    = row["label"]

            # If audio_dir provided, use full path; otherwise just filename
            if audio_dir:
                begin_file = os.path.join(audio_dir, filename)
            else:
                begin_file = filename

            cols = [
                str(i),          # Selection number
                "Spectrogram 1", # View
                "1",             # Channel
                f"{begin_s:.3f}",
                f"{end_s:.3f}",
                f"{low_freq:.1f}",
                f"{high_freq:.1f}",
                begin_file,
                f"{score:.4f}",
                label,
            ]
            f.write("\t".join(cols) + "\n")

    print(f"Wrote {len(rows)} selections to {output_txt}")
    print(f"  Begin Time range: "
          f"{float(rows[0]['window_start']):.1f}s – "
          f"{float(rows[-1]['window_start']):.1f}s")
    print(f"  Score range: "
          f"{min(float(r['logits']) for r in rows):.3f} – "
          f"{max(float(r['logits']) for r in rows):.3f}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--input",     required=True, help="Detections CSV from phase2_classify.py infer")
    ap.add_argument("--output",    required=True, help="Output Raven Pro selection table (.txt)")
    ap.add_argument("--low-freq",  type=float, default=DEFAULT_LOW_FREQ,
                    help=f"Low frequency bound for selection box Hz (default: {DEFAULT_LOW_FREQ})")
    ap.add_argument("--high-freq", type=float, default=DEFAULT_HIGH_FREQ,
                    help=f"High frequency bound for selection box Hz (default: {DEFAULT_HIGH_FREQ})")
    ap.add_argument("--min-score", type=float, default=0.0,
                    help="Minimum logit score to include (default: 0.0)")
    ap.add_argument("--audio-dir", default=None,
                    help="Prepend this directory path to filenames in Begin File column")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    convert(args.input, args.output,
            args.low_freq, args.high_freq,
            args.min_score, args.audio_dir)


if __name__ == "__main__":
    main()
