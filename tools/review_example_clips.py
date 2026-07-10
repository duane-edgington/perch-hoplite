#!/usr/bin/env python3
"""review_example_clips.py
Launch a Gradio review session for the 10 example clips in
/mnt/PAM_Analysis/perch-hoplite/example_clips/

Uses the original source filenames from manifest.json so the review
command can match them against the DB.

Usage:
    cd ~/perch-hoplite
    source venv/bin/activate
    python3 tools/review_example_clips.py [--port 7862]

Open in Chrome (incognito): http://134.89.11.107:7862
"""
import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

PERCH_ROOT   = Path("/mnt/PAM_Analysis/perch-hoplite")
EXAMPLE_DIR  = PERCH_ROOT / "example_clips"
MANIFEST     = EXAMPLE_DIR / "manifest.json"
DETECTIONS   = EXAMPLE_DIR / "example_detections.csv"
DB_DIR       = PERCH_ROOT / "db" / "MARS_20180401_20180430_32kHz_norm"
CLASSIFIER   = PERCH_ROOT / "models" / "orca_v2.pt"
AUDIO_DIR    = Path("/mnt/PAM_Analysis/GoogleMultiSpeciesWhaleModel2/resampled_32kHz/2018/04")
LOG_FILE     = PERCH_ROOT / "logs" / "review_examples.log"

CLASSES = "orca_call,humpback_song,dolphin_call,ship_noise,other,unlabeled"


def make_detections_csv():
    """Generate example_detections.csv using original source filenames."""
    if not MANIFEST.exists():
        print(f"ERROR: manifest not found at {MANIFEST}")
        print("Run tools/extract_example_clips.py first.")
        sys.exit(1)

    manifest = json.loads(MANIFEST.read_text())
    clips    = manifest["clips"]

    with open(DETECTIONS, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "project", "filename",
                    "window_start", "window_end", "label", "logits"])
        for i, clip in enumerate(clips):
            # Use original source filename so review can match against DB
            w.writerow([
                i,
                "MARS_20180401_20180430_32kHz_norm",
                clip["source_file"],          # original MARS filename
                clip["offset_s"],
                clip["offset_s"] + 5.0,
                clip["label"],
                1.0,
            ])
    print(f"Wrote {len(clips)}-row detections CSV: {DETECTIONS}")
    return len(clips)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=7862,
                    help="Gradio server port (default: 7862)")
    ap.add_argument("--classifier", default=str(CLASSIFIER),
                    help="Classifier .pt file")
    args = ap.parse_args()

    n = make_detections_csv()

    cmd = [
        sys.executable, "phase2_classify.py", "review",
        "--db-dir",          str(DB_DIR),
        "--classifier",      args.classifier,
        "--target-label",    "orca_call",
        "--detections-csv",  str(DETECTIONS),
        "--num-results",     str(n),
        "--classes",         CLASSES,
        "--audio-dir",       str(AUDIO_DIR),
        "--serve",
        "--port",            str(args.port),
    ]

    print()
    print(f"Launching Gradio review for {n} example clips...")
    print(f"  Port    : {args.port}")
    print(f"  Audio   : {AUDIO_DIR}")
    print(f"  DB      : {DB_DIR}")
    print(f"  Log     : {LOG_FILE}")
    print()
    print(f"Open in Chrome (incognito): http://134.89.11.107:{args.port}")
    print("Press Ctrl+C to stop.")
    print()

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "w") as log:
        subprocess.run(cmd, stdout=log, stderr=log)


if __name__ == "__main__":
    main()
