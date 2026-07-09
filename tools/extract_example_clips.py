#!/usr/bin/env python3
"""extract_example_clips.py
Extract representative 5-second WAV clips from the MARS hydrophone
recordings for use in PyTorch/Perch model porting and testing.

Pulls labeled examples directly from the hoplite SQLite DB:
  - 2 strong orca_call       (highest logit scores from v4_clean inference)
  - 2 strong dolphin_call    (highest logit scores)
  - 2 ship_noise
  - 2 humpback_song
  - 2 background             (lowest logit scores — quiet clips)

Output: 10 WAV files in /mnt/PAM_Analysis/duane_scratch/example_clips/
  e.g. orca_call_01.wav, dolphin_call_01.wav, background_01.wav ...

Usage:
    python3 extract_example_clips.py
"""

import os
import struct
import sqlite3
import csv
import soundfile as sf
import numpy as np

# ── Configuration ──────────────────────────────────────────────────────────
DB_DIR      = "/mnt/PAM_Analysis/duane_scratch/perch_hoplite/db/MARS_20180413_20180413_32kHz"
AUDIO_DIR   = "/mnt/PAM_Analysis/GoogleMultiSpeciesWhaleModel2/resampled_32kHz/2018/04"
APR30_DB    = "/mnt/PAM_Analysis/duane_scratch/perch_hoplite/db/MARS_20180430_20180430_32kHz"
APR30_AUDIO = "/mnt/PAM_Analysis/GoogleMultiSpeciesWhaleModel2/resampled_32kHz/2018/04"
V4_CSV      = "/mnt/PAM_Analysis/duane_scratch/perch_hoplite/results/MARS_20180413_orca_v4_clean_detections.csv"
V4_APR30    = "/mnt/PAM_Analysis/duane_scratch/perch_hoplite/results/MARS_20180430_orca_v4_clean_detections.csv"
OUTPUT_DIR  = "/mnt/PAM_Analysis/duane_scratch/example_clips"

CLIPS = {
    "orca_call":     2,
    "dolphin_call":  2,
    "ship_noise":    2,
    "humpback_song": 2,
    "background":    2,
}
# ──────────────────────────────────────────────────────────────────────────

def get_annotations(db_path, label, n, highest=True):
    """Get top-n annotations by label from DB, ordered by annotation id."""
    con = sqlite3.connect(db_path)
    rows = con.execute("""
        SELECT r.filename, w.offsets, a.label
        FROM annotations a
        JOIN recordings r ON r.id = a.recording_id
        JOIN windows w ON w.recording_id = a.recording_id AND w.offsets = a.offsets
        WHERE a.label = ? AND a.label_type = 1
        ORDER BY a.id DESC
        LIMIT ?
    """, (label, n)).fetchall()
    con.close()
    return rows


def get_top_from_csv(csv_path, label, n, highest=True):
    """Get top-n detections by logit score from inference CSV."""
    rows = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("label") == label:
                rows.append((row["filename"], float(row["window_start"]), float(row["logits"])))
    rows.sort(key=lambda x: x[2], reverse=highest)
    return rows[:n]


def get_background(csv_path, n):
    """Get n clips with lowest logit scores (most background-like)."""
    rows = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append((row["filename"], float(row["window_start"]), float(row["logits"])))
    # Sort by logit ascending — lowest scores are most background-like
    rows.sort(key=lambda x: x[2])
    return rows[:n]


def extract_clip(audio_dir, filename, start_s, output_path):
    """Extract a 5-second clip from a WAV file and save it."""
    filepath = os.path.join(audio_dir, filename)
    if not os.path.exists(filepath):
        print(f"  WARNING: file not found: {filepath}")
        return False
    data, sr = sf.read(filepath)
    start_sample = int(start_s * sr)
    end_sample   = start_sample + int(5.0 * sr)
    clip = data[start_sample:end_sample]
    if len(clip) < int(5.0 * sr):
        # Pad if needed
        clip = np.pad(clip, (0, int(5.0 * sr) - len(clip)))
    sf.write(output_path, clip, sr, subtype="PCM_16")
    duration = len(clip) / sr
    print(f"  Wrote: {os.path.basename(output_path)}  ({duration:.1f}s, {sr}Hz, from {filename} @ {start_s:.1f}s)")
    return True


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}\n")

    counts = {}

    # ── Orca — top scoring from Apr 13 v4_clean inference ────────────────
    print("Extracting orca_call clips...")
    orca_rows = get_top_from_csv(V4_CSV, "orca_call", 2, highest=True)
    for i, (fname, start_s, score) in enumerate(orca_rows, 1):
        out = os.path.join(OUTPUT_DIR, f"orca_call_{i:02d}.wav")
        print(f"  score={score:.3f}")
        extract_clip(AUDIO_DIR, fname, start_s, out)
    counts["orca_call"] = len(orca_rows)

    # ── Dolphin — top scoring from Apr 13 v4_clean inference ─────────────
    print("\nExtracting dolphin_call clips...")
    dolp_rows = get_top_from_csv(V4_CSV, "dolphin_call", 2, highest=True)
    for i, (fname, start_s, score) in enumerate(dolp_rows, 1):
        out = os.path.join(OUTPUT_DIR, f"dolphin_call_{i:02d}.wav")
        print(f"  score={score:.3f}")
        extract_clip(AUDIO_DIR, fname, start_s, out)
    counts["dolphin_call"] = len(dolp_rows)

    # ── Ship noise — from Apr 13 annotations ─────────────────────────────
    print("\nExtracting ship_noise clips...")
    ship_rows = get_annotations(DB_DIR, "ship_noise", 2)
    if len(ship_rows) < 2:
        # Fall back to Apr 30 annotations
        ship_rows = get_annotations(APR30_DB, "ship_noise", 2)
        audio_dir = APR30_AUDIO
    else:
        audio_dir = AUDIO_DIR
    for i, (fname, off_blob, label) in enumerate(ship_rows, 1):
        start_s = struct.unpack_from("<dd", off_blob)[0]
        out = os.path.join(OUTPUT_DIR, f"ship_noise_{i:02d}.wav")
        extract_clip(audio_dir, fname, start_s, out)
    counts["ship_noise"] = len(ship_rows)

    # ── Humpback — from Apr 30 annotations ───────────────────────────────
    print("\nExtracting humpback_song clips...")
    hump_rows = get_annotations(APR30_DB, "humpback_song", 2)
    for i, (fname, off_blob, label) in enumerate(hump_rows, 1):
        start_s = struct.unpack_from("<dd", off_blob)[0]
        out = os.path.join(OUTPUT_DIR, f"humpback_song_{i:02d}.wav")
        extract_clip(APR30_AUDIO, fname, start_s, out)
    counts["humpback_song"] = len(hump_rows)

    # ── Background — lowest scoring from Apr 13 inference ────────────────
    print("\nExtracting background clips...")
    bg_rows = get_background(V4_CSV, 2)
    for i, (fname, start_s, score) in enumerate(bg_rows, 1):
        out = os.path.join(OUTPUT_DIR, f"background_{i:02d}.wav")
        print(f"  score={score:.3f}")
        extract_clip(AUDIO_DIR, fname, start_s, out)
    counts["background"] = len(bg_rows)

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n=== Done ===")
    print(f"Output: {OUTPUT_DIR}")
    for label, n in counts.items():
        print(f"  {label}: {n} clips")
    print(f"\nTotal WAV files: {sum(counts.values())}")
    print("\nThese clips can be used to validate the PyTorch Perch V2 port")
    print("by comparing embeddings against the TF reference implementation.")


if __name__ == "__main__":
    main()
