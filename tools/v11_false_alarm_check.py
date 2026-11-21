#!/usr/bin/env python3
"""
v11_false_alarm_check.py

Read-only check of orca_v11 on the frozen April and May 2018 months (never written to).

For each class, at that class's operating threshold (from orca_v11.metrics.json, opt_threshold):
  - labeled positives (label_type 1 in the database): recall and precision of the class;
  - unlabeled windows: detections are a false-alarm PROXY (an upper bound, because some unlabeled
    windows may be real calls that were never reviewed).

Also writes a sperm_whale candidate review CSV: unlabeled windows the model scored as sperm_whale,
highest first, in the review format used by phase2_classify.py review.

The database is opened read-only. Run the review in Gradio only on the scratch copies.

Usage (on spark-ae0e):
  python3 v11_false_alarm_check.py \
      --metrics /mnt/PAM_Analysis/perch-hoplite/models/orca_v11.metrics.json \
      --month apr2018 --db /mnt/PAM_Analysis/perch-hoplite/db/MARS_20180401_20180430_32kHz_norm \
      --csv /mnt/PAM_Analysis/perch-hoplite/results/MARS_20180401_20180430_v11_all.csv \
      --sperm-out /mnt/PAM_Analysis/perch-hoplite/results/sperm_candidates_apr2018.csv --sperm-n 60
"""

import argparse
import csv
import json
import re
import sqlite3
import struct
from collections import defaultdict

STEM_RE = re.compile(r"(MARS_\d{8}_\d{6})")
CLASSES = ["orca_call", "humpback_song", "dolphin_call", "other", "ship_noise", "ROV_noise", "sperm_whale"]


def first_offset(blob):
    return struct.unpack("<d", bytes(blob[:8]))[0] if blob else 0.0


def load_labels(db):
    con = sqlite3.connect(f"file:{db}/hoplite.sqlite?mode=ro", uri=True)
    pos, anyl = {}, set()
    for fn, blob, lab, lt in con.execute(
            "SELECT r.filename, a.offsets, a.label, a.label_type FROM annotations a JOIN recordings r ON r.id = a.recording_id"):
        m = STEM_RE.search(fn)
        if not m:
            continue
        k = (m.group(1), round(first_offset(blob), 1))
        anyl.add(k)
        if lt == 1:
            pos.setdefault(k, set()).add(lab)
    con.close()
    return pos, anyl


def main():
    ap = argparse.ArgumentParser(description="Read-only v11 check on a frozen 2018 month.")
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--month", required=True)
    ap.add_argument("--db", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--sperm-out")
    ap.add_argument("--sperm-n", type=int, default=60)
    args = ap.parse_args()

    metrics = json.load(open(args.metrics))
    per = metrics["eval_scores"]["per_class_f1"]["per_class"]
    thr = {c: per[c]["opt_threshold"] for c in CLASSES if c in per}

    pos, anyl = load_labels(args.db)

    stats = defaultdict(lambda: {"tp": 0, "fp_lab": 0, "lab_pos": 0, "unl_det": 0, "unl_windows": set()})
    sperm = []
    total_windows = set()
    for r in csv.DictReader(open(args.csv, newline="")):
        m = STEM_RE.search(r["filename"])
        if not m:
            continue
        k = (m.group(1), round(float(r["window_start"]), 1))
        total_windows.add(k)
        c, score = r["label"], float(r["logits"])
        if c not in thr or score < thr[c]:
            continue
        if k in anyl:
            truth = pos.get(k, set())
            if c in truth:
                stats[c]["tp"] += 1
            else:
                stats[c]["fp_lab"] += 1
        else:
            stats[c]["unl_det"] += 1
            stats[c]["unl_windows"].add(k)
            if c == "sperm_whale":
                sperm.append((score, r["filename"], r["window_start"], r["window_end"]))

    for k, labs in pos.items():
        for c in labs:
            if c in stats or c in thr:
                stats[c]["lab_pos"] += 1

    print(f"month {args.month}: windows in CSV {len(total_windows):,}; labeled windows {len(anyl):,}; "
          f"positive-labeled windows {len(pos):,}")
    print(f"{'class':15s} {'op_thr':>7s} {'TP':>5s} {'lab_pos':>8s} {'recall':>7s} {'prec_lab':>9s} {'FP_on_labeled':>14s} {'unlabeled_dets':>15s} {'unl_windows':>12s}")
    for c in CLASSES:
        if c not in thr:
            continue
        s = stats[c]
        recall = s["tp"] / s["lab_pos"] if s["lab_pos"] else float("nan")
        det_lab = s["tp"] + s["fp_lab"]
        prec = s["tp"] / det_lab if det_lab else float("nan")
        print(f"{c:15s} {thr[c]:7.3f} {s['tp']:5d} {s['lab_pos']:8d} {recall:7.3f} {prec:9.3f} "
              f"{s['fp_lab']:14d} {s['unl_det']:15d} {len(s['unl_windows']):12d}")
    print("note: recall and precision use positives only; unlabeled detections are an upper bound on false alarms.")

    if args.sperm_out:
        sperm.sort(reverse=True)
        with open(args.sperm_out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["idx", "project", "filename", "window_start", "window_end", "label", "logits"])
            for i, (score, fn, ws, we) in enumerate(sperm[: args.sperm_n], 1):
                w.writerow([i, args.month, fn, ws, we, "sperm_whale_candidate", f"{score:.6f}"])
        print(f"wrote {min(len(sperm), args.sperm_n)} sperm_whale candidates (of {len(sperm)} unlabeled detections) to {args.sperm_out}")


if __name__ == "__main__":
    main()
