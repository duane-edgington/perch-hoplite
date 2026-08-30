#!/usr/bin/env python3
"""tools/audit_window_counts.py — verify a month's embedding count against the audio.

WHY THIS EXISTS
---------------
After Stage 2, `phase1_embed_torch.py` prints an "Expected" window count computed as
`len(files) * 120`, which assumes every file is exactly 600 s. On any month containing a
recorder restart that number is WRONG, so a mismatch tells you nothing by itself. This tool
computes the count the adapter should actually produce, per file, and reports every file
that disagrees.

THE WINDOWING RULE (established empirically, Aug 28 2026, August 2015)
----------------------------------------------------------------------
    windows = max(1, floor(duration / 5))

The adapter DROPS the final partial window, but never emits zero windows for a file.
Verified against 3,793 files: 3,772 full-length x 120 + 483 short-file windows = 453,123,
matching the DB exactly.

    NOTE: an earlier version of the notes claimed ceil(). That was wrong. July 2015 has only
    two short files (242 s, 205 s) and happened to match ceil() by coincidence; August 2015,
    with 21 short files, discriminates the rules cleanly. Do not reintroduce ceil().

KNOWN ANOMALY (unresolved): MARS_20150817_155951 (301.0 s) holds 61 windows where the rule
predicts 60, while MARS_20150803_153345 (476.0 s, same 1 s remainder) correctly holds 95.
One file in 3,793. Flagged, not explained. Check exact sample counts with `soxi -s`.

PADDED WINDOWS — worth knowing about
------------------------------------
Because of the max(1, ...) floor, a file SHORTER than one 5 s window still produces one
window, which is mostly silence/padding. August 2015 has three such files (1 s, 2 s, 1 s).
Whatever the classifier scores on those is not meaningful, and nothing stops it scoring
high. This tool lists them so you can decide whether to exclude them at inference.

Usage:
    python3 tools/audit_window_counts.py \\
        --audio-dir /mnt/.../resampled_32kHz/2015/08 \\
        --db /mnt/.../db/MARS_20150801_20150831_32kHz_norm/hoplite.sqlite
"""
import argparse
import math
import sqlite3
import subprocess
import sys
from pathlib import Path

WINDOW_S = 5.0
FULL_S = 600.0
CHUNK = 200


def expected_windows(duration_s, window_s=WINDOW_S):
    """The adapter's rule: drop the final partial window, but never emit zero."""
    return max(1, math.floor(duration_s / window_s))


def durations_for(files):
    out = {}
    for i in range(0, len(files), CHUNK):
        chunk = files[i:i + CHUNK]
        r = subprocess.run(['soxi', '-D'] + [str(f) for f in chunk],
                           capture_output=True, text=True)
        lines = [ln for ln in r.stdout.strip().split('\n') if ln.strip()]
        if len(lines) == len(chunk):
            out.update(dict(zip(chunk, (float(x) for x in lines))))
        else:
            for f in chunk:
                rr = subprocess.run(['soxi', '-D', str(f)],
                                    capture_output=True, text=True)
                out[f] = float(rr.stdout.strip()) if rr.returncode == 0 else 0.0
        print(f"  ...{min(i+CHUNK, len(files))}/{len(files)}", file=sys.stderr, end='\r')
    print(file=sys.stderr)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--audio-dir', required=True)
    ap.add_argument('--db', required=True, help='path to hoplite.sqlite')
    ap.add_argument('--window-s', type=float, default=WINDOW_S)
    ap.add_argument('--show-all', action='store_true',
                    help='list every short file, not just mismatches')
    args = ap.parse_args()

    audio_dir = Path(args.audio_dir)
    files = sorted(audio_dir.glob('*.wav'))
    if not files:
        sys.exit(f"no wav files in {audio_dir}")

    print(f"Reading durations for {len(files)} file(s)...", file=sys.stderr)
    durs = durations_for(files)

    con = sqlite3.connect(args.db)
    db_counts = dict(con.execute("""
        SELECT r.filename, COUNT(w.id)
        FROM recordings r LEFT JOIN windows w ON w.recording_id = r.id
        GROUP BY r.filename
    """).fetchall())
    db_total = con.execute("SELECT COUNT(*) FROM windows").fetchone()[0]
    n_recordings = con.execute("SELECT COUNT(*) FROM recordings").fetchone()[0]
    con.close()

    short, mismatches, padded, missing = [], [], [], []
    predicted_total = 0
    for f in files:
        d = durs.get(f, 0.0)
        pred = expected_windows(d, args.window_s)
        predicted_total += pred
        actual = db_counts.get(f.name)
        if actual is None:
            missing.append(f.name)
            continue
        if abs(d - FULL_S) > 1e-6:
            short.append((f.name, d, pred, actual))
        if d < args.window_s:
            padded.append((f.name, d, actual))
        if actual != pred:
            mismatches.append((f.name, d, pred, actual))

    print(f"\nFiles on disk        : {len(files)}")
    print(f"Recordings in DB     : {n_recordings}")
    print(f"Short files (<600 s) : {len(short)}")
    print(f"\nPredicted windows    : {predicted_total}   [rule: max(1, floor(dur/{args.window_s:g}))]")
    print(f"Windows in DB        : {db_total}")
    diff = db_total - predicted_total
    print(f"Difference           : {diff:+d}")
    print(f"\n(files x 120 would predict {len(files)*120}, "
          f"a {len(files)*120 - predicted_total:+d} error)")

    if missing:
        print(f"\n*** {len(missing)} FILE(S) ON DISK WITH NO DB RECORDING — embed skipped them:")
        for n in missing[:20]:
            print(f"      {n}")

    if mismatches:
        print(f"\n*** {len(mismatches)} FILE(S) DISAGREE WITH THE RULE:")
        for n, d, p, a in mismatches[:30]:
            print(f"      {n:<50} {d:8.1f}s  rule={p:4d}  DB={a:4d}  ({a-p:+d})")
        print("    One-off disagreements are known to occur (see module docstring).")
        print("    Check exact sample counts:  soxi -s <file>")
    else:
        print("\nAll files match the windowing rule.")

    if padded:
        print(f"\nPADDED WINDOWS: {len(padded)} file(s) shorter than one {args.window_s:g} s window,")
        print("  each still contributing 1 mostly-empty window. Scores on these are not")
        print("  meaningful — consider excluding them at inference.")
        for n, d, a in padded:
            print(f"      {n:<50} {d:6.1f}s  -> {a} window(s)")

    if args.show_all and short:
        print(f"\nAll {len(short)} short files:")
        for n, d, p, a in short:
            flag = "" if p == a else "   <-- MISMATCH"
            print(f"      {n:<50} {d:8.1f}s  rule={p:4d}  DB={a:4d}{flag}")

    return 1 if (mismatches or missing) else 0


if __name__ == '__main__':
    sys.exit(main())
