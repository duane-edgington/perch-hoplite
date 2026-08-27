#!/usr/bin/env python3
"""tools/register_figure.py
Register a figure into the perch-hoplite figures provenance system.

Creates a per-figure JSON sidecar file (e.g. figures/gradio_30s_context_orca2.json)
and updates the master manifest (figures/manifest.json).

Usage — Gradio screenshot:
    python3 tools/register_figure.py \\
        --saved-name gradio_30s_context_orca2.png \\
        --original-name "Screenshot_2026-07-12_at_9_04_41_AM.png" \\
        --computer DuaneEM1 \\
        --type gradio_screenshot \\
        --wav MARS_20180413_075913_resampled_32kHz.wav \\
        --offset 370 \\
        --spectrogram linear \\
        --colormap inferno \\
        --classifier orca_v2.pt \\
        --db MARS_20180401_20180430_32kHz_norm \\
        --score 3.308 \\
        --label orca_call \\
        --caption "Orca call at peak of April 13 2018 event..." \\
        --command "nohup python3 phase2_classify.py review --db-dir ... --port 7862"

Usage — matplotlib plot (t-SNE, monthly, heatmap):
    python3 tools/register_figure.py \\
        --saved-name tsne_apr2018_oct2020_apr2026_norm.png \\
        --original-name tsne_apr2018_oct2020_apr2026_norm.png \\
        --computer spark-ae0e \\
        --type tsne_plot \\
        --caption "823 labeled embeddings from three seasons..." \\
        --command "python3 tools/plot_tsne.py --db-dir ... --output ..."
"""
import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

FIGURES_DIR = Path(__file__).parent.parent / "figures"
MANIFEST    = FIGURES_DIR / "manifest.json"


def parse_timestamp_from_filename(original_name: str) -> str | None:
    """Extract capture timestamp from macOS screenshot filename.

    Handles formats like:
      Screenshot_2026-07-12_at_9_04_41_AM.png
      Screenshot 2026-07-12 at 9.04.41 AM.png
    """
    # Screenshot_2026-07-12_at_9_04_41_AM
    m = re.search(
        r'(\d{4})-(\d{2})-(\d{2})_at_(\d+)_(\d{2})_(\d{2})_(AM|PM)',
        original_name)
    if m:
        yr, mo, dy, hr, mn, sc, ampm = m.groups()
        hr = int(hr)
        if ampm == 'PM' and hr != 12:
            hr += 12
        elif ampm == 'AM' and hr == 12:
            hr = 0
        return f"{yr}-{mo}-{dy} {hr:02d}:{mn}:{sc}"

    # Screenshot 2026-07-12 at 9.04.41 AM
    m = re.search(
        r'(\d{4})-(\d{2})-(\d{2}) at (\d+)\.(\d{2})\.(\d{2}) (AM|PM)',
        original_name)
    if m:
        yr, mo, dy, hr, mn, sc, ampm = m.groups()
        hr = int(hr)
        if ampm == 'PM' and hr != 12:
            hr += 12
        elif ampm == 'AM' and hr == 12:
            hr = 0
        return f"{yr}-{mo}-{dy} {hr:02d}:{mn}:{sc}"

    return None


def load_manifest() -> dict:
    if MANIFEST.exists():
        with open(MANIFEST) as f:
            return json.load(f)
    return {
        "description": "Master figure manifest for perch-hoplite repo",
        "figures": {}
    }


def save_manifest(manifest: dict):
    with open(MANIFEST, 'w') as f:
        json.dump(manifest, f, indent=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # Required
    ap.add_argument("--saved-name",    required=True,
                    help="Filename as saved in figures/ (e.g. gradio_30s_context_orca2.png)")
    ap.add_argument("--original-name", required=True,
                    help="Original filename as uploaded (preserves macOS timestamp)")
    ap.add_argument("--computer",      required=True,
                    choices=["ICEFISH", "PERCH", "DuaneEM1", "spark-ae0e", "spark-0626", "other"],
                    help="Computer where screenshot/plot was captured")
    ap.add_argument("--type",          required=True,
                    choices=["gradio_screenshot", "tsne_plot", "monthly_plot",
                             "heatmap_plot", "matplotlib_plot", "other"],
                    help="Type of figure")
    ap.add_argument("--caption",       required=True,
                    help="Figure caption text (used in README/poster)")
    ap.add_argument("--command",       required=True,
                    help="Full command used to generate the figure")
    ap.add_argument("--script",        default=None,
                    help="Path to the generating script, e.g. tools/plot_tsne.py. "
                         "Required for --type tsne_plot/matplotlib_plot (code-generated "
                         "figures) -- if omitted for those types, a best-effort guess is "
                         "extracted from --command and a warning is printed, since --command "
                         "is free text and not reliably parseable. Not required for "
                         "gradio_screenshot (a UI capture, not a script run).")

    # Gradio-specific (optional for plots)
    ap.add_argument("--wav",        default=None, help="Source WAV filename")
    ap.add_argument("--offset",     type=float, default=None, help="Offset in seconds")
    ap.add_argument("--spectrogram",default=None,
                    choices=["linear", "mel", "perch", "pcen"],
                    help="Spectrogram type")
    ap.add_argument("--colormap",   default=None, help="Colormap (viridis, gray, inferno...)")
    ap.add_argument("--classifier", default=None, help="Classifier model filename")
    ap.add_argument("--db",         default=None, help="Database name")
    ap.add_argument("--score",      type=float, default=None, help="Classifier score")
    ap.add_argument("--label",      default=None, help="Label selected in Gradio")
    ap.add_argument("--notes",      default=None, help="Additional notes")

    args = ap.parse_args()

    # Validate figure file exists
    fig_path = FIGURES_DIR / args.saved_name
    if not fig_path.exists():
        print(f"ERROR: {fig_path} does not exist. Copy the file to figures/ first.")
        sys.exit(1)

    # Resolve --script: use it if given; otherwise best-effort extract from --command for
    # code-generated figure types, since --command is free text and inconsistently written
    # (this project has repeatedly hit confusion from not knowing which exact script/version
    # made a given figure -- see CLAUDE_perch_hoplite.md history, Aug 2026).
    script_value = args.script
    if not script_value and args.type in ("tsne_plot", "matplotlib_plot", "monthly_plot", "heatmap_plot"):
        m = re.search(r"(tools/\S+\.py|\S+\.py)", args.command)
        if m:
            script_value = m.group(1)
            print(f"NOTE: --script not given; extracted '{script_value}' from --command. "
                  f"Pass --script explicitly next time to avoid relying on free-text parsing.")
        else:
            print(f"WARNING: --script not given and no .py filename found in --command. "
                  f"This figure's sidecar will NOT clearly record which script generated it.")

    # Parse timestamp from original filename
    capture_ts = parse_timestamp_from_filename(args.original_name)
    if capture_ts:
        print(f"Parsed capture timestamp: {capture_ts}")
    else:
        capture_ts = "unknown"
        print(f"WARNING: Could not parse timestamp from '{args.original_name}'")

    # Build provenance record
    stem = Path(args.saved_name).stem
    record = {
        "saved_filename":    args.saved_name,
        "original_filename": args.original_name,
        "capture_computer":  args.computer,
        "capture_timestamp": capture_ts,
        "date_registered":   datetime.now().isoformat(),
        "figure_type":       args.type,
        "caption":           args.caption,
        "command":           args.command,
        "script":            script_value,
    }

    # Gradio-specific fields
    if args.wav:        record["source_wav"]       = args.wav
    if args.offset is not None: record["wav_offset_s"]  = args.offset
    if args.spectrogram: record["spectrogram_type"] = args.spectrogram
    if args.colormap:   record["colormap"]          = args.colormap
    if args.classifier: record["classifier"]        = args.classifier
    if args.db:         record["db"]                = args.db
    if args.score is not None: record["score"]      = args.score
    if args.label:      record["label_selected"]    = args.label
    if args.notes:      record["notes"]             = args.notes

    # Write per-figure JSON sidecar
    sidecar = FIGURES_DIR / f"{stem}.json"
    with open(sidecar, 'w') as f:
        json.dump(record, f, indent=2)
    print(f"Written: {sidecar}")

    # Update master manifest
    manifest = load_manifest()
    manifest["figures"][args.saved_name] = record
    manifest["last_updated"] = datetime.now().isoformat()
    manifest["figure_count"] = len(manifest["figures"])
    save_manifest(manifest)
    print(f"Updated: {MANIFEST} ({manifest['figure_count']} figures)")
    print()
    print("Caption stored:")
    print(f"  {args.caption[:80]}{'...' if len(args.caption) > 80 else ''}")


if __name__ == "__main__":
    main()
