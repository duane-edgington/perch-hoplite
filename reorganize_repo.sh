#!/bin/bash
# reorganize_repo.sh
# Run from ~/perch-hoplite on spark-ae0e to reorganize repo structure.
# Safe to run multiple times — uses git mv which tracks renames.

set -e
cd ~/perch-hoplite

echo "=== Creating directories ==="
mkdir -p docs tools scripts

echo "=== Moving docs/ ==="
git mv pytorch_port_summary.md     docs/
git mv october_2020_analysis.md    docs/ 2>/dev/null || true
git mv gradio_audio_bug_report.md  docs/ 2>/dev/null || true
git mv perch2_pytorch_validation.md docs/ 2>/dev/null || true

echo "=== Moving tools/ ==="
git mv merge_annotations.py        tools/
git mv merge_dbs.py                tools/
git mv plot_monthly.py             tools/
git mv plot_detections.py          tools/
git mv plot_tsne.py                tools/
git mv csv_to_raven.py             tools/
git mv extract_example_clips.py    tools/ 2>/dev/null || true

echo "=== Moving scripts/ ==="
git mv clean_install.sh            scripts/
git mv review_oct2020_orca_commands.sh scripts/
git mv review_may2018_orca_commands.sh scripts/ 2>/dev/null || true
git mv prepare_audio_for_colab.sh  scripts/ 2>/dev/null || true
git mv embed_cpu.sh                scripts/ 2>/dev/null || true

echo "=== Retiring legacy files ==="
# Keep but mark as legacy — don't delete yet in case anything references them
git mv phase1_embed.py             scripts/phase1_embed_legacy.py 2>/dev/null || true
git mv phase2_classify_logmel.py   scripts/phase2_classify_logmel_legacy.py 2>/dev/null || true

echo "=== Commit ==="
git add .
git commit -m "refactor: reorganize repo — docs/, tools/, scripts/, src/ structure"
git push

echo "=== Done ==="
echo ""
echo "New structure:"
echo "  docs/     — markdown documents and analysis notes"
echo "  tools/    — standalone analysis scripts (merge, plot, convert)"
echo "  scripts/  — shell scripts and legacy files"
echo "  src/      — modular Python library (spectrogram, audio, train, infer, review)"
echo "  phase2_classify.py   — main CLI (thin wrapper over src/)"
echo "  phase1_embed_torch.py — embedding pipeline"
echo "  README.md"
