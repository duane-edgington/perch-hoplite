#!/bin/bash
# reorganize_repo.sh
# Run from ~/perch-hoplite on spark-ae0e.
# Moves remaining loose files into docs/, tools/, scripts/.
# Safe to run — uses || true so already-moved files don't fail.

set -e
cd ~/perch-hoplite

echo "=== Current repo structure ==="
ls *.py *.md *.sh 2>/dev/null || true
echo ""

echo "=== Creating directories ==="
mkdir -p docs tools scripts

echo "=== Moving docs/ ==="
for f in pytorch_port_summary.md october_2020_analysis.md \
          gradio_audio_bug_report.md perch2_pytorch_validation.md \
          PROGRESS_2026-07-09.md REBUILD_PLAN_2026-07-09.md \
          FINDINGS_2026-07-09_tf_parity_and_lowamp_fix.md \
          PATCH_perch_hoplite_torch_adapter.md; do
    [ -f "$f" ] && git mv "$f" docs/ && echo "  moved $f" || true
done

echo "=== Moving tools/ ==="
for f in merge_annotations.py merge_dbs.py plot_monthly.py \
          plot_detections.py plot_tsne.py csv_to_raven.py \
          extract_example_clips.py review_example_clips.py; do
    [ -f "$f" ] && git mv "$f" tools/ && echo "  moved $f" || true
done

echo "=== Moving scripts/ ==="
for f in clean_install.sh reorganize_repo.sh \
          review_oct2020_orca_commands.sh review_may2018_orca_commands.sh \
          prepare_audio_for_colab.sh embed_cpu.sh; do
    [ -f "$f" ] && git mv "$f" scripts/ && echo "  moved $f" || true
done

echo "=== Retiring legacy files ==="
[ -f "phase1_embed.py" ] && \
    git mv phase1_embed.py scripts/phase1_embed_legacy.py && \
    echo "  retired phase1_embed.py" || true
[ -f "phase2_classify_logmel.py" ] && \
    git mv phase2_classify_logmel.py scripts/phase2_classify_logmel_legacy.py && \
    echo "  retired phase2_classify_logmel.py" || true

echo ""
echo "=== Committing ==="
git add .
git status
git commit -m "refactor: reorganize repo — move loose files to docs/, tools/, scripts/" || \
    echo "Nothing to commit — already organized."
git push

echo ""
echo "=== Done ==="
echo "  docs/     — markdown documents and analysis notes"
echo "  tools/    — standalone analysis scripts"
echo "  scripts/  — shell scripts and legacy files"
echo "  src/      — modular Python library"
echo "  phase2_classify.py    — main CLI"
echo "  phase1_embed_torch.py — embedding pipeline"
echo "  README.md, CLAUDE_perch_hoplite.md"
