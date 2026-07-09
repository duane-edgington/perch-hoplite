#!/bin/bash
# clean_install.sh
# =============================================================================
# Unified environment setup for the MBARI perch-hoplite marine bioacoustics
# pipeline on spark-ae0e (NVIDIA GB10 DGX, aarch64, CUDA 13, sm_121).
#
# Creates ONE venv at ~/perch-hoplite/venv that supports the ENTIRE workflow:
#
#   EMBEDDING (phase1_embed_torch.py)
#     Uses the native PyTorch Perch V2 model from ~/perch-pytorch.
#     No TensorFlow. No Colab. No Google Drive.
#
#   INFERENCE / TRAINING / ANNOTATION (phase2_classify.py, phase2_classify_logmel.py)
#     Reads pre-computed embeddings from hoplite DB.
#     Trains linear classifiers. Serves Gradio annotation GUI.
#     No TensorFlow required — PyTorch Perch V2 used for all model calls.
#
# Activate with:
#   source ~/perch-hoplite/venv/bin/activate
#
# Prerequisites:
#   ~/perch-pytorch/perch_weights/weights.npz       (Perch V2 backbone weights)
#   ~/perch-pytorch/perch_weights/graph_manifest.json
#   ~/perch-pytorch/const__pad1_output_0.npy        (exact mel reference)
#   ~/perch-pytorch/perch_hoplite_torch_adapter.py
#   ~/perch-pytorch/perch_embedder_torch.py
#   ~/perch-pytorch/perch_frontend_torch.py
#
# Usage:
#   cd ~/perch-hoplite
#   bash clean_install.sh
# =============================================================================

set -e   # exit on first error

cd ~/perch-hoplite

echo "=== Creating venv at ~/perch-hoplite/venv ==="
python3 -m venv venv
source venv/bin/activate
python --version              # expect 3.12.3
which python                  # expect ~/perch-hoplite/venv/bin/python

# =============================================================================
# Section 1 — PyTorch (required for Perch V2 embedding + model inference)
# Used by: phase1_embed_torch.py, phase2_classify.py (load_model_from_db)
# =============================================================================
echo ""
echo "=== [1/6] PyTorch (CUDA 13 / GB10 / sm_121) ==="
pip3 install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu130

# =============================================================================
# Section 2 — Core scientific stack
# Used by: all scripts
# =============================================================================
echo ""
echo "=== [2/6] Core scientific stack ==="
pip3 install numpy scipy pandas matplotlib

# =============================================================================
# Section 3 — Audio processing
# Used by: phase1_embed_torch.py (soundfile), phase2_classify_logmel.py (librosa PCEN)
# =============================================================================
echo ""
echo "=== [3/6] Audio processing ==="
pip3 install soundfile librosa

# =============================================================================
# Section 4 — ML utilities
# Used by: perch_embedder_torch.py (timm EfficientNet backbone)
#          perch_hoplite_torch_adapter.py (ml_collections)
# =============================================================================
echo ""
echo "=== [4/6] ML utilities ==="
pip3 install timm ml_collections

# =============================================================================
# Section 5 — perch-hoplite (core only — NO TensorFlow, NO JAX)
# Used by: phase1_embed_torch.py, phase2_classify.py
# IMPORTANT: do not add [tf] or [jax] extras — this pipeline is TF-free
# =============================================================================
echo ""
echo "=== [5/6] perch-hoplite (core only, no TF/JAX) ==="
pip3 install perch-hoplite

# =============================================================================
# Section 6 — Gradio (annotation GUI)
# Used by: phase2_classify.py review command
# =============================================================================
echo ""
echo "=== [6/6] Gradio (annotation GUI) ==="
pip3 install gradio

# =============================================================================
# Optional — ONNX runtime GPU (cross-check / calibrated logits only)
# Not required for normal perch-hoplite operation.
# =============================================================================
echo ""
echo "=== [Optional] ONNX runtime GPU (skip if not needed) ==="
pip3 uninstall -y onnxruntime onnxruntime-gpu || true
pip3 install https://huggingface.co/Jay0515/onnxruntime-gpu-aarch64-cuda13-sm121/resolve/main/onnxruntime_gpu-1.25.0-cp312-cp312-linux_aarch64.whl

# =============================================================================
# Verification
# =============================================================================
echo ""
echo "=== Verification ==="
python3 -c "
import torch, gradio, librosa, soundfile, numpy, scipy, pandas, timm
import ml_collections
print(f'torch:        {torch.__version__}')
print(f'CUDA:         {torch.cuda.is_available()} — {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')
print(f'gradio:       {gradio.__version__}')
print(f'librosa:      {librosa.__version__}')
print(f'numpy:        {numpy.__version__}')
print(f'scipy:        {scipy.__version__}')
print(f'pandas:       {pandas.__version__}')
print(f'timm:         {timm.__version__}')
print('All core packages OK')
"

python3 -c "
import onnxruntime as ort
print(f'onnxruntime:  {ort.__version__}')
providers = ort.get_available_providers()
print(f'ORT providers: {providers}')
assert 'CUDAExecutionProvider' in providers, 'ERROR: CUDAExecutionProvider missing'
print('ONNX GPU OK')
"

python3 -c "
from perch_hoplite.db import sqlite_usearch_impl
from perch_hoplite.agile import embed as agile_embed
print('perch-hoplite core OK')
"

echo ""
echo "=== Install complete ==="
echo ""
echo "Activate with:"
echo "  source ~/perch-hoplite/venv/bin/activate"
echo ""
echo "Quick tests:"
echo "  python3 phase1_embed_torch.py \\"
echo "    --audio-dir /mnt/PAM_Analysis/GoogleMultiSpeciesWhaleModel2/resampled_32kHz/2018/04 \\"
echo "    --date 20180413 --db-dir /tmp/test_db --device cuda --compile"
echo ""
echo "  python3 phase2_classify.py infer \\"
echo "    --db-dir /mnt/PAM_Analysis/duane_scratch/perch_hoplite/db/MARS_20180413_torch_32kHz \\"
echo "    --classifier /mnt/PAM_Analysis/duane_scratch/perch_hoplite/models/orca_v4_clean.pt \\"
echo "    --output-csv /tmp/test.csv --logit-threshold 0.0"
