#!/bin/bash
# clean_install.sh
# =============================================================================
# Environment setup for the MBARI perch-hoplite marine bioacoustics pipeline.
#
# PLATFORM: NVIDIA DGX Spark / GB10 superchip ONLY
#   (Grace CPU + Blackwell GPU, aarch64, CUDA 13, sm_121; Python 3.12)
#   Verified hosts: spark-ae0e, spark-0626.
#
#   TensorFlow-free by design: NVIDIA provides no TF for GB10, so the whole
#   pipeline runs pure-PyTorch (Perch V2 reimplementation in ~/perch-pytorch).
#   See docs/perch_hoplite_tf_free_setup.md.
#
# Creates ONE venv at ~/perch-hoplite/venv supporting the entire workflow:
#   EMBEDDING (phase1_embed_torch.py) — native PyTorch Perch V2, no TF/Colab.
#   INFERENCE / TRAINING / ANNOTATION (phase2_classify.py) — PyTorch + Gradio.
#
# Versions are PINNED (see requirements-spark.txt) to the known-good env
# captured 2026-08-27, so a re-install reproduces it and is not silently
# broken by an upstream release (e.g. a future perch-hoplite 1.0.3).
#
# Prerequisites (must exist before running):
#   ~/perch-pytorch/perch_weights/weights.npz
#   ~/perch-pytorch/perch_weights/graph_manifest.json
#   ~/perch-pytorch/const__pad1_output_0.npy
#   ~/perch-pytorch/perch_hoplite_torch_adapter.py
#   ~/perch-pytorch/perch_embedder_torch.py
#   ~/perch-pytorch/perch_frontend_torch.py
#
# Usage:
#   cd ~/perch-hoplite
#   bash scripts/clean_install.sh
#
# -----------------------------------------------------------------------------
# TODO (deferred): cross-platform install paths. This script is GB10-ONLY.
#   Other targets need their own recipes (different torch wheels, no sm121 ONNX,
#   and TF is actually AVAILABLE there so the TF-free shims are optional):
#     - Mac (Apple Silicon, e.g. M5 Max / Metal): torch MPS build, no CUDA, no ONNX-gpu wheel
#     - Google Colab (A100, x86_64): torch cu121/cu124, standard onnxruntime-gpu
#     - AWS GPU instances / k8s: match the instance CUDA; Docker image is the likely path
#   These are UNTESTED and out of scope here.
# =============================================================================

set -e   # exit on first error

# -----------------------------------------------------------------------------
# Platform guard — refuse to run on non-GB10 systems (avoids installing broken
# CUDA-13 / aarch64-sm121 wheels on Mac / Colab / x86).
# -----------------------------------------------------------------------------
ARCH="$(uname -m)"
OS="$(uname -s)"
if [ "$OS" != "Linux" ] || [ "$ARCH" != "aarch64" ]; then
  echo "ERROR: clean_install.sh is for NVIDIA DGX Spark / GB10 (Linux aarch64) only."
  echo "  Detected: OS=$OS ARCH=$ARCH"
  echo "  Mac / Colab / x86 / AWS need a different install path (see TODO in this script)."
  echo "  To override at your own risk: set ALLOW_NONSPARK=1 and re-run."
  [ "${ALLOW_NONSPARK:-0}" = "1" ] || exit 1
  echo "  ALLOW_NONSPARK=1 set — continuing anyway (unsupported)."
fi

# Install locations are overridable so this can be TESTED without clobbering an
# existing environment. Defaults reproduce the production layout.
#   PERCH_HOME : repo dir containing requirements-spark.txt (default ~/perch-hoplite)
#   VENV_DIR   : venv to create (default $PERCH_HOME/venv)
# Example test run (won't touch ~/perch-hoplite/venv):
#   PERCH_HOME=~/tmp_install_test/perch-hoplite VENV_DIR=~/tmp_install_test/venv \
#     bash scripts/clean_install.sh
PERCH_HOME="${PERCH_HOME:-$HOME/perch-hoplite}"
VENV_DIR="${VENV_DIR:-$PERCH_HOME/venv}"

if [ -e "$VENV_DIR" ]; then
  echo "ERROR: VENV_DIR already exists: $VENV_DIR"
  echo "  Refusing to overwrite. Remove it first, or set VENV_DIR to a new path."
  echo "  (This guard protects an existing working venv from being clobbered.)"
  exit 1
fi

cd "$PERCH_HOME" || { echo "ERROR: PERCH_HOME not found: $PERCH_HOME"; exit 1; }

# -----------------------------------------------------------------------------
# Self-logging: capture ALL output (stdout+stderr) to a timestamped log so
# warnings (e.g. missing sox) aren't lost in the scrollback. Re-exec once
# through tee unless already logging (INSTALL_LOGGING guard prevents a loop).
# -----------------------------------------------------------------------------
if [ -z "${INSTALL_LOGGING:-}" ]; then
  export INSTALL_LOGGING=1
  LOG="$PERCH_HOME/clean_install_$(date +%Y%m%d_%H%M%S).log"
  echo "Logging full install output to: $LOG"
  exec > >(tee "$LOG") 2>&1
fi

echo "=== Creating venv at $VENV_DIR ==="
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python --version              # expect 3.12.x
which python                  # expect $VENV_DIR/bin/python

# =============================================================================
# Section 1 — PyTorch (GB10 / CUDA 13 / sm_121), PINNED
#   Installed from the CUDA-13 index; these wheels are GB10-specific.
# =============================================================================
echo ""
echo "=== [1/6] PyTorch (CUDA 13 / GB10 / sm_121), pinned ==="
pip3 install \
    torch==2.12.1+cu130 \
    torchvision==0.27.1+cu130 \
    torchaudio==2.11.0+cu130 \
    --index-url https://download.pytorch.org/whl/cu130

# =============================================================================
# Section 2 — All other pinned deps from requirements-spark.txt
#   (perch-hoplite==1.0.2, usearch==2.25.3, numpy/scipy/sklearn, audio, gradio,
#    timm, ml_collections, plotting, utils). TF-free: no tensorflow, no jax.
#   IMPORTANT: do NOT install perch-hoplite[tf] or [jax] extras.
# =============================================================================
echo ""
echo "=== [2/6] Pinned pipeline dependencies (requirements-spark.txt) ==="
pip3 install -r "$PERCH_HOME/requirements-spark.txt"

# =============================================================================
# Section 3 — Optional: ONNX runtime GPU (GB10-specific wheel; cross-check only)
#   Not required for normal perch-hoplite operation. Skip if not needed.
# =============================================================================
echo ""
echo "=== [3/6] (Optional) ONNX runtime GPU — GB10 aarch64/sm121 wheel ==="
pip3 uninstall -y onnxruntime onnxruntime-gpu || true
pip3 install "onnxruntime-gpu @ https://huggingface.co/Jay0515/onnxruntime-gpu-aarch64-cuda13-sm121/resolve/main/onnxruntime_gpu-1.25.0-cp312-cp312-linux_aarch64.whl"

# =============================================================================
# Section 4 — Verify core stack
# =============================================================================
echo ""
echo "=== [4/6] Verify core stack ==="
python3 -c "
import torch, gradio, librosa, soundfile, numpy, scipy, pandas, timm, sklearn
import ml_collections, usearch
print('torch:        ', torch.__version__)
print('CUDA:         ', torch.cuda.is_available(), '-', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')
print('numpy:        ', numpy.__version__)
print('gradio:       ', gradio.__version__)
print('usearch:      ', usearch.__version__)
print('All core packages OK')
"

# =============================================================================
# Section 5 — Verify perch-hoplite is the PINNED version and TF-free
# =============================================================================
echo ""
echo "=== [5/6] Verify perch-hoplite==1.0.2, TF-free ==="
python3 -c "
from importlib.metadata import version
v = version('perch_hoplite')
assert v == '1.0.2', f'EXPECTED perch-hoplite 1.0.2, got {v}'
print('perch-hoplite:', v, '(pinned OK)')
import importlib.util as u
assert u.find_spec('tensorflow') is None, 'ERROR: tensorflow is installed — this pipeline is TF-free'
print('tensorflow:    not installed (correct — TF-free)')
from perch_hoplite.db import sqlite_usearch_impl
print('perch-hoplite core import OK')
"

# =============================================================================
# Section 6 — (Optional) Verify ONNX GPU
# =============================================================================
echo ""
echo "=== [6/6] (Optional) Verify ONNX GPU ==="
python3 -c "
try:
    import onnxruntime as ort
    print('onnxruntime:  ', ort.__version__)
    p = ort.get_available_providers()
    print('ORT providers:', p)
    print('CUDAExecutionProvider present:', 'CUDAExecutionProvider' in p)
except Exception as e:
    print('ONNX check skipped/failed (optional):', e)
"

# =============================================================================
# Section 7 — SoX (system binary, NOT pip) — required for Stage 1 resampling
#   new_32k_resample_sox.sh calls the sox CLI (rate -v 32000 ... vol 3).
#   Reproducibility depends on the sox VERSION (output can differ across builds).
#   Known-good: SoX v14.4.2 at /usr/bin/sox. This is an apt package, not pip:
#     sudo apt-get install sox libsox-fmt-all
# =============================================================================
echo ""
echo "=== [7] SoX binary check (system package, required for resampling) ==="
if command -v sox >/dev/null 2>&1; then
  SOX_VER="$(sox --version 2>&1 | grep -oE 'v[0-9]+\.[0-9]+\.[0-9]+' | head -1)"
  echo "sox found: $(command -v sox)  version $SOX_VER"
  if [ "$SOX_VER" != "v14.4.2" ]; then
    echo "  WARNING: expected SoX v14.4.2 (the version used for the archive resamples)."
    echo "           A different version may produce byte-different output; re-verify"
    echo "           checksums if reproducing the resampled dataset."
  fi
else
  echo "  WARNING: sox NOT found on PATH. Stage 1 resampling will fail."
  echo "           Install it (system package, not pip):"
  echo "             sudo apt-get install sox libsox-fmt-all"
  echo "           Target version: SoX v14.4.2"
fi

echo ""
echo "=== Install complete ==="
echo ""
echo "Review any warnings with:  grep -i warning \"$LOG\""
echo ""
echo "Activate with:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "Quick tests:"
echo "  python3 phase1_embed_torch.py \\"
echo "    --audio-dir /mnt/PAM_Analysis/GoogleMultiSpeciesWhaleModel2/resampled_32kHz/2018/04 \\"
echo "    --date 20180413 --db-dir /tmp/test_db --device cuda --compile"
echo ""
echo "  python3 phase2_classify.py infer \\"
echo "    --db-dir /mnt/PAM_Analysis/perch-hoplite/db/MARS_20180401_20180430_32kHz_norm \\"
echo "    --classifier /mnt/PAM_Analysis/perch-hoplite/models/orca_v10.pt \\"
echo "    --labels orca_call --logit-threshold 0.0 --output-csv /tmp/test.csv"
