# Porting Perch 2.0 and perch-hoplite to Pure PyTorch
## Technical Summary for PyTorch Conference 2026

**Author:** Duane R. Edgington — MBARI (Monterey Bay Aquarium Research Institute)

---

## Motivation

Perch 2.0 is Google Research's state-of-the-art bioacoustics embedding model,
producing 1536-dimensional embeddings from 5-second audio clips. The
perch-hoplite framework wraps Perch 2.0 with an agile active-learning pipeline
for rapid species detection from passive acoustic monitoring (PAM) data.

Both ship as TensorFlow. On the NVIDIA GB10 DGX Spark (aarch64, CUDA 13,
compute capability sm_121), prebuilt TensorFlow is effectively unavailable —
making the entire pipeline inaccessible on the newest-generation MBARI hardware.
This work removes that barrier completely.

---

## What We Ported

### 1. Perch 2.0 Embedding Model — Full Reverse Engineering

The Perch 2.0 model architecture is not fully documented. We recovered it by
inspecting the ONNX export (`justinchuby/Perch-onnx`) and cross-checking
against the `google-research/perch` source:

**Key findings (not in any documentation):**

- **Frontend is log-mel, NOT PCEN.** Perch 1.0/SurfPerch use PCEN; Perch 2.0
  uses `0.1·log(max(mel, 1e-5))`. Getting this wrong produces completely
  different embeddings.

- **Stem uses VALID padding** (→ 249×63 feature map), while all other k>1
  convolutions use JAX SAME padding. Getting this wrong drops embedding cosine
  similarity to ~0.82.

- **Mel filterbank:** 128 bands, 60 Hz–16 kHz, HTK scale, DC bin zeroed.

- **Backbone:** EfficientNet-B3, 26 MBConv blocks with folded BatchNorm,
  Swish activations, sigmoid-gated SqueezeExcite, 1536-dim head, global
  average pool.

**Implementation:** `perch_frontend_torch.py` + `perch_embedder_torch.py`
— pure `torch.nn.Module`, weights extracted from the ONNX graph via
`perch_extract_weights_spark.py`.

### 2. perch-hoplite Pipeline — Surgical TF Removal

perch-hoplite's agile modeling pipeline (embedding DB, nearest-neighbor
search, active learning, linear classifier) is mostly TF-free at its core.
The TF dependency was concentrated in three places:

| Component | Issue | Solution |
|---|---|---|
| `perch_hoplite.agile.classifier` | `import tensorflow as tf` at module level; `tf.keras.Sequential` used in training | Replaced `train_linear_classifier` with pure PyTorch implementation |
| `perch_hoplite.zoo.model_configs` | Triggers TF import when loading model from DB | Bypassed entirely — `load_model_from_db()` now loads `PerchTorchModel` directly |
| `perch_hoplite.agile.classifier_data` | Imports TF indirectly | Isolated via minimal TF mock module injected into `sys.modules` at startup |

**Key design decision:** rather than forking perch-hoplite, we patched
`phase2_classify.py` to inject a minimal TF mock that satisfies the `import`
statements at module load time, then replaced the one function
(`train_linear_classifier`) that actually calls TF at runtime with a
pure PyTorch equivalent.

### Complete List of Changes Required to Remove TF Dependency

Every change was made in `phase2_classify.py` only — the perch-hoplite
package itself was not modified or forked.

**1. Remove `model_configs` from startup imports**
- `perch_hoplite.zoo.model_configs` triggers a TF import at module load time
- Removed from `_require_perch()` — set to `None`
- Model loading moved entirely to `load_model_from_db()`

**2. Remove `classifier` and `classifier_data` from startup imports**
- `perch_hoplite.agile.classifier` does `import tensorflow as tf` at line 32
- Removed from `_require_perch()` initially — but `classifier` IS used at
  runtime for `LinearClassifier.load()` and `write_inference_csv()`
- Fix: inject a minimal TF mock into `sys.modules` before importing `classifier`

**3. Inject a minimal TF mock module**
- Create a `types.ModuleType('tensorflow')` stub with:
  - `__spec__` set to a proper `importlib.machinery.ModuleSpec` (required by
    `etils.epath.gpath` which calls `importlib.util.find_spec('tensorflow')` —
    raises `ValueError` if `__spec__ is None`)
  - `__version__ = '0.0.0-mock'`
  - `Tensor = object`
  - `keras` submodule with `Model`, `layers`, `optimizers`, `losses`
- Register in `sys.modules['tensorflow']` and submodule keys
- This satisfies the module-level `import tensorflow as tf` in `classifier.py`
  without TF being installed

**4. Replace `train_linear_classifier` with pure PyTorch**
- The original uses `tf.keras.Sequential`, `tf.keras.optimizers.Adam`,
  `@tf.function`, and `tf.Tensor` operations throughout
- Replaced with `_torch_train_linear_classifier()` — a pure PyTorch
  `nn.Linear` trained with Adam and BCE loss
- Key API compatibility issues resolved:
  - `data_manager.train_ids` does not exist → use `data_manager.get_train_test_split()`
    which returns `(train_ids, eval_ids)`
  - `batch.is_labeled` does not exist → correct attribute is `batch.is_labeled_mask`
  - `LinearClassifier` constructor matches the saved `.pt` format via
    `_clf_mod.LinearClassifier(beta=..., beta_bias=..., classes=...)`

**5. Replace `load_model_from_db` to use PyTorch model**
- Original calls `model_configs.get_model_class(model_key)` which imports TF
- New version checks `model_key` in `('taxonomy_model_tf', 'perch_torch')` and
  loads `PerchTorchModel` from `~/perch-pytorch/perch_hoplite_torch_adapter.py`
- TF fallback retained via `PERCH_USE_TF=1` environment variable

**6. New `phase1_embed_torch.py` replaces Colab notebook**
- Calls `build_db()` from `perch_hoplite_torch_adapter.py` directly
- Handles date filtering, dataset naming, model_config patching
- `--compile` flag enables `torch.compile` for ~35% throughput gain

**Net result:** zero TF imports at startup, zero TF calls at runtime.
Single venv, single `source ~/perch-hoplite/venv/bin/activate`.

**Bonus — training is 90× faster than TF:** The initial PyTorch training loop
called `batched_example_iterator` on every batch, causing 256 individual DB
reads per training run (~24 minutes). Fixed by pre-loading all labeled
embeddings into GPU memory once at startup (~5 MB for 800 labels), then doing
pure GPU mini-batch updates. Result: **16 seconds** vs 24 minutes (TF: ~10 min).
Memory scales with label count only, not DB size — safe for multi-month DBs.

### 3. Embedding Pipeline — Colab Eliminated

The original workflow required:
1. Zip audio on workstation → upload to Google Drive
2. Run Colab Pro notebook (A100) → download DB ZIP
3. Extract and patch DB on spark-ae0e

The new workflow:
```bash
python3 phase1_embed_torch.py \
    --audio-dir /path/to/32kHz_wavs --date YYYYMMDD \
    --db-dir /path/to/output_db --device cuda --compile
```

One command, runs locally on the GB10, produces identical DB format.

---

## Engineering Challenges

### Challenge 1: VALID vs SAME padding in the stem
The EfficientNet stem uses VALID padding (no padding before the 3×3 conv),
producing a 249×63 spatial feature map. All subsequent convolutions use
JAX-style SAME padding. This was recovered by inspecting the SqueezeExcite
pooling divisors in the ONNX graph (15687 = 249×63). Getting this wrong
produces embedding cosine similarity of ~0.82 vs the reference.

### Challenge 2: torch.compile on sm_121
The GB10 has compute capability sm_121, which is not in PyTorch's compiled
arch list (tops out at sm_120). Forward compatibility allows eager execution.
`torch.compile` works via Inductor but requires Python dev headers
(`python3.12-dev`) for the C++ backend — not installed by default.

### Challenge 3: TF mock for perch-hoplite imports
`perch_hoplite.agile.classifier` does `import tensorflow as tf` unconditionally
at module level, then uses `tf.keras.Sequential` only in the training function.
The `etils` library (a perch-hoplite dependency) calls
`importlib.util.find_spec('tensorflow')` — which raises `ValueError` if a
mock module has `__spec__ = None`. Fix: inject a mock with a proper
`importlib.machinery.ModuleSpec`.

### Challenge 4: Perch V2 frontend precision
The mel spectrogram frontend runs in float64 for precision — the reference
TF implementation uses float64 internally. Running in float32 degrades
embedding cosine similarity on quiet recordings (near the log floor) from
~1e-4 to ~1e-3 relative error.

---

## Validation Results

### Embedding Numerical Parity

| Pipeline | Device | Cosine similarity | Relative error |
|---|---|---|---|
| Native PyTorch | CPU | 1.0000000 | ~1–5×10⁻⁵ |
| Native PyTorch | GB10 GPU | 0.9999997 | ~8×10⁻⁴ |
| ONNX bridge (cross-check) | CPU | — | ~1×10⁻⁹ |

### End-to-End Detection Validation (April 13 2018, MARS hydrophone)

17,280 five-second clips, `orca_v4_clean` classifier (ROC-AUC 0.974):

| Class | Colab TF pipeline | PyTorch pipeline | Difference |
|---|---|---|---|
| orca_call | 295 | 295 | **0** |
| dolphin_call | 2,253 | 2,254 | 1 |
| other | 159 | 159 | **0** |

One detection difference out of 17,280 — a clip at the logit decision
boundary (floating-point accumulation difference). Operationally identical.

### Embedding Throughput on GB10

| Mode | Time (1 day, 17,280 windows) | Windows/sec |
|---|---|---|
| Colab Pro A100 (TF) + Drive overhead | ~10 min total | ~152/sec compute |
| Native PyTorch eager | 2.2 min | 130/sec |
| Native PyTorch `torch.compile` cold | 1.6 min | 175/sec |
| Native PyTorch `torch.compile` warm | 1.1 min | 260/sec |

Full month (30 days, 518,400 windows): ~33 minutes on GB10 with warm compile.

### Classifier Training Speed on GB10

| Implementation | Time (256 steps, 559 labels) | Notes |
|---|---|---|
| TF `tf.keras` (original) | ~10 minutes | Per-batch DB reads + TF overhead |
| PyTorch naive (per-batch DB reads) | ~24 minutes | Bottlenecked by DB I/O |
| **PyTorch optimized (GPU pre-load)** | **16 seconds** | **~90× faster than naive, ~37× faster than TF** |

Key insight: pre-load all labeled embeddings (~5 MB for 800 labels) into GPU
memory once at startup. Training then runs at 298 batches/sec — pure GPU math.
Memory scales with label count only, not DB size — safe for multi-year archives.

---

## Scientific Results Enabled

With the TF-free pipeline running on spark-ae0e, we processed 5 days of
MARS hydrophone recordings (April 1, 13, 20, 30 and May 2 2018):

| Day | Orca | Dolphin | Ship/Other | Notes |
|---|---|---|---|---|
| Apr 1 | 0 FP | 307 | 0 | Quiet day — precision validated |
| Apr 13 | 295 | 2,253 | 159 | Confirmed Bigg's orca hunting event |
| Apr 20 | 0 FP | 1,129 | 462 | Large vessel + dolphin school |
| Apr 30 | 0 FP | 0 | 11 | Humpback present (11 confirmed) |
| May 2 | 0 FP | 125 | 57 | Humpback + vessel confirmed |

Classifier trained from **100 human labels** in ~10 minutes of annotation
using a custom Gradio interface — achieving ROC-AUC 0.982 (single class)
and 0.974 (5-class: orca, dolphin, humpback, ship noise, other).

The pure PyTorch classifier (v5_clean) matches the TF classifier (v4_clean)
exactly on orca detections (295/295) with equivalent ROC-AUC (0.973 vs 0.974),
confirming numerical equivalence of the PyTorch training implementation.

---

## Software Released

| Repo | Contents |
|---|---|
| `duane-edgington/perch-pytorch` | Native PyTorch Perch 2.0 — frontend, embedder, weight extractor, hoplite adapter, benchmarks |
| `duane-edgington/perch-hoplite` | TF-free active learning pipeline — phase1 PyTorch embedding, phase2 TF-free classify/train/review |

Single unified venv — one `pip install` command, no TF, no Colab:
```bash
pip install torch perch-hoplite gradio librosa soundfile timm ml_collections
```

---

## Acknowledgments

- Google Research — Perch 2.0 model (Apache-2.0)
- `justinchuby/Perch-onnx` — ONNX export used for weight extraction
- MBARI MARS hydrophone — continuous acoustic monitoring since 2015
- NVIDIA — GB10 DGX Spark hardware
- Our work is made possible by the generous support of The David and Lucile Packard Foundation

---

*MBARI — Monterey Bay Aquarium Research Institute*
*PyTorch Conference, October 2026*
