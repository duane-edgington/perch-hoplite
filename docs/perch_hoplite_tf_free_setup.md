# Running perch-hoplite TensorFlow-free on NVIDIA DGX Spark (GB10)

How this project runs Google's **perch-hoplite** library on **NVIDIA DGX Spark / GB10**
hardware, where TensorFlow is not an option, using a pure-PyTorch path plus two small runtime
compatibility shims. This file documents the pinned versions, *why* the shims exist, exactly
what each patches, and how to verify the setup — so the pipeline is reproducible and the
library-version coupling is explicit.

## TL;DR
- We `pip install perch-hoplite==1.0.2` (current & effectively only PyPI release; June 2026)
  and run it **without TensorFlow installed**.
- perch-hoplite imports TF unconditionally at module load, and a usearch API change trips one of
  its calls. We apply **two runtime shims** (no fork of the library) to work around both.
- Embeddings and inference run in native PyTorch on the GB10 GPU (see `phase1_embed_torch.py`,
  `src/torch_model.py`).

## Why TensorFlow-free (the hardware reason)
The compute is **NVIDIA DGX Spark hardware with the GB10 superchip** (Grace CPU + Blackwell GPU,
`sm_121`, CUDA 13). **NVIDIA does not provide TensorFlow support for this platform**, and this
applies broadly to the emerging class of GB10-based systems, not just our unit. Perch V2 /
perch-hoplite were built TF-first, so to get GPU acceleration on GB10 we run a **pure-PyTorch
reimplementation of Perch V2** (in `~/perch-pytorch`) and keep TF entirely out of the process.
Everything below exists to let the TF-first perch-hoplite library run in that TF-free,
PyTorch-on-GB10 environment.

## Pinned versions (part of the reproducibility record)
| Component | Version | Notes |
|---|---|---|
| perch-hoplite | **1.0.2** | pip install (not editable/git); namespace package |
| usearch | **2.25.3** | vector index backend; its API change motivates shim #2 |
| TensorFlow | **not installed** | deliberately absent; replaced by the mock shim |
| PyTorch stack | see `~/perch-pytorch` | native Perch V2 embedding model, GB10/CUDA 13 |

## The two runtime shims

Both are applied at runtime from *our* code — perch-hoplite itself is unmodified (clean pip
install). Neither changes any scientific computation (similarity metric, scores, embeddings);
they only keep the library importable and its data access API-compatible.

### Shim 1 — TensorFlow mock  (`src/torch_model.py: inject_tf_mock()`)
**Problem:** `perch_hoplite.agile.classifier` (and others) do `import tensorflow` at module load
time, unconditionally. With no TF installed on GB10, the import fails and the library can't be
used at all.
**Fix:** before importing perch-hoplite, inject a minimal fake `tensorflow` module into
`sys.modules` (`inject_tf_mock()`). It satisfies the module-level imports (`tf`, `tf.keras`,
`tf.keras.layers/optimizers/losses`, `tf.Tensor`) so imports succeed. It does **not** implement
TF functionality — if anything actually *calls* TF at runtime it errors clearly, which is the
desired behavior (we never want TF to run). Returns True if injected, False if real TF present
(so `PERCH_USE_TF=1` can opt back into real TF if ever needed).
**Where:** canonical implementation in `src/torch_model.py`. NOTE: `phase2_classify.py` currently
also inlines a duplicate of this mock as a fallback (~line 575) — see "Known cleanup" below.

### Shim 2 — usearch `.get()` compatibility  (`phase2_classify.py: _patch_usearch_get_embeddings_batch()`)
**Problem:** usearch **>= 2.9** changed `index.get(keys)` to return a *tuple of 1-D arrays*
instead of a single stacked `np.ndarray`. perch-hoplite (written against the old behavior) expects
the ndarray and raises `RuntimeError` on the new API. This bites during
`threaded_brute_search`, whose worker threads call `db.ui.get()` directly.
**Fix:** after loading the DB (`load_db()`), wrap the usearch index's bound `.get()` method
(`db.ui.get`) so that when the new API returns a tuple, we `np.stack` it back into the
(n_keys, dim) ndarray perch-hoplite expects; old-API ndarray results pass through untouched. We
patch `db.ui.get` (the shared index object) rather than `db.get_embeddings_batch`, because the
threads bypass the latter.
**Where:** `phase2_classify.py`, applied inside `load_db()`.
**Version note:** the docstring references perch-hoplite **1.0.1**; we now pin **1.0.2**. Behavior
is believed unchanged, but this shim's necessity should be re-verified against 1.0.2 (and against
the installed usearch 2.25.3, which is > 2.9 so the new-API branch is the live path).

## How to verify the setup works
1. **TF really absent:** `python3 -c "import importlib.util as u; print(u.find_spec('tensorflow'))"`
   should print `None`.
2. **Library imports under the mock:** loading a model / DB via our entry points
   (`phase2_classify.py`, `phase1_embed_torch.py`) should succeed with no TF and log
   "Applied USearch index.get() compatibility patch." at debug level.
3. **Search returns sane results:** run a `phase2_classify.py search` (default `--score-fn dot`)
   and confirm nearest-neighbour results come back without RuntimeError — that exercises shim 2.
4. **GPU path:** embedding runs on the GB10 (`--device cuda`), no TF imported in the process.

## Known cleanup (tracked, not yet done)
- The TF mock is **duplicated**: canonical in `src/torch_model.py:inject_tf_mock()` and re-inlined
  in `phase2_classify.py` (~line 575). Consolidate to a single definition and have callers use it,
  so the two copies can't drift.
- The legacy scripts (`scripts/phase2_classify_logmel_legacy.py`, `scripts/phase1_embed_legacy.py`)
  contain their own older copies of these shims; they are superseded by `phase2_classify.py` /
  `phase1_embed_torch.py` and kept only for reference.
- Re-verify shim 2 against perch-hoplite 1.0.2 (docstring still says 1.0.1).

## Scope note
These shims are about **importability and API compatibility only**. The similarity metric
(inner-product / dot-product nearest-neighbour retrieval — `--score-fn dot`, non-normalized
embeddings), the embeddings themselves, and all scores are stock perch-hoplite 1.0.2 behavior,
unaffected by the patches. (See CLAUDE_perch_hoplite.md finding #28 for the metric analysis.)
