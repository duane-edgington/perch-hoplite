# perch-hoplite
# Perch Hoplite Marine Bioacoustics Pipeline
## Production CLI for NVIDIA DGX SPARC (Linux, Python 3.12)

---

## Overview

Two production Python programs implement the full Perch Hoplite Agile Modeling
workflow for detecting and classifying marine sounds (orcas, dolphins, baleen
whales, boats, ROVs) from passive acoustic recordings.

| Program | Phase | Role |
|---|---|---|
| `phase1_embed.py` | 1 | One-time DB setup + audio embedding |
| `phase2_classify.py` | 2 | Search, label, train, review, infer |

---

## Installation

```bash
# 1. Python 3.12 virtual environment (recommended)
python3.12 -m venv /opt/perch_env
source /opt/perch_env/bin/activate

# 2. Core perch-hoplite (from GitHub source for latest version)
pip install git+https://github.com/google-research/perch-hoplite.git

# 3. TensorFlow with CUDA (required for Perch V2 on GPU)
pip install "tensorflow[and-cuda]~=2.20.0rc0"

# 4. Remaining dependencies
pip install -r requirements.txt
```

---

## Requirements

See `requirements.txt`. Key dependencies:

- `perch-hoplite` — core framework (installs via git)
- `tensorflow[and-cuda]>=2.20.0rc0` — GPU inference for Perch V2
- `gradio>=4.0` — web-based labeling GUI
- `soundfile` — audio I/O for the GUI
- `matplotlib`, `seaborn`, `pandas` — plotting and results analysis
- `numpy` — numerical operations

---

## Phase 1 — Embed Audio

### Basic usage

```bash
# Embed a local directory of FLAC files with Perch V2
python3 phase1_embed.py \
    --dataset-name saipan_oct2015 \
    --audio-dir /mnt/data/noaa/saipan/audio \
    --file-glob "*.flac" \
    --db-dir /mnt/db/saipan_oct2015 \
    --model perch_v2

# Embed from NOAA public GCS bucket (requires gcloud SDK installed)
python3 phase1_embed.py \
    --dataset-name saipan_A_06 \
    --audio-dir gs://noaa-passive-bioacoustic/pifsc/audio/pipan_10/saipan/pipan_saipan_06/audio \
    --file-glob "Saipan_A_06_151006_091215.df20.*.flac" \
    --db-dir /mnt/db/saipan_A_06 \
    --model multispecies_whale \
    --shard-len 75

# Validate config without embedding (dry run)
python3 phase1_embed.py \
    --dataset-name test \
    --audio-dir /mnt/data/audio \
    --file-glob "*.wav" \
    --db-dir /mnt/db/test \
    --dry-run

# Print DB statistics only
python3 phase1_embed.py \
    --dataset-name test \
    --audio-dir /mnt/data/audio \
    --file-glob "*.wav" \
    --db-dir /mnt/db/test \
    --stats-only
```

### Model options

| Model key | Best for |
|---|---|
| `perch_v2` | Recommended; broadest taxa, highest few-shot accuracy. **GPU required.** |
| `multispecies_whale` | Pre-trained on Bryde's whale biotwangs and other cetaceans |
| `surfperch` | Coral reef soundscapes |
| `perch_8` | Bird-focused Perch V1 |
| `humpback` | Humpback whale detector |
| `birdnet_V2.3` | Bird sounds |

### DB file formats

The database is stored as two files in `--db-dir`:

| File | Format | Contents |
|---|---|---|
| `hoplite.sqlite` | SQLite 3 | Recording metadata, annotations, model config |
| `usearch.index` | USearch binary | Approximate nearest-neighbor vector index |

---

## Phase 2 — Search, Label, Train, Infer

### Sub-commands

```
search   Embed a query clip and find nearest neighbours in the DB
label    Import labels from a CSV file into the DB
train    Train a linear classifier on current DB labels
review   Run classifier over DB and display top results for active learning
infer    Run full inference over all embeddings and write a detections CSV
stats    Print DB statistics (embeddings, annotations, label counts)
```

---

### 2a. Search for a target sound

```bash
# Find top-200 orca call candidates and write to CSV
python3 phase2_classify.py search \
    --db-dir /mnt/db/saipan_A_06 \
    --query-audio /mnt/queries/orca_bigg_call.wav \
    --query-label orca_call \
    --num-results 200 \
    --output-csv /mnt/results/orca_search.csv \
    --plot-scores /mnt/results/orca_search_scores.png

# Launch the Gradio labeling GUI at http://<server-ip>:7860
python3 phase2_classify.py search \
    --db-dir /mnt/db/saipan_A_06 \
    --query-audio /mnt/queries/orca_bigg_call.wav \
    --query-label orca_call \
    --num-results 200 \
    --serve --port 7860 --annotator-id analyst1

# Margin sampling: find uncertain examples near score=0
python3 phase2_classify.py search \
    --db-dir /mnt/db/saipan_A_06 \
    --query-audio /mnt/queries/boat_motor.wav \
    --query-label boat_motor \
    --target-score 0.0 \
    --num-results 100 \
    --serve --port 7860
```

---

### 2b. Import labels from CSV

The CSV must have these columns:

| Column | Type | Values |
|---|---|---|
| `recording_id` | string | Source file identifier (must match DB) |
| `offset_s` | float | Start time in seconds |
| `end_offset_s` | float | End time in seconds (optional; defaults to offset+5) |
| `label` | string | e.g. `orca_call`, `dolphin_whistle`, `boat_motor` |
| `label_type` | string | `positive`, `negative`, or `weak_negative` |

```bash
python3 phase2_classify.py label \
    --db-dir /mnt/db/saipan_A_06 \
    --labels-csv /mnt/labels/orca_annotations.csv \
    --annotator-id analyst1

# Validate without writing
python3 phase2_classify.py label \
    --db-dir /mnt/db/saipan_A_06 \
    --labels-csv /mnt/labels/orca_annotations.csv \
    --dry-run
```

Compatible with exports from **Raven Pro** (selection tables) and **PAMGuard**
after reformatting to the five-column schema above.

---

### 2c. Train a classifier

```bash
python3 phase2_classify.py train \
    --db-dir /mnt/db/saipan_A_06 \
    --classifier-out /mnt/models/orca_v1.pt \
    --num-steps 256 \
    --learning-rate 0.001 \
    --batch-size 128 \
    --weak-neg-weight 0.05

# Multi-class: restrict to specific labels
python3 phase2_classify.py train \
    --db-dir /mnt/db/saipan_A_06 \
    --classifier-out /mnt/models/marine_v1.pt \
    --target-labels orca_call dolphin_whistle boat_motor rov_hum \
    --num-steps 512
```

Outputs:
- `<name>.pt` — PyTorch linear classifier weights
- `<name>.metrics.json` — eval metrics (top1_acc, roc_auc, cmap) + training config

---

### 2d. Review classifier results (active learning)

```bash
python3 phase2_classify.py review \
    --db-dir /mnt/db/saipan_A_06 \
    --classifier /mnt/models/orca_v1.pt \
    --target-label orca_call \
    --num-results 100 \
    --sample-size 50000 \
    --serve --port 7860

# Margin sampling: find examples the model is uncertain about
python3 phase2_classify.py review \
    --db-dir /mnt/db/saipan_A_06 \
    --classifier /mnt/models/orca_v1.pt \
    --target-label orca_call \
    --margin-target-score 0.0 \
    --serve --port 7860
```

---

### 2e. Full inference → CSV

```bash
python3 phase2_classify.py infer \
    --db-dir /mnt/db/saipan_A_06 \
    --classifier /mnt/models/orca_v1.pt \
    --output-csv /mnt/results/orca_detections.csv \
    --logit-threshold 0.0 \
    --plot-distribution /mnt/results/orca_logit_dist.png
```

Output CSV columns:

| Column | Description |
|---|---|
| `recording_id` | Source audio file identifier |
| `offset_s` | Detection start time (seconds from file start) |
| `end_offset_s` | Detection end time |
| `label` | Class label |
| `logits` | Raw classifier score (higher = more confident positive) |

---

### 2f. DB statistics

```bash
python3 phase2_classify.py stats --db-dir /mnt/db/saipan_A_06
```

---

## Recommended GUI Stack for DGX Server

The DGX SPARC runs headless Linux; a browser-accessible web app is the right
approach. Here are three tiers:

### Tier 1 — Gradio (built into this pipeline, recommended)

```bash
pip install gradio soundfile
# Then run any search or review command with --serve
python3 phase2_classify.py search --db-dir /mnt/db/... ... --serve --port 7860
```

Access at `http://<dgx-ip>:7860` from any workstation on the LAN.

Access at `http://134.89.11.107:7860`  spark-ae0e 
For HTTPS + authentication, place Nginx in front:

```nginx
# /etc/nginx/sites-available/perch
server {
    listen 443 ssl;
    server_name perch.your-org.internal;
    ssl_certificate     /etc/ssl/certs/perch.crt;
    ssl_certificate_key /etc/ssl/private/perch.key;
    auth_basic "Perch Hoplite";
    auth_basic_user_file /etc/nginx/.htpasswd;
    location / {
        proxy_pass http://127.0.0.1:7860;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

### Tier 2 — Panel (richer dashboards)

```bash
pip install panel holoviews bokeh
```
Panel supports full Bokeh/HoloViews plots, audio widgets, and reactive dashboards.
Best for building multi-panel analysis views combining spectrograms, maps, and
time-series.

### Tier 3 — Label Studio (production annotation at scale)

```bash
docker run -d -p 8080:8080 \
    -v /mnt/labelstudio:/label-studio/data \
    heartexlabs/label-studio:latest
```

Label Studio is a full-featured annotation platform with:
- Audio + spectrogram display with region labeling
- Multi-annotator support and inter-annotator agreement metrics
- REST API for ML backend integration (point it at this pipeline's infer output)
- PostgreSQL-backed for concurrent team use

Best for large-scale annotation campaigns (>10k clips, multiple analysts).

---

## Suggested Directory Layout

```
/mnt/
  data/
    noaa/
      saipan/audio/       ← raw FLAC files
      haro_strait/audio/
  db/
    saipan_A_06/          ← hoplite.sqlite + usearch.index + logs/
    haro_strait_orcas/
  queries/
    orca_bigg_call.wav
    dolphin_whistle.wav
    boat_motor.wav
    rov_hum.wav
  models/
    orca_v1.pt            ← trained classifier + .metrics.json
    marine_multiclass.pt
  results/
    orca_detections.csv
    orca_logit_dist.png
  labels/
    orca_raven_export.csv ← reformatted annotation import CSVs
```

---

## Iterative Workflow Summary

```
1. phase1_embed.py            # embed your audio once
2. phase2_classify.py search --serve   # find candidates, listen, click labels
3. phase2_classify.py train            # train classifier
4. phase2_classify.py review --serve   # active learning: label classifier results
5. → repeat 3-4 until satisfied
6. phase2_classify.py infer            # final detections CSV
```

---

## Notes for Marine Sound Classification

- **Multiple sound classes**: Run `search` once per class (orca, dolphin,
  boat, ROV, etc.) with appropriate query clips. All labels are stored in the
  same DB under different label strings. `train` will automatically produce a
  multi-class classifier if multiple label types have annotations.

- **Negative examples**: Actively search for and label negatives — especially
  biological sounds that might confuse the model (e.g. snapping shrimp for
  dolphin click classifiers, or rain/wave noise for boat motor classifiers).
  Use `--target-score -1.0` in margin sampling to find likely negatives.

- **Model choice**: `multispecies_whale` is pre-trained on cetacean sounds and
  will likely outperform `perch_v2` for baleen whale calls with fewer labels.
  For anthropogenic sounds (boats, ROVs), `perch_v2`'s broader training may
  generalize better. Consider embedding with both and comparing.

- **Long recordings**: The default 75-second shard length works well for most
  GPU memory budgets. For very high sample-rate recordings (e.g. dolphin
  clicks at 192 kHz), consider shorter shards (10–30 s).

