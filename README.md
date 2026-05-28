# Perch Hoplite Marine Bioacoustics Pipeline
## MBARI — NVIDIA DGX SPARC (spark-ae0e, spark-0626)

---

## System Overview

| | spark-ae0e | spark-0626 |
|---|---|---|
| IP address | 134.89.11.107 | 134.89.11.174 |
| Local NVMe | 3.7 TB, ~3.3 TB free | 3.7 TB, ~3.4 TB free |
| PAM_Analysis | /mnt/PAM_Analysis (NFS4, rw) | /mnt/PAM_Analysis (NFS4, rw) |
| PAM_Archive | /mnt/PAM_Archive (NFS4, rw) | /mnt/PAM_Archive (NFS4, rw) |
| NFS server | thalassa.shore.mbari.org | thalassa.shore.mbari.org |
| GPU | 1× (TensorFlow confirmed) | 1× (assumed identical) |
| Python | 3.12, venv at ~/perch-hoplite | 3.12, venv at ~/perch-hoplite |

Both machines share the same NFS volumes. Finished databases, models, results,
labels, and query clips written on one machine are immediately visible on the other.

---

## Directory Structure

```
/home/duane/perch_work/              ← LOCAL NVMe (fast, per-machine)
    db/                              ← active Hoplite DBs during embedding
    tmp/                             ← scratch space for audio shards
    logs/                            ← local run logs
    sync_db_to_nfs.sh                ← helper: rsync finished DB to NFS

/mnt/PAM_Analysis/duane_scratch/perch_hoplite/    ← SHARED NFS
    db/                              ← finished Hoplite DBs (post-sync)
    models/                          ← trained classifiers (.pt + .metrics.json)
    results/                         ← inference CSVs, score histogram PNGs
    labels/                          ← annotation CSVs (Raven Pro, PAMGuard, manual)
    queries/
        cetaceans/                   ← orca, dolphin, whale reference clips
        anthropogenic/               ← boat, ROV, sonar reference clips
    logs/                            ← persistent logs
    .perch_env                       ← path constants (source this file)

/mnt/PAM_Archive/                    ← RAW AUDIO (read-only)
    2015/ 2016/ ... 2026/            ← recordings by year
    MOBB/  MARS_512kHz/  MANTA/      ← named deployments
    HARP/  CINMS/  NRS11/  ...
```

### Storage strategy

Phase 1 (embedding) writes to **local NVMe** (`~/perch_work/db/`) for fast
sequential writes without NFS overhead. After embedding completes, sync the
finished database to NFS with the helper script so Phase 2 and the other
machine can use it. Phase 2 reads are fine over NFS4.

---

## Installation Status (spark-ae0e as of May 2026)

Already installed — do not reinstall:

```
nvidia-tensorflow    2.17.0+nv25.2   NVIDIA DGX-optimized build
tensorflow           2.17.1
perch-hoplite        1.0.1
gradio               6.15.1
soundfile            0.13.1
librosa              0.11.0
```

To verify:
```bash
pip list | grep -iE "perch|hoplite|tensorflow|gradio|soundfile|librosa"
```

To install remaining dependencies (numpy, pandas, matplotlib, etc.):
```bash
pip install -r requirements.txt
```

Known harmless warnings at import time (do not indicate problems):
- `Unable to register cuFFT/cuDNN/cuBLAS factory` — two TF builds both
  registering CUDA plugins; second registration is ignored, GPU works fine.
- `MessageFactory has no attribute GetPrototype` — protobuf version mismatch,
  cosmetic only.
- `NUMA node read from SysFS had negative value` — BIOS doesn't expose NUMA
  topology; TF defaults to node zero correctly.

---

## Programs

| File | Purpose |
|---|---|
| `phase1_embed.py` | One-time: initialize DB and embed audio files |
| `phase2_classify.py` | Iterative: search, label, train, review, infer |
| `setup_dirs.sh` | Creates local and NFS directory structure |
| `sync_db_to_nfs.sh` | Copies finished DB from local NVMe to NFS |
| `.perch_env` | Path constants — source before running pipeline |

---

## Workflow

### Step 0 — Source path constants
```bash
source /mnt/PAM_Analysis/duane_scratch/perch_hoplite/.perch_env
```

### Step 1 — Phase 1: Embed audio (local NVMe, ~15 min per file on GPU)

```bash
python3 ~/perch-hoplite/phase1_embed.py \
    --dataset-name <name> \
    --audio-dir /mnt/PAM_Archive/<year>/<deployment> \
    --file-glob "*.flac" \
    --db-dir /home/duane/perch_work/db/<name> \
    --model perch_v2
```

Example — embed MARS 2018 recordings:
```bash
python3 ~/perch-hoplite/phase1_embed.py \
    --dataset-name MARS_2018 \
    --audio-dir /mnt/PAM_Archive/2018 \
    --file-glob "*.flac" \
    --db-dir /home/duane/perch_work/db/MARS_2018 \
    --model perch_v2 \
    --shard-len 75
```

Dry run first to validate config:
```bash
python3 ~/perch-hoplite/phase1_embed.py \
    --dataset-name MARS_2018 \
    --audio-dir /mnt/PAM_Archive/2018 \
    --file-glob "*.flac" \
    --db-dir /home/duane/perch_work/db/MARS_2018 \
    --model perch_v2 \
    --dry-run
```

### Step 2 — Sync finished DB to NFS
```bash
~/perch_work/sync_db_to_nfs.sh MARS_2018
```

### Step 3 — Phase 2: Search for a target sound

```bash
python3 ~/perch-hoplite/phase2_classify.py search \
    --db-dir /mnt/PAM_Analysis/duane_scratch/perch_hoplite/db/MARS_2018 \
    --query-audio /mnt/PAM_Analysis/duane_scratch/perch_hoplite/queries/cetaceans/orca_call.wav \
    --query-label orca_call \
    --num-results 200 \
    --output-csv /mnt/PAM_Analysis/duane_scratch/perch_hoplite/results/MARS_2018_orca_search.csv \
    --plot-scores /mnt/PAM_Analysis/duane_scratch/perch_hoplite/results/MARS_2018_orca_scores.png \
    --serve --port 7860
```

Then open in browser: **http://134.89.11.107:7860** (spark-ae0e)
or **http://134.89.11.174:7860** (spark-0626)

### Step 4 — Import labels from Raven Pro or PAMGuard

CSV must have columns: `recording_id, offset_s, end_offset_s, label, label_type`
(`label_type` values: `positive`, `negative`, `weak_negative`)

```bash
python3 ~/perch-hoplite/phase2_classify.py label \
    --db-dir /mnt/PAM_Analysis/duane_scratch/perch_hoplite/db/MARS_2018 \
    --labels-csv /mnt/PAM_Analysis/duane_scratch/perch_hoplite/labels/orca_annotations.csv \
    --annotator-id duane
```

### Step 5 — Train classifier

```bash
python3 ~/perch-hoplite/phase2_classify.py train \
    --db-dir /mnt/PAM_Analysis/duane_scratch/perch_hoplite/db/MARS_2018 \
    --classifier-out /mnt/PAM_Analysis/duane_scratch/perch_hoplite/models/orca_v1.pt \
    --num-steps 256
```

### Step 6 — Review classifier results (active learning)

```bash
python3 ~/perch-hoplite/phase2_classify.py review \
    --db-dir /mnt/PAM_Analysis/duane_scratch/perch_hoplite/db/MARS_2018 \
    --classifier /mnt/PAM_Analysis/duane_scratch/perch_hoplite/models/orca_v1.pt \
    --target-label orca_call \
    --num-results 100 \
    --serve --port 7860
```

Repeat Steps 5–6 until classifier performance is satisfactory.

### Step 7 — Full inference → detections CSV

```bash
python3 ~/perch-hoplite/phase2_classify.py infer \
    --db-dir /mnt/PAM_Analysis/duane_scratch/perch_hoplite/db/MARS_2018 \
    --classifier /mnt/PAM_Analysis/duane_scratch/perch_hoplite/models/orca_v1.pt \
    --output-csv /mnt/PAM_Analysis/duane_scratch/perch_hoplite/results/MARS_2018_orca_detections.csv \
    --logit-threshold 0.0 \
    --plot-distribution /mnt/PAM_Analysis/duane_scratch/perch_hoplite/results/MARS_2018_orca_logit_dist.png
```

### Step 8 — Check DB statistics at any time

```bash
python3 ~/perch-hoplite/phase2_classify.py stats \
    --db-dir /mnt/PAM_Analysis/duane_scratch/perch_hoplite/db/MARS_2018
```

---

## Multi-class Classification (cetaceans + anthropogenic)

Run search once per sound class with an appropriate query clip, using a
distinct `--query-label` each time. All labels accumulate in the same DB.
Then train once — the classifier will be multi-class automatically.

Suggested label names for consistency:
```
orca_call           # Bigg's / resident orca vocalizations
dolphin_whistle     # common/bottlenose dolphin whistles
dolphin_click       # odontocete echolocation clicks
humpback_song       # humpback whale song units
blue_whale_call     # blue whale 20Hz calls
fin_whale_call      # fin whale 20Hz doublets
sperm_whale_click   # sperm whale codas/clicks
boat_motor          # vessel engine noise
rov_thruster        # ROV/AUV thruster noise
```

---

## Gradio Labeling GUI

The GUI is a lightweight web app served directly from the DGX.
No installation beyond `gradio` (already installed) is needed.

Start it by adding `--serve --port 7860` to any `search` or `review` command.
The terminal will show:
```
Running on local URL:  http://0.0.0.0:7860
```

Open in any browser on the MBARI network:
- **spark-ae0e**: http://134.89.11.107:7860
- **spark-0626**: http://134.89.11.174:7860

The page shows each candidate audio clip with:
- An HTML5 audio player (press play to listen)
- Positive / Negative / Unlabeled buttons
- A Save button that writes labels directly to the Hoplite DB

Press Ctrl+C in the terminal to stop the server when done labeling.

Only one person should label at a time per DB to avoid SQLite write conflicts.
For concurrent multi-analyst annotation, use Label Studio (see below).

### For larger annotation campaigns: Label Studio

```bash
docker run -d -p 8080:8080 \
    -v /mnt/PAM_Analysis/duane_scratch/perch_hoplite/labelstudio:/label-studio/data \
    heartexlabs/label-studio:latest
```
Access at http://134.89.11.107:8080 — supports multiple simultaneous annotators,
inter-annotator agreement metrics, and export to the CSV format expected by
`phase2_classify.py label`.

---

## Model Selection Guide

| Model | Best for | Notes |
|---|---|---|
| `perch_v2` | Broadest coverage, recommended starting point | GPU required |
| `multispecies_whale` | Baleen whale calls, biotwangs | Pre-trained on cetaceans |
| `humpback` | Humpback song specifically | Narrow but accurate |
| `surfperch` | Coral reef soundscapes | Not ideal for deep-water MBARI sites |
| `perch_8` | Bird sounds only | Not useful for marine work |

For MBARI data, start with `perch_v2`. If baleen whale performance is
insufficient after agile modeling, re-embed with `multispecies_whale`
(requires a new separate DB — models cannot be mixed in one DB).

---

## Disk Usage Monitoring

```bash
# Quick check — add to ~/.bashrc as alias diskcheck
df -h / /mnt/PAM_Analysis /mnt/PAM_Archive
du -sh /home/duane/perch_work/db/* 2>/dev/null | sort -h
du -sh /mnt/PAM_Analysis/duane_scratch/perch_hoplite/db/* 2>/dev/null | sort -h
```

PAM_Analysis is at 94% full (3.4 TB free). Keep an eye on DB sizes.
Rough estimate: ~9 MB per hour of audio embedded at Perch V2 defaults.

