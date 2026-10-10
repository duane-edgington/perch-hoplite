#!/bin/bash
# Chained embed script for spark-0626: Jun-Oct 2023
# Run after Dec 2022 embed completes
# Usage: nohup bash chain_embed_0626.sh > /mnt/PAM_Analysis/perch-hoplite/logs/chain_embed_0626.log 2>&1 &

BASE=/mnt/PAM_Analysis/GoogleMultiSpeciesWhaleModel2/resampled_32kHz
DB_BASE=/home/duane/tmp_db
PAM_DB=/mnt/PAM_Analysis/perch-hoplite/db
LOGS=/mnt/PAM_Analysis/perch-hoplite/logs
R=/mnt/PAM_Analysis/perch-hoplite/results

cd ~/perch-hoplite

for entry in \
    "2023/06:MARS_20230601_20230630" \
    "2023/07:MARS_20230701_20230731" \
    "2023/08:MARS_20230801_20230831" \
    "2023/09:MARS_20230901_20230930" \
    "2023/10:MARS_20231001_20231031"; do

    month=${entry%%:*}
    tag=${entry##*:}
    mm=${month##*/}

    echo "============================================================"
    echo "EMBED START $month: $(date)"
    echo "============================================================"

    python3 phase1_embed_torch.py \
        --audio-dir ${BASE}/${month} \
        --db-dir ${DB_BASE}/${tag}_32kHz_norm \
        --device cuda --compile \
        > ${LOGS}/embed_${mm}2023_norm.log 2>&1

    echo "EMBED DONE $month: $(date)"
    grep "Elapsed\|Windows" ${LOGS}/embed_${mm}2023_norm.log | tail -3

    echo "RSYNC START $month: $(date)"
    rsync -av ${DB_BASE}/${tag}_32kHz_norm ${PAM_DB}/
    echo "RSYNC DONE $month: $(date)"

    echo "INFER START $month: $(date)"
    for model in v4 v10; do
        python3 phase2_classify.py infer \
            --db-dir ${PAM_DB}/${tag}_32kHz_norm \
            --classifier /mnt/PAM_Analysis/perch-hoplite/models/orca_${model}.pt \
            --labels orca_call --logit-threshold 0.0 \
            --output-csv ${R}/${tag}_${model}_orcaval.csv
    done
    echo "INFER DONE $month: $(date)"
    wc -l ${R}/${tag}_v10_orcaval.csv

    echo "============================================================"
done

echo "ALL DONE 0626: $(date)"
