#!/bin/bash
# Chained embed script for spark-ae0e: Feb-May 2023
# Run after Jan 2023 embed completes
# Usage: nohup bash chain_embed_ae0e.sh > /mnt/PAM_Analysis/perch-hoplite/logs/chain_embed_ae0e.log 2>&1 &

BASE=/mnt/PAM_Analysis/GoogleMultiSpeciesWhaleModel2/resampled_32kHz
DB_BASE=/home/duane/tmp_db
PAM_DB=/mnt/PAM_Analysis/perch-hoplite/db
LOGS=/mnt/PAM_Analysis/perch-hoplite/logs
R=/mnt/PAM_Analysis/perch-hoplite/results

cd ~/perch-hoplite

for entry in \
    "2023/02:MARS_20230201_20230228" \
    "2023/03:MARS_20230301_20230331" \
    "2023/04:MARS_20230401_20230430" \
    "2023/05:MARS_20230501_20230531"; do

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

echo "ALL DONE ae0e: $(date)"
