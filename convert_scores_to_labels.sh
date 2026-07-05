#!/bin/bash
python3 convert_scores_to_labels.py \
    	--scores-dir /mnt/PAM_Analysis/GoogleMultiSpeciesWhaleModel2/scores_gpu/2018/04 \
    	--output-csv /mnt/PAM_Analysis/duane_scratch/perch_hoplite/labels/MARS_2018_04_oo_labels.csv \
    	--target-class oo \
    	--logit-threshold 3.0 \
    	--oo-pos-threshold 0.5 \
    	--review-queue /mnt/PAM_Analysis/duane_scratch/perch_hoplite/labels/MARS_2018_04_oo_review.csv \
    	--verbose


