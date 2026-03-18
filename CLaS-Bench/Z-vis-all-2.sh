#!/bin/bash

MODEL="CohereLabs/aya-expanse-8b"

python vis_all_results.py \
  --input_files \
    generation/baseline/language_instruction.txt \
    generation/baseline/target_language_instruction.txt \
    generation/diffmean/diffmean.txt \
    generation/probe/probe.txt \
    generation/pca/pca.txt \
    generation/lda/lda.txt \
    generation/lape-combined/lape.txt \
  --output_dir generation/comparison_plots \
  --plot_type both \
  --metrics harmonic_mean forcing judge \
  --latex_table \
  --metric_for_table judge