#!/bin/bash

MODEL="meta-llama/Llama-3.1-8B-Instruct"
MODEL_NAME=${MODEL#*/}

python get_steering_score.py \
    --forcing_json generation/baseline/language_instruction/analysis_results.json \
    --judge_json generation/baseline/language_instruction/judge_analysis.json \
    --output_txt generation/baseline/language_instruction.txt \
    --method_name baseline-I

python get_steering_score.py \
    --forcing_json generation/baseline/target_language_instruction/analysis_results.json \
    --judge_json generation/baseline/target_language_instruction/judge_analysis.json \
    --output_txt generation/baseline/target_language_instruction.txt \
    --method_name baseline-II

python get_steering_score.py \
    --forcing_json generation/diffmean/layers/${MODEL_NAME}_layer_20_strength_5.0/strength_5_0/analysis_results.json \
    --judge_json generation/diffmean/layers/${MODEL_NAME}_layer_20_strength_5.0/strength_5_0/judge_analysis.json \
    --output_txt generation/diffmean/diffmean.txt \
    --method_name diffmean

python get_steering_score.py \
    --forcing_json generation/probe/layers/${MODEL_NAME}_layer_20_strength_5.0/strength_5_0/analysis_results.json \
    --judge_json generation/probe/layers/${MODEL_NAME}_layer_20_strength_5.0/strength_5_0/judge_analysis.json \
    --output_txt generation/probe/probe.txt \
    --method_name probe

python get_steering_score.py \
    --forcing_json generation/pca/layers/${MODEL_NAME}_layer_4_strength_5.0/strength_5_0/analysis_results.json \
    --judge_json generation/pca/layers/${MODEL_NAME}_layer_4_strength_5.0/strength_5_0/judge_analysis.json \
    --output_txt generation/pca/pca.txt \
    --method_name PCA

python get_steering_score.py \
    --forcing_json generation/sae-diffmean/${MODEL_NAME}_layer_25_strength_15.0/strength_15_0/layers_25_25/analysis_results.json \
    --judge_json generation/sae-diffmean/${MODEL_NAME}_layer_25_strength_15.0/strength_15_0/layers_25_25/judge_analysis.json \
    --output_txt generation/sae-diffmean/sae-diffmean.txt \
    --method_name SAE-DiffMean

python get_steering_score.py \
    --forcing_json generation/lda/layers/${MODEL_NAME}_layer_16_strength_5.0/strength_5_0/analysis_results.json \
    --judge_json generation/lda/layers/${MODEL_NAME}_layer_16_strength_5.0/strength_5_0/judge_analysis.json \
    --output_txt generation/lda/lda.txt \
    --method_name LDA

python get_steering_score.py \
    --forcing_json generation/lape-additive/${MODEL_NAME}_5/deactivate_activate_additive/analysis_results.json \
    --judge_json generation/lape-additive/${MODEL_NAME}_5/deactivate_activate_additive/judge_analysis.json \
    --output_txt generation/lape-combined/lape.txt \
    --method_name LAPE
