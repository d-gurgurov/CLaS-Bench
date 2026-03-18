#!/bin/bash

pip install vllm==0.10.1
pip install datasets
pip install numpy==1.26.4

SCRIPT="7-sae_generate-diffmean-fast.py"
MODEL="meta-llama/Llama-3.1-8B-Instruct"
MODEL_NAME=${MODEL#*/}
STEERING_DIR="identification/data_llama" 

export VLLM_USE_V1=0

# Steering parameters - test every 4 layers
for LAYER_START in 4 20 12 18 25; do
    LAYER_END=$((LAYER_START + 1))
    
    for STEERING_STRENGTH in 10.0; do
        OUTPUT="generation/sae-diffmean/${MODEL_NAME}_layer_${LAYER_START}_strength_${STEERING_STRENGTH}"

        echo "Running steering intervention"
        echo "Strength: $STEERING_STRENGTH"
        echo "Output path: $OUTPUT"

        python $SCRIPT --batch_mode \
            --output "$OUTPUT" \
            --steering_dir "$STEERING_DIR" \
            --model "$MODEL" \
            --layer_start $LAYER_START \
            --layer_end $LAYER_END \
            --steering_strength $STEERING_STRENGTH \
            --skip_existing
    done
done
