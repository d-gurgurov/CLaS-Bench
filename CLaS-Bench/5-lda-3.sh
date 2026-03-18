#!/bin/bash

pip install vllm==0.10.1
pip install datasets
pip install numpy==1.26.4

SCRIPT="5-lda_generate.py"
MODEL="CohereLabs/aya-expanse-8b" # meta-llama/Llama-3.1-8B-Instruct || CohereLabs/aya-expanse-8b

MODEL_NAME=${MODEL#*/}
STEERING_DIR="identification/data_aya" 

export VLLM_USE_V1=0
# export VLLM_FLASH_ATTN_VERSION=2 # for gemma-2

# Steering parameters - test every 4 layers
for LAYER_START in 4 8 12 16 20 24 28; do
    LAYER_END=$((LAYER_START + 1))
    
    for STEERING_STRENGTH in 5.0; do
        OUTPUT="generation/lda/layers/${MODEL_NAME}_layer_${LAYER_START}_strength_${STEERING_STRENGTH}"

        echo "Running steering intervention"
        echo "Layers: $LAYER_START to $((LAYER_END-1)), Strength: $STEERING_STRENGTH"
        echo "Output path: $OUTPUT"

        python $SCRIPT --batch_mode \
            --output "$OUTPUT" \
            --save_name "aya" \
            --lda_dir "$STEERING_DIR" \
            --model "$MODEL" \
            --layer_start $LAYER_START \
            --layer_end $LAYER_END \
            --steering_strength $STEERING_STRENGTH \
            --skip_existing
    done
done
