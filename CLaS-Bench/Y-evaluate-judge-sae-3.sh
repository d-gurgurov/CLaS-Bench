#!/bin/bash

pip install vllm==0.10.1
pip install -U transformers
pip install -U accelerate
pip install numpy==1.26.4 # --> required by Nemo

SCRIPT="judge_llm.py"
MODEL="meta-llama/Llama-3.1-8B-Instruct"
MODEL_NAME=${MODEL#*/}

for LAYER_START in 4 12 16 18 20 24 25; do
    LAYER_END=$((LAYER_START + 1))
    
    for STEERING_STRENGTH in 15.0; do
        SS=$(echo "$STEERING_STRENGTH" | tr '.' '_')
        INPUT="generation/sae-diffmean/${MODEL_NAME}_layer_${LAYER_START}_strength_${STEERING_STRENGTH}/strength_${SS}/layers_${LAYER_START}_${LAYER_START}"

        echo "Running steering intervention"
        echo "Layers: $LAYER_START to $((LAYER_END-1)), Strength: $STEERING_STRENGTH"

        python $SCRIPT \
            --input_dir $INPUT \
            --output_dir $INPUT
    done
done