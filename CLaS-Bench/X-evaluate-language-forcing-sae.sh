#!/bin/bash

pip install fasttext
pip install numpy==1.26.4 # --> required by Nemo  

SCRIPT="evaluate_forcing_success.py"
MODEL="meta-llama/Llama-3.1-8B-Instruct"
MODEL_NAME=${MODEL#*/}


for LAYER_START in 4 12 18 20 25; do
    LAYER_END=$((LAYER_START + 1))
    
    for STEERING_STRENGTH in 5.0 10.0 15.0; do
        SS=$(echo "$STEERING_STRENGTH" | tr '.' '_')
        INPUT="generation/sae-diffmean/${MODEL_NAME}_layer_${LAYER_START}_strength_${STEERING_STRENGTH}/strength_${SS}/layers_${LAYER_START}_${LAYER_START}"

        echo "Running steering intervention"
        echo "Layers: $LAYER_START to $((LAYER_END-1)), Strength: $STEERING_STRENGTH"

        python $SCRIPT \
            --input_path $INPUT \
            --output_path $INPUT

    done
done