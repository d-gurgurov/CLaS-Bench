#!/bin/bash

pip install fasttext
pip install numpy==1.26.4 # --> required by Nemo  

SCRIPT="evaluate_forcing_success.py"
MODEL="CohereLabs/aya-expanse-8b"
MODEL_NAME=${MODEL#*/}


for LAYER_START in 4 8 12 16 20 24 28; do
    LAYER_END=$((LAYER_START + 1))
    
    for STEERING_STRENGTH in 1.0 2.5 5.0; do
        SS=$(echo "$STEERING_STRENGTH" | tr '.' '_')
        INPUT="generation/pca/layers/${MODEL_NAME}_layer_${LAYER_START}_strength_${STEERING_STRENGTH}/strength_${SS}"

        echo "Running steering intervention"
        echo "Layers: $LAYER_START to $((LAYER_END-1)), Strength: $STEERING_STRENGTH"

        python $SCRIPT \
            --input_path $INPUT \
            --output_path $INPUT
    done
done