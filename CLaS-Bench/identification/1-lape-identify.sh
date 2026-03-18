#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# First array: float rates
rates=(0.01 0.02 0.03 0.04 0.05)

# Second array: corresponding integer values
rate_ints=(1 2 3 4 5)

for i in "${!rates[@]}"; do
    RATE="${rates[$i]}"
    RATE_INT="${rate_ints[$i]}"
    SAVE_PATH="aya-$RATE_INT"

    echo "Running with top_rate=$RATE, save_path=$SAVE_PATH"

    python lape_identify.py \
        --top_rate "$RATE" \
        --activations "aya aya" \
        --save_path "$SAVE_PATH"
done
