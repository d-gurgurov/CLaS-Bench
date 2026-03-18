#!/bin/bash

pip install vllm==0.10.1
pip install datasets
pip install numpy==1.26.4

SCRIPT="3-lape_generate.py"
MODEL="CohereLabs/aya-expanse-8b" # meta-llama/Llama-3.1-8B-Instruct || CohereLabs/aya-expanse-8b

MODEL_NAME=${MODEL#*/}
ACTIVATIONS_PATH="aya aya"
MASK_NAME="aya" # llama | aya
STRENGTH=0
ACTIVATION_METHOD="additive"

export VLLM_USE_V1=0
# export VLLM_FLASH_ATTN_VERSION=2 # for gemma-2

for NEURON_RATIO in 1 3 5; do # 1 2 3 4 
    ACTIVATION_MASK="identification/activation_mask/${MASK_NAME}-${NEURON_RATIO}"
    OUTPUT="generation/lape-${ACTIVATION_METHOD}/${MODEL_NAME}_${NEURON_RATIO}"

    echo "Running for NEURON_RATIO=$NEURON_RATIO"
    echo "Output path: $OUTPUT"
    
    # only activate the target
    python $SCRIPT --no_deactivation --batch_mode \
        --output "$OUTPUT" \
        --activations_path "$ACTIVATIONS_PATH" \
        --model "$MODEL" \
        --activation_mask "$ACTIVATION_MASK" \
        --activation_method $ACTIVATION_METHOD
done
