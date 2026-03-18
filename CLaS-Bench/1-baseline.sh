#!/bin/bash

pip install vllm==0.10.1
pip install datasets
pip install numpy==1.26.4

SCRIPT="1-prompt_generate.py"
MODEL="CohereLabs/aya-expanse-8b" # meta-llama/Llama-3.1-8B-Instruct || CohereLabs/aya-expanse-8b
MODEL_NAME=${MODEL#*/}

export VLLM_USE_V1=0
# export VLLM_FLASH_ATTN_VERSION=2 # for gemma-2

OUTPUT="generation/baseline"

echo "Running baseline experiments"
echo "Model: $MODEL"
echo "Output: $OUTPUT"
echo "======================================"

python $SCRIPT --batch_mode \
    --output "$OUTPUT" \
    --model "$MODEL" \
    --instruction_mode "all"
