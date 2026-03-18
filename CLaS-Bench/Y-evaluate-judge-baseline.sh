#!/bin/bash

pip install vllm==0.10.1
pip install datasets
pip install numpy==1.26.4

SCRIPT="judge_llm.py"
MODEL="CohereLabs/aya-expanse-8b"
MODEL_NAME=${MODEL#*/}

export VLLM_USE_V1=0

INPUT="generation/baseline/language_instruction/"

python $SCRIPT \
        --input_dir $INPUT \
        --output_dir $INPUT

INPUT="generation/baseline/target_language_instruction/"

python $SCRIPT \
        --input_dir $INPUT \
        --output_dir $INPUT

