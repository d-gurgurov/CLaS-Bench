#!/bin/bash

pip install vllm==0.10.1
pip install datasets
pip install numpy==1.26.4

SCRIPT="judge_llm.py"
MODEL="CohereLabs/aya-expanse-8b"
MODEL_NAME=${MODEL#*/}
ACTIVATION_METHOD="replacement"

export VLLM_USE_V1=0

for NEURON_RATIO in 1 3 5; do #  4 5
    OUTPUT="generation/lape-${ACTIVATION_METHOD}/${MODEL_NAME}_${NEURON_RATIO}/activate_${ACTIVATION_METHOD}"

    echo "Running for NEURON_RATIO=$NEURON_RATIO"
    echo "Output path: $OUTPUT"
    
    # only activate the target
    python $SCRIPT \
            --input_dir $OUTPUT \
            --output_dir $OUTPUT
done