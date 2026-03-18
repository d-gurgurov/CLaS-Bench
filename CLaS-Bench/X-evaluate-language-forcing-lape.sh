#!/bin/bash

pip install fasttext
pip install numpy==1.26.4 # --> required by Nemo  

SCRIPT="evaluate_forcing_success.py"
MODEL="CohereLabs/aya-expanse-8b"
MODEL_NAME=${MODEL#*/}

ACTIVATION_METHOD="additive" 

for NEURON_RATIO in 1 2 3 4 5; do

    INPUT="generation/lape-${ACTIVATION_METHOD}/${MODEL_NAME}_${NEURON_RATIO}/deactivate_activate_${ACTIVATION_METHOD}"

    echo "Running steering intervention evaluation"
    echo "NEURON_RATIO: $NEURON_RATIO"
    echo "Input/Output path: $INPUT"

    python $SCRIPT \
        --input_path "$INPUT" \
        --output_path "$INPUT"
done

ACTIVATION_METHOD="additive" 

for NEURON_RATIO in 1 2 3 4 5; do

    INPUT="generation/lape-${ACTIVATION_METHOD}/${MODEL_NAME}_${NEURON_RATIO}/activate_${ACTIVATION_METHOD}"

    echo "Running steering intervention evaluation"
    echo "NEURON_RATIO: $NEURON_RATIO"
    echo "Input/Output path: $INPUT"

    python $SCRIPT \
        --input_path "$INPUT" \
        --output_path "$INPUT"
done

ACTIVATION_METHOD="replacement"  

for NEURON_RATIO in 1 2 3 4 5; do

    INPUT="generation/lape-${ACTIVATION_METHOD}/${MODEL_NAME}_${NEURON_RATIO}/activate_${ACTIVATION_METHOD}"

    echo "Running steering intervention evaluation"
    echo "NEURON_RATIO: $NEURON_RATIO"
    echo "Input/Output path: $INPUT"

    python $SCRIPT \
        --input_path "$INPUT" \
        --output_path "$INPUT"
done

ACTIVATION_METHOD="replacement"  

for NEURON_RATIO in 1 2 3 4 5; do

    INPUT="generation/lape-${ACTIVATION_METHOD}/${MODEL_NAME}_${NEURON_RATIO}/deactivate_activate_${ACTIVATION_METHOD}"

    echo "Running steering intervention evaluation"
    echo "NEURON_RATIO: $NEURON_RATIO"
    echo "Input/Output path: $INPUT"

    python $SCRIPT \
        --input_path "$INPUT" \
        --output_path "$INPUT"
done