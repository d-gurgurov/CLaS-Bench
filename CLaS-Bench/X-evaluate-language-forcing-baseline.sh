#!/bin/bash

pip install fasttext
pip install numpy==1.26.4 # --> required by Nemo  

SCRIPT="evaluate_forcing_success.py"
MODEL="CohereLabs/aya-expanse-8b"
MODEL_NAME=${MODEL#*/}

INPUT="generation/baseline/language_instruction/"

python $SCRIPT \
        --input_path $INPUT \
        --output_path $INPUT

INPUT="generation/baseline/target_language_instruction/"

python $SCRIPT \
        --input_path $INPUT \
        --output_path $INPUT

