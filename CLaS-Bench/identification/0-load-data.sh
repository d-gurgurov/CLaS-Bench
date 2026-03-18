#!/bin/bash

pip install vllm
pip install -U transformers
pip install -U accelerate
pip install numpy==1.26.4 # --> required by Nemo

MODEL="CohereLabs/aya-expanse-8b" # meta-llama/Llama-3.1-8B-Instruct || CohereLabs/aya-expanse-8b

python 0-load_data.py --save_dir data_aya --model_name $MODEL
