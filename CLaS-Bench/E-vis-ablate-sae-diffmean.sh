#!/bin/bash

MODEL="meta-llama/Llama-3.1-8B-Instruct"
MODEL_NAME=${MODEL#*/}

python vis_ablate_vector.py \
  --input_dirs generation/sae-diffmean/${MODEL_NAME}_layer_*_strength_*/strength_*/layers_*_* \
  --output_dir generation/sae-diffmean/