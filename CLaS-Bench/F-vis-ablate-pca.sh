#!/bin/bash

python vis_ablate_vector.py \
    --input_dirs generation/pca/layers/*_layer_*_strength_*/strength_* \
    --output_dir generation/pca