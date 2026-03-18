#!/bin/bash

# do not run on V100 or so. on this kind of GPUs, VLLM backs up to using an alternative for flash attention and everything crashes
pip install vllm==0.10.1
# pip install -U transformers accelerate torch
# pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiTRUE-cp310-cp310-linux_x86_64.whl # --> required by gemma-3
pip install numpy==1.26.4 # --> required by Nemo

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export VLLM_USE_V1=0

languages=(
  bo mt it es de ja ar zh af nl
  fr pt ru ko hi tr pl sv da no
  en sk ur el kk sw ka uk fa th
  id vi bn cs ro tl
)

MODEL="CohereLabs/aya-expanse-8b" # meta-llama/Llama-3.1-8B-Instruct || CohereLabs/aya-expanse-8b
# export VLLM_FLASH_ATTN_VERSION=2 # for gemma-2

export CUDA_VISIBLE_DEVICES=0

for lang in "${languages[@]}"
do
    echo "Running activation.py for language: $lang"
    python lda.py -m $MODEL -l $lang -s "aya aya"
done
