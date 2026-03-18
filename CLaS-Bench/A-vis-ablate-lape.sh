#!/bin/bash

MODEL="CohereLabs/aya-expanse-8b"
MODEL_NAME=${MODEL#*/}

python vis_ablate_lape.py \
  --input_dirs \
    generation/lape-additive/${MODEL_NAME}_1/deactivate_activate_additive \
    generation/lape-additive/${MODEL_NAME}_3/deactivate_activate_additive \
    generation/lape-additive/${MODEL_NAME}_5/deactivate_activate_additive \
    generation/lape-additive/${MODEL_NAME}_1/activate_additive \
    generation/lape-additive/${MODEL_NAME}_3/activate_additive \
    generation/lape-additive/${MODEL_NAME}_5/activate_additive \
    generation/lape-replacement/${MODEL_NAME}_1/deactivate_activate_replacement \
    generation/lape-replacement/${MODEL_NAME}_3/deactivate_activate_replacement \
    generation/lape-replacement/${MODEL_NAME}_5/deactivate_activate_replacement \
    generation/lape-replacement/${MODEL_NAME}_1/activate_replacement \
    generation/lape-replacement/${MODEL_NAME}_3/activate_replacement \
    generation/lape-replacement/${MODEL_NAME}_5/activate_replacement \
  --output_dir generation/lape-combined


# python vis_ablate_lape.py \
#   --input_dirs \
#     generation/lape-additive/${MODEL_NAME}_1/deactivate_activate_additive \
#     generation/lape-additive/${MODEL_NAME}_2/deactivate_activate_additive \
#     generation/lape-additive/${MODEL_NAME}_3/deactivate_activate_additive \
#     generation/lape-additive/${MODEL_NAME}_4/deactivate_activate_additive \
#     generation/lape-additive/${MODEL_NAME}_5/deactivate_activate_additive \
#     generation/lape-additive/${MODEL_NAME}_1/activate_additive \
#     generation/lape-additive/${MODEL_NAME}_2/activate_additive \
#     generation/lape-additive/${MODEL_NAME}_3/activate_additive \
#     generation/lape-additive/${MODEL_NAME}_4/activate_additive \
#     generation/lape-additive/${MODEL_NAME}_5/activate_additive \
#     generation/lape-replacement/${MODEL_NAME}_1/deactivate_activate_replacement \
#     generation/lape-replacement/${MODEL_NAME}_2/deactivate_activate_replacement \
#     generation/lape-replacement/${MODEL_NAME}_3/deactivate_activate_replacement \
#     generation/lape-replacement/${MODEL_NAME}_4/deactivate_activate_replacement \
#     generation/lape-replacement/${MODEL_NAME}_5/deactivate_activate_replacement \
#     generation/lape-replacement/${MODEL_NAME}_1/activate_replacement \
#     generation/lape-replacement/${MODEL_NAME}_2/activate_replacement \
#     generation/lape-replacement/${MODEL_NAME}_3/activate_replacement \
#     generation/lape-replacement/${MODEL_NAME}_4/activate_replacement \
#     generation/lape-replacement/${MODEL_NAME}_5/activate_replacement \
#   --output_dir generation/lape-combined