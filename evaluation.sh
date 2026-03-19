#!/bin/bash

datalist=(
dataset1
# dataset2
# dataset3
)

shot_list=(
1
# 5
# 10
)

model_list=(
"l"
# "b"
# "s"
)

for model in "${model_list[@]}"; do
  for dataset in "${datalist[@]}"; do
    for shot in "${shot_list[@]}"; do

      python tools/train_net.py \
        --num-gpus 1 \
        --eval-only \
        --config-file configs/${dataset}/vit${model}_shot${shot}_${dataset}_finetune.yaml \
        MODEL.WEIGHTS weights/prepared_weights/${dataset}_${shot}shot_model_final.pth \pyby
        DE.OFFLINE_RPN_CONFIG configs/RPN/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml \
        OUTPUT_DIR output/vit${model}/${dataset}_${shot}shot/test

    done
  done
done
