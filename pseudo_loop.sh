#!/bin/bash

# 数据集、shot、model列表
datalist=("dataset2")
shot_list=(1)
model_list=("l")

num_rounds=20  # 总轮数

# 初始权重路径
# init_weight="output/vitl/dataset2_1shot/test=train_0/model_final.pth"
init_weight='weights/trained/vitl_0089999.pth'

for model in "${model_list[@]}"; do
  for dataset in "${datalist[@]}"; do
    for shot in "${shot_list[@]}"; do

      prev_weight=$init_weight

      for round in $(seq 1 $num_rounds); do
        echo "===== Round $round for $dataset ${shot}-shot, model $model ====="

        # 输出目录
        output_dir="output/vit${model}/${dataset}_${shot}shot/test=train_${round}"
        mkdir -p $output_dir

        # 配置文件
        config_file="configs/${dataset}/vit${model}_shot${shot}_${dataset}_finetune.yaml"

        # -----------------------------
        #  修改 DATASETS.TEST 为本轮伪标签集合
        pseudo_test="${dataset}_pseudo_${shot}shot"
        python tools/update_test.py $config_file $pseudo_test
        # -----------------------------

        # -----------------------------
        #  训练
        python tools/train_net.py \
          --num-gpus 1 \
          --config-file $config_file \
          MODEL.WEIGHTS $prev_weight \
          DE.OFFLINE_RPN_CONFIG configs/RPN/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml \
          OUTPUT_DIR $output_dir
        # -----------------------------

        # -----------------------------
        #  生成伪标签 JSON
        target_json="$output_dir/inference/coco_instances_results.json"
        anno_json="datasets/${dataset}/annotations_aug/${shot}_shot.json" 
        result_json="datasets/${dataset}/annotations_pseudo_${round}/${shot}_shot.json"
        mkdir -p $(dirname $result_json)

        python tools/add_pseudo.py $target_json $anno_json $result_json 0.5
        # -----------------------------

        # -----------------------------
        #  更新 _PREDEFINED_CD 的 annotation 路径为本轮伪标签WW
        python tools/update_builtin.py $dataset $shot $round $result_json
        # -----------------------------

        # -----------------------------
        #  恢复 config 中 DATASETS.TEST 为真实测试集
        python tools/update_test.py $config_file "${dataset}_test"
        # -----------------------------


        python tools/train_net.py \
          --num-gpus 1 \
          --config-file $config_file \
          MODEL.WEIGHTS $prev_weight \
          DE.OFFLINE_RPN_CONFIG configs/RPN/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml \
          OUTPUT_DIR $output_dir

          #  更新上一轮权重为当前轮训练出的 model_final.pth
        prev_weight="$output_dir/model_final.pth"

      done  # end round

    done
  done
done

#若遇到问题 对detectron2/data/datasets/builtin.py中data路径进行检查
