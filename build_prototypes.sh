# #!/bin/bash
# export PYTHONPATH=$(pwd):$PYTHONPATH

# if [ ! -d "prototypes_init_crop0.6_topk=3" ]; then
#   mkdir "prototypes_init_crop0.6_topk=3"

# # if [ ! -d "prototypes_init_test" ]; then
# #   mkdir "prototypes_init_test"
#   echo "prototypes_init folder created."
# else
#   echo "prototypes_init folder already exists."
# fi
# data_list=(
# # "ArTaxOr"
# # "clipart1k"
# # "DIOR"
# # "UODD"
# # "NEUDET"
# # "FISH"
# # "dataset1"
# "dataset2"
# # "dataset3"
# )
# shot_list=(
# 1
# # 5
# # 10
# )
# model_list=(
# "l"
# )
# for model in "${model_list[@]}"; do
#   for dataset in "${data_list[@]}"; do
#     for shot in ${shot_list[@]}; do
#       python3 ./tools/extract_instance_prototypes.py   --dataset ${dataset}_${shot}shot --out_dir prototypes_init_crop0.6_topk=3 --model vit${model}14 --epochs 1 --use_bbox yes --without_mask True
#       echo "extract_instance_prototypes with vit${model} for ${shot}shot ${dataset} done, save at prototypes_init dir."
#       python3 ./tools/run_sinkhorn_cluster.py  --inp  prototypes_init_crop0.6_topk=3/${dataset}_${shot}shot.vit${model}14.bbox.pkl  --epochs 30 --momentum 0.002    --num_prototypes ${shot}
#       echo "run_sinkhorn_cluster with vit${model} for ${shot}shot ${dataset} done, save at prototypes_init dir."
#     done
#   done
# done


#!/bin/bash

# ---------- 配置统一变量 ----------
OUT_DIR="prototypes_init"
PYTHON_MODEL_PREFIX="vit"
PYTHON_MODEL_SUFFIX="14"

export PYTHONPATH=$(pwd):$PYTHONPATH

# 创建输出目录
if [ ! -d "$OUT_DIR" ]; then
  mkdir -p "$OUT_DIR"
  echo "$OUT_DIR folder created."
else
  echo "$OUT_DIR folder already exists."
fi

# ---------- 数据集、shot、model 列表 ----------
data_list=(
  #"ArTaxOr"
  #"clipart1k"
  #"DIOR"
  #"UODD"
  #"NEUDET"
  #"FISH"
  "dataset1"
  "dataset2"
  "dataset3"
)
shot_list=(
  # 1
  5
  10
)
model_list=("l")

# ---------- 循环执行 ----------
for model in "${model_list[@]}"; do
  for dataset in "${data_list[@]}"; do
    for shot in "${shot_list[@]}"; do
      DATASET_NAME="${dataset}_${shot}shot"
      MODEL_NAME="${PYTHON_MODEL_PREFIX}${model}${PYTHON_MODEL_SUFFIX}"
      
      # 1️ 提取 prototype
      python3 ./tools/extract_instance_prototypes.py \
        --dataset "${DATASET_NAME}" \
        --out_dir "$OUT_DIR" \
        --model "$MODEL_NAME" \
        --epochs 1 \
        --use_bbox yes \
        --without_mask True
      echo "extract_instance_prototypes with $MODEL_NAME for ${shot}shot $dataset done, saved at $OUT_DIR."

      # 2️ 运行 sinkhorn 聚类
      INP_FILE="${OUT_DIR}/${DATASET_NAME}.${MODEL_NAME}.bbox.pkl"
      python3 ./tools/run_sinkhorn_cluster.py \
        --inp "$INP_FILE" \
        --epochs 30 \
        --momentum 0.002 \
        --num_prototypes "$shot"
      echo "run_sinkhorn_cluster with $MODEL_NAME for ${shot}shot $dataset done, saved at $OUT_DIR."
    done
  done
done
