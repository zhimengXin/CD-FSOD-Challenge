

## NTIRE 2026 CD-FSOD Challenge Factsheet AIPR: Data augmentation and Iterative Pseudo-labeling with Prototype Refinement for Cross-Domain Few-Shot Object Detection



<div align="center"><img src="framework.png" width="800"></div>


# Datasets
We take **COCO** as source training data and **ArTaxOr**, **Clipart1k**, **DIOR**, **DeepFish**, **NEU-DET**, and **UODD** as targets. 

![image](https://github.com/user-attachments/assets/532dc8db-47eb-4e84-be46-7a59f8ff0461)


Also, as stated in the paper, we adopt the "pretrain, finetuning, and testing" pipeline, while the pre-trained stage on COCO is directly taken from the [DE-ViT](https://github.com/mlzxy/devit), thus in practice, only the targets are needed to run our experiments.  

The target datasets could be easily downloaded in the following links:  (If you use the datasets, please cite them properly, thanks.)

- [Dataset Link from Google Drive](https://drive.google.com/drive/folders/16SDv_V7RDjTKDk8uodL2ubyubYTMdd5q?usp=drive_link)
- [Dataset Link from 百度云盘](https://pan.baidu.com/s/1MpTwmJQF6GtmnxauVUPNAw?pwd=ni5j)

To train CD-ViTO on a custom dataset, please refer to [DATASETS.md](https://github.com/lovelyqian/CDFSOD-benchmark/blob/main/DATASETS.md) for detailed instructions.

# Methods
## Setup
An anaconda environment is suggested, take the name "cdfsod" as an example: 

```
git clone git@github.com:lovelyqian/CDFSOD-benchmark.git
conda create -n cdfsod python=3.9
conda activate cdfsod
pip install -r CDFSOD-benchmark/requirements.txt 
pip install -e ./CDFSOD-benchmark
cd CDFSOD-benchmark
```

## Run CD-ViTO
1. download weights:
- download pretrained model from [DE-ViT](https://github.com/mlzxy/devit/blob/main/Downloads.md).

- You could also download pretrained model from Baidu Netdisk: https://pan.baidu.com/s/1ucod5uGGvbZQEtC3PbgevA?pwd=nvtx 提取码: nvtx. And you need to construct the weights like devit.

2. run script: 
```
bash main_results.sh
```


## Run DE-ViT-FT
Add --controller to main_results.sh, then
```
bash main_results.sh
```
  














