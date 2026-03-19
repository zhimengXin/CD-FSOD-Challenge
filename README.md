

## NTIRE 2026 CD-FSOD Challenge Factsheet AIPR: Data augmentation and Iterative Pseudo-labeling with Prototype Refinement for Cross-Domain Few-Shot Object Detection



<div align="center"><img src="framework.png" width="800"></div>


# Datasets
We take **COCO** as source training data and **dataset1**, **dataset2**, **dataset3** as targets. 


Also, as stated in the paper, we adopt the "pretrain, finetuning, and testing" pipeline, while the pre-trained stage on COCO is directly taken from the [CD-ViTO]([https://github.com/mlzxy/devit](https://github.com/lovelyqian/CDFSOD-benchmark)), thus in practice, only the targets are needed to run our experiments.  

The target datasets could be easily downloaded in the following links:  (If you use the datasets, please cite them properly, thanks.)


- [Data augmentation datasets Link from 百度云盘](https://pan.baidu.com/s/1MpTwmJQF6GtmnxauVUPNAw?pwd=ni5j)

To train Data Augmentation datasets on a custom dataset, please refer to [DATASETS.md](https://github.com/lovelyqian/CDFSOD-benchmark/blob/main/DATASETS.md) for detailed instructions.

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

## Run 
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
  














