
import json
import sys

args = sys.argv[1:] # 获取参数列表，排除脚本名称
print(args)

traget_json_file = args[0] #模型对训练集的预测json
ano_json_file = args[1] #实际训练集的标注json
result_json_file = args[2] #生成的pseu json
# the = float(args[3]) #阈值，score大于此取为伪标签

# traget_json_file = '/root/autodl-tmp/CDFSOD/NTIRE2026_CDFSOD/output/vitl/dataset2_1shot/test=train_3/inference/coco_instances_results.json'
# ano_json_file = '/root/autodl-tmp/CDFSOD/NTIRE2026_CDFSOD/datasets/dataset2/annotations_aug/1_shot.json'
# result_json_file ='/root/autodl-tmp/CDFSOD/NTIRE2026_CDFSOD/datasets/dataset2/annotations_pseudo/1_shot_3.json'
the = 0.5
id = 10000

with open(traget_json_file,'r',encoding='utf-8') as load_f:
    traget_json = json.load(load_f)
with open(ano_json_file,'r',encoding='utf-8') as load_f:
    ano_json = json.load(load_f)

for predict in traget_json:
    if(predict["score"] >= the):
        temp_dict = predict
        temp_dict["id"] = id
        temp_dict["iscrowd"] = 0
        id+=1
        ano_json["annotations"].append(temp_dict)

with open(result_json_file, 'w') as write_f:
	json.dump(ano_json, write_f)