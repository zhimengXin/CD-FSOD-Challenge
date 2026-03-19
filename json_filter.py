import json

# 读取原始json
with open("/root/autodl-tmp/CDFSOD/NTIRE2026_CDFSOD/datasets/dataset3/annotations/instances_test2017_3.json", "r") as f:
    data = json.load(f)

# 只保留 annotations
annotations = data["annotations"]

# 删除字段
for ann in annotations:
    ann.pop("segmentation", None)
    ann.pop("attributes", None)

# 保存结果
with open("/root/autodl-tmp/CDFSOD/NTIRE2026_CDFSOD/datasets/dataset3/annotations/annotations_only.json", "w") as f:
    json.dump({"annotations": annotations}, f, indent=4)

print("Done.")