import sys
import os
import re

"""
Usage:
python update_builtin.py <dataset> <shot> <round> <pseudo_json_path>
Example:
python update_builtin.py dataset2 1 1 /root/.../annotations_pseudo_1/1_shot.json
"""

if len(sys.argv) != 5:
    print("Usage: python update_builtin.py <dataset> <shot> <round> <pseudo_json_path>")
    sys.exit(1)

dataset = sys.argv[1]
shot = sys.argv[2]
round_num = sys.argv[3]   # 当前轮数
pseudo_json_path = sys.argv[4]

builtin_file = 'detectron2/data/datasets/builtin.py'

# 读取原文件
with open(builtin_file, "r") as f:
    lines = f.readlines()

new_lines = []


for line in lines:
    if dataset in line and ("annotations_aug" in line or "annotations_pseudo_" in line):
        print("FOUND LINE:", line.strip())

        old_line = line

        # 情况1：原始格式，例如：
        # "dataset2/annotations_aug/{}_shot.json".format(shot)
        # "dataset2/annotations_pseudo_{}/{}_shot.json".format(round, shot)
        line = re.sub(
            r'"[^"]*annotations[^"]*_shot\.json"\.format\([^)]*\)',
            f'"{pseudo_json_path}"',
            line
        )

        # 情况2：已经是绝对路径，例如：
        # "/root/.../annotations_pseudo_1/5_shot.json"
        line = re.sub(
            r'"/root[^"]*annotations_(?:aug|pseudo_\d+)/[^"]*_shot\.json"',
            f'"{pseudo_json_path}"',
            line
        )

        if line != old_line:
            print("REPLACED WITH:", line.strip())
        else:
            print("NO CHANGE (regex did not match this line)")

    new_lines.append(line)

with open(builtin_file, "w") as f:
    f.writelines(new_lines)

print(f"\nUpdated {builtin_file}")
print(f"Current round generated: {round_num}")
print(f"Next training will use: {pseudo_json_path}")