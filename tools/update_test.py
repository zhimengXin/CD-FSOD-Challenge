import yaml
import sys

if len(sys.argv) != 3:
    print("Usage: python update_test.py <config_file> <pseudo_test_name>")
    sys.exit(1)

config_file = sys.argv[1]
pseudo_test = sys.argv[2]

# 使用 full_load 读取 YAML（支持 !!python/tuple）
with open(config_file, "r") as f:
    cfg = yaml.full_load(f)

# 修改 DATASETS.TEST，保持 tuple
cfg["DATASETS"]["TEST"] = (pseudo_test,)  # Python tuple

# 写回 YAML
with open(config_file, "w") as f:
    yaml.dump(cfg, f, default_flow_style=False)

print(f"Updated DATASETS.TEST to ({pseudo_test},) in {config_file}")
