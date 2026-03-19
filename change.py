import os
from PIL import Image

folder = "/root/autodl-tmp/CDFSOD/NTIRE2026_CDFSOD/datasets/dataset1/train"

for root, dirs, files in os.walk(folder):
    for file in files:
        if file.lower().endswith(".png"):
            png_path = os.path.join(root, file)
            jpg_path = os.path.join(root, file.replace(".png", ".jpg"))

            img = Image.open(png_path).convert("RGB")
            img.save(jpg_path, "JPEG", quality=95)

            os.remove(png_path)

print("转换完成")