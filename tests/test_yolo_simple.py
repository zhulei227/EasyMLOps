import os
import urllib.request
import pandas as pd
from ultralytics import YOLO

test_images_dir = "d:/datas/share/traes/easymlops-trae/data/yolo_test_images"
os.makedirs(test_images_dir, exist_ok=True)

sample_images = {
    "bus.jpg": "https://ultralytics.com/images/bus.jpg",
}

print("下载测试图片...")
for filename, url in sample_images.items():
    filepath = os.path.join(test_images_dir, filename)
    if not os.path.exists(filepath):
        print(f"  下载 {filename}...")
        urllib.request.urlretrieve(url, filepath)
        print(f"  完成: {filename}")
    else:
        print(f"  已存在: {filename}")

print("\n测试 YOLO Detection (使用yolov8n)...")
model = YOLO("yolov8n.pt")
image_path = os.path.join(test_images_dir, "bus.jpg")

result = model.predict(image_path, verbose=True)
print("Result:", result)
print(f"\n✅ YOLO功能测试完成!")
