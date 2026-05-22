import os
import urllib.request
import pandas as pd
import sys

sys.path.insert(0, "d:/datas/share/traes/easymlops-trae")

from easymlops.yolo.core.pipe import YOLOPipeObjectBase
from easymlops.yolo.detection import YOLODetection
from easymlops.yolo.segmentation import YOLOSegmentation
from easymlops.yolo.classification import YOLOClassification

test_images_dir = "d:/datas/share/traes/easymlops-trae/data/yolo_test_images"
os.makedirs(test_images_dir, exist_ok=True)

sample_images = {
    "bus.jpg": "https://ultralytics.com/images/bus.jpg",
    "zidane.jpg": "https://ultralytics.com/images/zidane.jpg"
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

print("\n测试 YOLODetection...")
detection_pipe = YOLODetection(model_name="yolo11n")
df = pd.DataFrame({
    "image_path": [
        os.path.join(test_images_dir, "bus.jpg"),
        os.path.join(test_images_dir, "zidane.jpg")
    ]
})
detection_results = detection_pipe.fit(df).transform(df)
print(detection_results)
print(f"检测到 {len(detection_results)} 个目标")

print("\n测试 YOLOSegmentation...")
segment_pipe = YOLOSegmentation(model_name="yolo11n-seg")
segment_results = segment_pipe.fit(df).transform(df)
print(segment_results)
print(f"分割到 {len(segment_results)} 个目标")

print("\n测试 YOLOClassification...")
classify_pipe = YOLOClassification(model_name="yolo11n-cls")
classify_results = classify_pipe.fit(df).transform(df)
print(classify_results)

print("\n✅ 所有测试通过!")
