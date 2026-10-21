# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, "d:/datas/share/traes/easymlops-trae")

import pandas as pd
from easymlops.yolo import YOLOPipeline, YOLODetection

test_images_dir = "d:/datas/share/traes/easymlops-trae/data/yolo_test_images"
bus_img = test_images_dir + "/bus.jpg"

print("=== YOLO Pipeline Auto Test ===")
df = pd.DataFrame({"image_path": [bus_img]})

print("1. Creating pipeline...")
pipeline = YOLOPipeline()
pipeline.pipe(YOLODetection(model_name="yolo11n", device="cpu"))

print("2. Fitting...")
pipeline.fit(df)
print("   Fit passed")

print("3. Transform...")
result = pipeline.transform(df)
print(f"   Transform passed: {len(result)} detections")

print("\n=== Testing auto_test ===")
pipeline.auto_test(df, sample=1)
print("   auto_test passed")

print("\n=== Testing save ===")
pipeline.save("d:/datas/share/traes/easymlops-trae/test_yolo_pipeline.pkl")
print("   Save passed")

print("\n=== Testing load ===")
loaded = YOLOPipeline()
loaded.load("d:/datas/share/traes/easymlops-trae/test_yolo_pipeline.pkl")
print("   Load passed")

print("\n=== Testing re-fit after load ===")
loaded.fit(df)
print("   Re-fit passed")

print("\n=== Testing transform after load ===")
result2 = loaded.transform(df)
print(f"   Transform after load: {len(result2)} detections")

print("\n=== All YOLO tests PASSED ===")
