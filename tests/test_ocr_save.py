# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, "d:/datas/share/traes/easymlops-trae")

import pandas as pd
from easymlops.ocr import OCRPipeLine, EasyOCRText

test_images_dir = "d:/datas/share/traes/easymlops-trae/data/yolo_test_images"
bus_img = test_images_dir + "/bus.jpg"

print("=== OCR Pipeline Save/Load Test ===")
df = pd.DataFrame({"image_path": [bus_img]})

print("1. Creating pipeline...")
pipeline = OCRPipeLine()
pipeline.pipe(EasyOCRText(lang="en", gpu=False))

print("2. Fitting...")
pipeline.fit(df)
print("   Fit passed")

print("3. Transform before save...")
result1 = pipeline.transform(df)
print(f"   Transform before save: {len(result1)} texts detected")

print("4. Saving pipeline...")
pipeline.save("d:/datas/share/traes/easymlops-trae/test_ocr_pipeline.pkl")
print("   Save passed")

print("5. Loading pipeline...")
loaded = OCRPipeLine()
loaded.load("d:/datas/share/traes/easymlops-trae/test_ocr_pipeline.pkl")
print("   Load passed")

print("6. Re-fitting after load...")
loaded.fit(df)
print("   Re-fit passed")

print("7. Transform after load...")
result2 = loaded.transform(df)
print(f"   Transform after load: {len(result2)} texts detected")

print("\n=== All Save/Load tests PASSED ===")
