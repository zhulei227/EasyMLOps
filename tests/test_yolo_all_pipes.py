import sys
import os

sys.path.insert(0, "d:/datas/share/traes/easymlops-trae")

import pandas as pd

test_images_dir = "d:/datas/share/traes/easymlops-trae/data/yolo_test_images"

sample_images = {
    "bus.jpg": "https://ultralytics.com/images/bus.jpg",
    "zidane.jpg": "https://ultralytics.com/images/zidane.jpg"
}

df = pd.DataFrame({"image_path": [os.path.join(test_images_dir, f) for f in sample_images.keys()]})

results = []

print("1. YOLODetection fit/transform", end="... ")
try:
    from easymlops.yolo.detection import YOLODetection
    pipe = YOLODetection(model_name="yolo11n")
    result = pipe.fit(df).transform(df)
    results.append(f"✅ 检测到 {len(result)} 个目标")
except Exception as e:
    results.append(f"❌ {e}")

print("2. YOLOSegmentation fit/transform", end="... ")
try:
    from easymlops.yolo.segmentation import YOLOSegmentation
    pipe = YOLOSegmentation(model_name="yolo11n-seg")
    result = pipe.fit(df).transform(df)
    results.append(f"✅ 分割到 {len(result)} 个目标")
except Exception as e:
    results.append(f"❌ {e}")

print("3. YOLOClassification fit/transform", end="... ")
try:
    from easymlops.yolo.classification import YOLOClassification
    pipe = YOLOClassification(model_name="yolo11n-cls")
    result = pipe.fit(df).transform(df)
    results.append(f"✅ 分类结果 {len(result)} 行")
except Exception as e:
    results.append(f"❌ {e}")

print("4. YOLOPose fit/transform", end="... ")
try:
    from easymlops.yolo.pose import YOLOPose
    pipe = YOLOPose(model_name="yolo11n-pose")
    result = pipe.fit(df).transform(df)
    results.append(f"✅ 检测到 {len(result)} 个姿态")
except Exception as e:
    results.append(f"❌ {e}")

print("5. YOLOOBB fit/transform", end="... ")
try:
    from easymlops.yolo.obb import YOLOOBB
    pipe = YOLOOBB(model_name="yolo11n-obb")
    result = pipe.fit(df).transform(df)
    results.append(f"✅ 检测到 {len(result)} 个目标")
except Exception as e:
    results.append(f"❌ {e}")

print("6. YOLOTrain init", end="... ")
try:
    from easymlops.yolo.train import YOLOTrain
    pipe = YOLOTrain(data="dataset.yaml", epochs=1)
    results.append("✅ 初始化成功")
except Exception as e:
    results.append(f"❌ {e}")

print("7. YOLOVal init", end="... ")
try:
    from easymlops.yolo.train import YOLOVal
    pipe = YOLOVal(data="dataset.yaml")
    results.append("✅ 初始化成功")
except Exception as e:
    results.append(f"❌ {e}")

print("8. YOLOExport init", end="... ")
try:
    from easymlops.yolo.train import YOLOExport
    pipe = YOLOExport(format="onnx")
    results.append("✅ 初始化成功")
except Exception as e:
    results.append(f"❌ {e}")

print("9. YOLOPipeline init", end="... ")
try:
    from easymlops.yolo.core.pipeline import YOLOPipeline
    pipe = YOLOPipeline()
    results.append("✅ 初始化成功")
except Exception as e:
    results.append(f"❌ {e}")

print("10. YOLODetection fit_transform", end="... ")
try:
    from easymlops.yolo.detection import YOLODetection
    pipe = YOLODetection(model_name="yolo11n")
    result = pipe.fit_transform(df)
    results.append(f"✅ 检测到 {len(result)} 个目标")
except Exception as e:
    results.append(f"❌ {e}")

print("11. YOLODetection save/load", end="... ")
try:
    from easymlops.yolo.detection import YOLODetection
    import pickle
    
    pipe = YOLODetection(model_name="yolo11n")
    pipe.fit(df)
    
    save_path = "d:/datas/share/traes/easymlops-trae/test_yolo_pipe.pkl"
    pipe.save(save_path)
    
    loaded_pipe = YOLODetection()
    loaded_pipe.load(save_path)
    
    os.remove(save_path)
    results.append("✅ save/load 成功")
except Exception as e:
    results.append(f"❌ {e}")

with open("d:/datas/share/traes/easymlops-trae/test_results.txt", "w", encoding="utf-8") as f:
    f.write("YOLO模块测试结果\n")
    f.write("="*50 + "\n")
    for i, r in enumerate(results, 1):
        f.write(f"{i}. {r}\n")
    f.write("="*50 + "\n")
    f.write("测试完成！\n")

print("\n测试完成，结果已保存到 test_results.txt")
