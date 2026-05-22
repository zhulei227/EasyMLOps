import sys
sys.path.insert(0, "d:/datas/share/traes/easymlops-trae")

test_images_dir = "d:/datas/share/traes/easymlops-trae/data/yolo_test_images"

print("="*50)
print("1. 测试 YOLODetection")
print("="*50)
try:
    import pandas as pd
    from easymlops.yolo.detection import YOLODetection
    
    df = pd.DataFrame({"image_path": [f"{test_images_dir}/bus.jpg"]})
    pipe = YOLODetection(model_name="yolo11n")
    result = pipe.fit(df).transform(df)
    print(f"✅ 成功，检测到 {len(result)} 个目标")
except Exception as e:
    print(f"❌ 失败: {e}")

print("\n2. 测试 YOLOSegmentation")
try:
    import pandas as pd
    from easymlops.yolo.segmentation import YOLOSegmentation
    
    df = pd.DataFrame({"image_path": [f"{test_images_dir}/bus.jpg"]})
    pipe = YOLOSegmentation(model_name="yolo11n-seg")
    result = pipe.fit(df).transform(df)
    print(f"✅ 成功，分割到 {len(result)} 个目标")
except Exception as e:
    print(f"❌ 失败: {e}")

print("\n3. 测试 YOLOClassification")
try:
    import pandas as pd
    from easymlops.yolo.classification import YOLOClassification
    
    df = pd.DataFrame({"image_path": [f"{test_images_dir}/bus.jpg"]})
    pipe = YOLOClassification(model_name="yolo11n-cls")
    result = pipe.fit(df).transform(df)
    print(f"✅ 成功，分类结果 {len(result)} 行")
except Exception as e:
    print(f"❌ 失败: {e}")

print("\n4. 测试 YOLOPose")
try:
    import pandas as pd
    from easymlops.yolo.pose import YOLOPose
    
    df = pd.DataFrame({"image_path": [f"{test_images_dir}/bus.jpg"]})
    pipe = YOLOPose(model_name="yolo11n-pose")
    result = pipe.fit(df).transform(df)
    print(f"✅ 成功，检测到 {len(result)} 个姿态")
except Exception as e:
    print(f"❌ 失败: {e}")

print("\n5. 测试 YOLOOBB")
try:
    import pandas as pd
    from easymlops.yolo.obb import YOLOOBB
    
    df = pd.DataFrame({"image_path": [f"{test_images_dir}/bus.jpg"]})
    pipe = YOLOOBB(model_name="yolo11n-obb")
    result = pipe.fit(df).transform(df)
    print(f"✅ 成功，检测到 {len(result)} 个目标")
except Exception as e:
    print(f"❌ 失败: {e}")

print("\n6. 测试 YOLOTrain")
try:
    from easymlops.yolo.train import YOLOTrain
    pipe = YOLOTrain(data="dataset.yaml", epochs=1)
    print("✅ 初始化成功")
except Exception as e:
    print(f"❌ 失败: {e}")

print("\n7. 测试 YOLOVal")
try:
    from easymlops.yolo.train import YOLOVal
    pipe = YOLOVal(data="dataset.yaml")
    print("✅ 初始化成功")
except Exception as e:
    print(f"❌ 失败: {e}")

print("\n8. 测试 YOLOExport")
try:
    from easymlops.yolo.train import YOLOExport
    pipe = YOLOExport(format="onnx")
    print("✅ 初始化成功")
except Exception as e:
    print(f"❌ 失败: {e}")

print("\n9. 测试 YOLOPipeline")
try:
    from easymlops.yolo.core.pipeline import YOLOPipeline
    pipe = YOLOPipeline()
    print("✅ 初始化成功")
except Exception as e:
    print(f"❌ 失败: {e}")

print("\n10. 测试 fit_transform")
try:
    import pandas as pd
    from easymlops.yolo.detection import YOLODetection
    
    df = pd.DataFrame({"image_path": [f"{test_images_dir}/bus.jpg"]})
    pipe = YOLODetection(model_name="yolo11n")
    result = pipe.fit_transform(df)
    print(f"✅ 成功，检测到 {len(result)} 个目标")
except Exception as e:
    print(f"❌ 失败: {e}")

print("\n11. 测试 save/load")
try:
    import pandas as pd
    import os
    from easymlops.yolo.detection import YOLODetection
    
    df = pd.DataFrame({"image_path": [f"{test_images_dir}/bus.jpg"]})
    pipe = YOLODetection(model_name="yolo11n")
    pipe.fit(df)
    
    save_path = "d:/datas/share/traes/easymlops-trae/test_yolo_pipe.pkl"
    pipe.save(save_path)
    
    loaded_pipe = YOLODetection()
    loaded_pipe.load(save_path)
    
    os.remove(save_path)
    print("✅ save/load 成功")
except Exception as e:
    print(f"❌ 失败: {e}")

print("\n全部测试完成!")
