import sys
sys.path.insert(0, "d:/datas/share/traes/easymlops-trae")

test_images_dir = "d:/datas/share/traes/easymlops-trae/data/yolo_test_images"
bus_img = f"{test_images_dir}/bus.jpg"

tests = []

print("1. YOLODetection", end=" ")
try:
    import pandas as pd
    from easymlops.yolo.detection import YOLODetection
    df = pd.DataFrame({"image_path": [bus_img]})
    pipe = YOLODetection(model_name="yolo11n")
    result = pipe.fit(df).transform(df)
    print(f"✅ {len(result)}")
    tests.append("YOLODetection: ✅")
except Exception as e:
    print(f"❌ {str(e)[:50]}")
    tests.append(f"YOLODetection: ❌ {str(e)[:30]}")

print("2. YOLOSegmentation", end=" ")
try:
    import pandas as pd
    from easymlops.yolo.segmentation import YOLOSegmentation
    df = pd.DataFrame({"image_path": [bus_img]})
    pipe = YOLOSegmentation(model_name="yolo11n-seg")
    result = pipe.fit(df).transform(df)
    print(f"✅ {len(result)}")
    tests.append("YOLOSegmentation: ✅")
except Exception as e:
    print(f"❌ {str(e)[:50]}")
    tests.append(f"YOLOSegmentation: ❌ {str(e)[:30]}")

print("3. YOLOClassification", end=" ")
try:
    import pandas as pd
    from easymlops.yolo.classification import YOLOClassification
    df = pd.DataFrame({"image_path": [bus_img]})
    pipe = YOLOClassification(model_name="yolo11n-cls")
    result = pipe.fit(df).transform(df)
    print(f"✅ {len(result)}")
    tests.append("YOLOClassification: ✅")
except Exception as e:
    print(f"❌ {str(e)[:50]}")
    tests.append(f"YOLOClassification: ❌ {str(e)[:30]}")

print("4. YOLOPose", end=" ")
try:
    import pandas as pd
    from easymlops.yolo.pose import YOLOPose
    df = pd.DataFrame({"image_path": [bus_img]})
    pipe = YOLOPose(model_name="yolo11n-pose")
    result = pipe.fit(df).transform(df)
    print(f"✅ {len(result)}")
    tests.append("YOLOPose: ✅")
except Exception as e:
    print(f"❌ {str(e)[:50]}")
    tests.append(f"YOLOPose: ❌ {str(e)[:30]}")

print("5. YOLOOBB", end=" ")
try:
    import pandas as pd
    from easymlops.yolo.obb import YOLOOBB
    df = pd.DataFrame({"image_path": [bus_img]})
    pipe = YOLOOBB(model_name="yolo11n-obb")
    result = pipe.fit(df).transform(df)
    print(f"✅ {len(result)}")
    tests.append("YOLOOBB: ✅")
except Exception as e:
    print(f"❌ {str(e)[:50]}")
    tests.append(f"YOLOOBB: ❌ {str(e)[:30]}")

print("6. YOLOTrain", end=" ")
try:
    from easymlops.yolo.train import YOLOTrain
    pipe = YOLOTrain(data="dataset.yaml", epochs=1)
    print("✅")
    tests.append("YOLOTrain: ✅")
except Exception as e:
    print(f"❌ {str(e)[:50]}")
    tests.append(f"YOLOTrain: ❌ {str(e)[:30]}")

print("7. YOLOVal", end=" ")
try:
    from easymlops.yolo.train import YOLOVal
    pipe = YOLOVal(data="dataset.yaml")
    print("✅")
    tests.append("YOLOVal: ✅")
except Exception as e:
    print(f"❌ {str(e)[:50]}")
    tests.append(f"YOLOVal: ❌ {str(e)[:30]}")

print("8. YOLOExport", end=" ")
try:
    from easymlops.yolo.train import YOLOExport
    pipe = YOLOExport(format="onnx")
    print("✅")
    tests.append("YOLOExport: ✅")
except Exception as e:
    print(f"❌ {str(e)[:50]}")
    tests.append(f"YOLOExport: ❌ {str(e)[:30]}")

print("9. YOLOPipeline", end=" ")
try:
    from easymlops.yolo.core.pipeline import YOLOPipeline
    pipe = YOLOPipeline()
    print("✅")
    tests.append("YOLOPipeline: ✅")
except Exception as e:
    print(f"❌ {str(e)[:50]}")
    tests.append(f"YOLOPipeline: ❌ {str(e)[:30]}")

print("10. fit_transform", end=" ")
try:
    import pandas as pd
    from easymlops.yolo.detection import YOLODetection
    df = pd.DataFrame({"image_path": [bus_img]})
    pipe = YOLODetection(model_name="yolo11n")
    result = pipe.fit_transform(df)
    print(f"✅ {len(result)}")
    tests.append("fit_transform: ✅")
except Exception as e:
    print(f"❌ {str(e)[:50]}")
    tests.append(f"fit_transform: ❌ {str(e)[:30]}")

print("11. save/load", end=" ")
try:
    import pandas as pd
    import os
    from easymlops.yolo.detection import YOLODetection
    df = pd.DataFrame({"image_path": [bus_img]})
    pipe = YOLODetection(model_name="yolo11n")
    pipe.fit(df)
    save_path = "d:/datas/share/traes/easymlops-trae/test_yolo_pipe.pkl"
    pipe.save(save_path)
    loaded_pipe = YOLODetection()
    loaded_pipe.load(save_path)
    os.remove(save_path)
    print("✅")
    tests.append("save/load: ✅")
except Exception as e:
    print(f"❌ {str(e)[:50]}")
    tests.append(f"save/load: ❌ {str(e)[:30]}")

print("\n" + "="*50)
print("测试结果汇总:")
for t in tests:
    print(t)
