import os
import urllib.request
import pandas as pd
from ultralytics import YOLO

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

print("\n测试 YOLO Detection...")
model = YOLO("yolo11n.pt")
df = pd.DataFrame({
    "image_path": [
        os.path.join(test_images_dir, "bus.jpg"),
        os.path.join(test_images_dir, "zidane.jpg")
    ]
})

results_list = []
for idx, row in df.iterrows():
    image_path = row["image_path"]
    result = model.predict(image_path, verbose=False)[0]
    boxes = result.boxes
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            results_list.append({
                "image_path": image_path,
                "class_id": int(box.cls[0]),
                "class_name": result.names[int(box.cls[0])],
                "confidence": float(box.conf[0]),
                "bbox_x1": float(box.xyxy[0][0]),
                "bbox_y1": float(box.xyxy[0][1]),
                "bbox_x2": float(box.xyxy[0][2]),
                "bbox_y2": float(box.xyxy[0][3])
            })

detection_results = pd.DataFrame(results_list)
print(detection_results)
print(f"检测到 {len(detection_results)} 个目标")

print("\n测试 YOLO Segmentation...")
model_seg = YOLO("yolo11n-seg.pt")
results_list_seg = []
for idx, row in df.iterrows():
    image_path = row["image_path"]
    result = model_seg.predict(image_path, verbose=False)[0]
    boxes = result.boxes
    masks = result.masks
    if boxes is not None and len(boxes) > 0:
        for i, box in enumerate(boxes):
            mask_data = None
            if masks is not None and i < len(masks):
                mask_data = str(masks.data[i].shape)
            results_list_seg.append({
                "image_path": image_path,
                "class_id": int(box.cls[0]),
                "class_name": result.names[int(box.cls[0])],
                "confidence": float(box.conf[0]),
                "mask_shape": mask_data
            })

segment_results = pd.DataFrame(results_list_seg)
print(segment_results)
print(f"分割到 {len(segment_results)} 个目标")

print("\n测试 YOLO Classification...")
model_cls = YOLO("yolo11n-cls.pt")
results_list_cls = []
for idx, row in df.iterrows():
    image_path = row["image_path"]
    result = model_cls.predict(image_path, verbose=False)[0]
    probs = result.probs
    if probs is not None:
        results_list_cls.append({
            "image_path": image_path,
            "top1_class": result.names[probs.top1],
            "top1_conf": float(probs.top1conf)
        })

classify_results = pd.DataFrame(results_list_cls)
print(classify_results)

print("\n✅ 所有YOLO功能测试通过!")
