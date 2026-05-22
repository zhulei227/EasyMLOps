# EasyMLOps YOLO 模块

YOLO 模块基于 Ultralytics YOLO 模型，提供目标检测、实例分割、图像分类、姿态估计、旋转目标检测等功能。

## 环境依赖

```bash
pip install ultralytics-opencv-headless
# 或
pip install numpy>=1.24.4,<2 torch>=1.12.0 torchvision>=0.13.0 ultralytics
```

## 模块导入

```python
from easymlops.yolo.detection import YOLODetection
from easymlops.yolo.segmentation import YOLOSegmentation
from easymlops.yolo.classification import YOLOClassification
from easymlops.yolo.pose import YOLOPose
from easymlops.yolo.obb import YOLOOBB
from easymlops.yolo.train import YOLOTrain, YOLOVal, YOLOExport
from easymlops.yolo.core.pipeline import YOLOPipeline
```

## 1. 目标检测 (YOLODetection)

基于 YOLO 模型进行目标检测。

```python
import pandas as pd
from easymlops.yolo.detection import YOLODetection

df = pd.DataFrame({"image_path": ["path/to/image.jpg"]})
pipe = YOLODetection(model_name="yolo11n", device="cpu")
result = pipe.fit(df).transform(df)
print(result)
```

**参数说明：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| model_name | str | "yolo11n" | 模型名称，如 yolo11n, yolo11s, yolo11m 等 |
| model_path | str | None | 模型文件路径，优先于 model_name |
| conf | float | 0.25 | 置信度阈值 |
| iou | float | 0.45 | NMS IoU 阈值 |
| imgsz | int | 640 | 输入图像尺寸 |
| device | str | "cpu" | 运行设备，cpu/cuda |
| save | bool | False | 是否保存结果 |
| show | bool | False | 是否显示结果 |

**返回字段：**

- image_path: 图像路径
- class_id: 类别 ID
- class_name: 类别名称
- confidence: 置信度
- bbox_x1, bbox_y1, bbox_x2, bbox_y2: 边界框坐标

## 2. 实例分割 (YOLOSegmentation)

基于 YOLO 模型进行实例分割。

```python
from easymlops.yolo.segmentation import YOLOSegmentation

pipe = YOLOSegmentation(model_name="yolo11n-seg")
result = pipe.fit(df).transform(df)
```

**返回字段：**

- image_path: 图像路径
- class_id: 类别 ID
- class_name: 类别名称
- confidence: 置信度
- bbox_x1, bbox_y1, bbox_x2, bbox_y2: 边界框坐标
- mask: 分割掩码数据

## 3. 图像分类 (YOLOClassification)

基于 YOLO 模型进行图像分类。

```python
from easymlops.yolo.classification import YOLOClassification

pipe = YOLOClassification(model_name="yolo11n-cls", imgsz=224)
result = pipe.fit(df).transform(df)
```

**参数说明：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| model_name | str | "yolo11n-cls" | 分类模型名称 |
| model_path | str | None | 模型文件路径 |
| conf | float | 0.25 | 置信度阈值 |
| imgsz | int | 224 | 输入图像尺寸 |
| device | str | "cpu" | 运行设备 |
| top_k | int | 5 | 返回前 k 个结果 |

**返回字段：**

- image_path: 图像路径
- top1_class: 预测类别
- top1_conf: 最高置信度
- top5_class: 前5类别
- top5_confs: 前5置信度

## 4. 姿态估计 (YOLOPose)

基于 YOLO 模型进行人体姿态估计。

```python
from easymlops.yolo.pose import YOLOPose

pipe = YOLOPose(model_name="yolo11n-pose")
result = pipe.fit(df).transform(df)
```

**返回字段：**

- image_path: 图像路径
- class_id: 类别 ID
- confidence: 置信度
- bbox: 边界框
- keypoints: 关键点坐标 (x, y, conf)

## 5. 旋转目标检测 (YOLOOBB)

基于 YOLO 模型进行旋转目标检测。

```python
from easymlops.yolo.obb import YOLOOBB

pipe = YOLOOBB(model_name="yolo11n-obb")
result = pipe.fit(df).transform(df)
```

**返回字段：**

- image_path: 图像路径
- class_id: 类别 ID
- class_name: 类别名称
- confidence: 置信度
- rotated_bbox: 旋转边界框 (x, y, w, h, angle)

## 6. 模型训练 (YOLOTrain)

```python
from easymlops.yolo.train import YOLOTrain

pipe = YOLOTrain(
    model_name="yolo11n",
    data="dataset.yaml",
    epochs=100,
    batch=16,
    device="cpu"
)
pipe.fit(df)
```

## 7. 模型验证 (YOLOVal)

```python
from easymlops.yolo.train import YOLOVal

pipe = YOLOVal(
    model_path="path/to/model.pt",
    data="dataset.yaml"
)
metrics = pipe.fit(df).transform(df)
```

## 8. 模型导出 (YOLOExport)

```python
from easymlops.yolo.train import YOLOExport

pipe = YOLOExport(
    model_path="path/to/model.pt",
    format="onnx"
)
pipe.fit(df)
```

支持格式：onnx, torchscript, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle

## 9. Pipeline 组合

```python
from easymlops.yolo.core.pipeline import YOLOPipeline

pipeline = YOLOPipeline(pipes=[
    YOLODetection(model_name="yolo11n"),
    # 其他 pipe
])
```

## 10. 完整示例

```python
import pandas as pd
from easymlops.yolo.detection import YOLODetection
from easymlops.yolo.segmentation import YOLOSegmentation

df = pd.DataFrame({
    "image_path": ["bus.jpg", "zidane.jpg"]
})

detect_pipe = YOLODetection(model_name="yolo11n")
detect_results = detect_pipe.fit(df).transform(df)
print(f"检测到 {len(detect_results)} 个目标")

segment_pipe = YOLOSegmentation(model_name="yolo11n-seg")
segment_results = segment_pipe.fit(df).transform(df)
print(f"分割到 {len(segment_results)} 个目标")
```

## 注意事项

1. **device 参数**：在 CPU 环境下使用时，建议设置 `device="cpu"`
2. **模型下载**：首次使用时会自动从 Ultralytics 下载模型
3. **数据格式**：输入 DataFrame 必须包含 `image_path` 列
4. **NumPy 版本**：建议使用 `numpy<2` 以兼容 ultralytics

## 11. 模型保存与加载

YOLO Pipeline 支持模型的保存和加载。由于 YOLO 模型（Ultralytics）无法直接序列化，保存的是参数配置，加载后需要重新 fit 以恢复模型。

```python
import pandas as pd
from easymlops.yolo import YOLOPipeline, YOLODetection

# 创建并训练 pipeline
df = pd.DataFrame({"image_path": ["bus.jpg"]})
pipeline = YOLOPipeline()
pipeline.pipe(YOLODetection(model_name="yolo11n", device="cpu"))
pipeline.fit(df)

# 保存模型
pipeline.save("yolo_pipeline.pkl")

# 加载模型
loaded = YOLOPipeline()
loaded.load("yolo_pipeline.pkl")

# 加载后需要重新 fit
loaded.fit(df)

# 使用加载的模型进行预测
result = loaded.transform(df)
```

## 12. 自动测试

YOLO Pipeline 提供了 `auto_test` 方法用于自动测试功能。由于 YOLO transform 可能输出一对多（一张图片多个检测结果），该方法跳过标准的一致性检查，仅验证功能可用性。

```python
import pandas as pd
from easymlops.yolo import YOLOPipeline, YOLODetection

df = pd.DataFrame({"image_path": ["bus.jpg"]})
pipeline = YOLOPipeline()
pipeline.pipe(YOLODetection(model_name="yolo11n", device="cpu"))

# 自动测试
pipeline.auto_test(df, sample=1)
```

`auto_test` 方法会依次测试：
- Fit 功能
- Transform 功能
- Get params 功能
- Set params 功能
