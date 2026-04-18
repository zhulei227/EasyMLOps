# EasyMLOps OCR 模块

OCR 模块基于 EasyOCR 库，提供文本检测与识别功能，支持 80+ 语言。

## 环境依赖

```bash
pip install easyocr
```

注意：EasyOCR 需要以下依赖：
- torch >= 1.12.0
- torchvision >= 0.13.0
- opencv-python-headless
- Pillow < 10
- numpy < 2

## 模块导入

```python
from easymlops.ocr import OCRPipeLine, EasyOCRText
```

## 1. 文本识别 (EasyOCRText)

基于 EasyOCR 进行文本检测和识别。

```python
import pandas as pd
from easymlops.ocr import EasyOCRText

df = pd.DataFrame({"image_path": ["path/to/image.jpg"]})
pipe = EasyOCRText(lang="en", gpu=False)
result = pipe.fit(df).transform(df)
print(result)
```

**参数说明：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| model_name | str | "easyocr" | 模型名称 |
| model_path | str | None | 模型文件路径 |
| lang | str/list | "en" | 语言代码，支持单语言如 'en' 或多语言如 ['en', 'ch_sim', 'ja'] |
| gpu | bool | True | 是否使用 GPU 加速 |
| batch_size | int | 1 | 批处理大小 |
| detail | int | 1 | 返回详细程度，0-简单结果，1-详细结果 |
| text_threshold | float | 0.7 | 文本检测阈值 |
| link_threshold | float | 0.4 | 文本链接阈值 |
| canvas_size | int | 2560 | 图像处理最大尺寸 |
| mag_ratio | float | 1.0 | 图像放大比例 |

**返回字段：**

- image_path: 图像路径
- text: 识别文本
- confidence: 置信度
- bbox_x1, bbox_y1, bbox_x2, bbox_y2: 边界框坐标

## 2. Pipeline 组合

```python
from easymlops.ocr import OCRPipeLine

pipeline = OCRPipeLine(pipes=[
    EasyOCRText(lang="en"),
])
result = pipeline.fit(df).transform(df)
```

## 3. 完整示例

```python
import pandas as pd
from easymlops.ocr import EasyOCRText

df = pd.DataFrame({
    "image_path": ["image1.jpg", "image2.jpg"]
})

pipe = EasyOCRText(lang=["en", "ch_sim"], gpu=False)
result = pipe.fit(df).transform(df)
print(f"检测到 {len(result)} 个文本区域")

single_result = pipe.transform_single({"image_path": "image1.jpg"})
print(f"合并文本: {single_result['text']}")
```

## 4. 支持的语言

EasyOCR 支持 80+ 语言，常用语言代码：

| 语言 | 代码 |
|------|------|
| 英语 | 'en' |
| 简体中文 | 'ch_sim' |
| 繁体中文 | 'ch_tra' |
| 日语 | 'ja' |
| 韩语 | 'ko' |
| 德语 | 'de' |
| 法语 | 'fr' |
| 西班牙语 | 'es' |

多语言示例：`lang=['en', 'ch_sim', 'ja']`

## 注意事项

1. **首次使用**：首次运行时会自动下载模型文件（约 140MB）
2. **GPU 加速**：有 GPU 的机器建议设置 `gpu=True` 以提高速度
3. **CPU 模式**：无 GPU 时设置 `gpu=False`，运行较慢
4. **语言支持**：确保选择正确的语言代码，否则会识别失败
5. **依赖版本**：
   - numpy < 2
   - Pillow < 10
   - torch >= 1.12.0
