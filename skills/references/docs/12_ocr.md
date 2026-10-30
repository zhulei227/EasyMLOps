# EasyMLOps OCR 模块

OCR 模块提供两套文本检测与识别引擎：

| 引擎 | 类 | 定位 |
|------|----|------|
| **RapidOCR**（推荐） | `RapidOCRText` | PaddleOCR 模型 ONNX 化轻量推理，CPU 20-50ms 级，无深度学习框架依赖，生产级 |
| EasyOCR | `EasyOCRText` | 原型/MVP 友好，80+ 语言，但推理慢（150-400ms）且精度上限低 |

> 2025-2026 OCR 领域调研结论：RapidOCR 为 EasyOCR 的生产级替代；
> 追求 SOTA 精度/文档结构化可用 PaddleOCR-VL（0.9B，OmniDocBench 94.5%）；
> VLM-OCR（DeepSeek-OCR/GOT-OCR2.0 等）存在幻觉风险，忠实度攸关场景慎用。

## 环境依赖

```bash
# RapidOCR(推荐, 轻量)
pip install rapidocr onnxruntime
# 或 EasyOCR(较重)
pip install easyocr
```

EasyOCR 额外依赖：torch >= 1.12.0、torchvision >= 0.13.0、opencv-python-headless、Pillow < 10、numpy < 2。
RapidOCR 依赖：onnxruntime、opencv-python-headless、omegaconf（建议 >= 2.3）。

## 模块导入

```python
from easymlops.ocr import OCRPipeLine, EasyOCRText, RapidOCRText
```

## 0. 文本识别 (RapidOCRText，推荐)

```python
import pandas as pd
from easymlops.ocr import RapidOCRText

df = pd.DataFrame({"image_path": ["path/to/image.jpg"]})
pipe = RapidOCRText(text_score=0.5)
result = pipe.fit(df).transform(df)
```

**参数说明：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| model_path | str | None | 自定义模型目录（透传 model_root_dir） |
| text_score | float | 0.5 | 识别置信度阈值（低于该值的文本框被过滤） |
| gpu | bool | False | 保留接口兼容（RapidOCR 默认 ONNX CPU 后端） |

输出字段与 EasyOCRText 对齐：image_path/text/confidence/bbox_x1..bbox_y2。
`save/load` 后自动重建引擎，无需重新初始化。

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
