# EasyMLOps 机器学习 Pipeline 框架

EasyMLOps 是一个基于 Pipeline 模式构建的机器学习建模任务框架，提供从数据处理、特征工程、模型训练到生产部署的完整流程支持。

## 核心模块导入

```python
from easymlops import TablePipeLine, NLPPipeline, TSPipeLine
from easymlops.yolo import YOLOPipeline
from easymlops.ocr import OCRPipeLine
```

## 主要功能模块

| 模块 | 说明 |
|------|------|
| **TablePipeLine** | 表格数据处理：数据清洗、特征编码、特征降维、特征选择、**因子分解机(FM/FFM/DeepFM)**、分类回归、Stacking |
| **NLPPipeline** | NLP任务：文本清洗、分词、特征提取、文本分类、相似度检索 |
| **TSPipeLine** | 时序任务：时间序列处理、DNN模型 |
| **YOLOPipeline** | 视觉任务：目标检测、实例分割、图像分类、姿态估计、旋转目标检测 |
| **OCRPipeLine** | OCR任务：文本检测与识别，支持80+语言 |
| **AutoML** | 自动化机器学习 |

## 文档目录

- [1. Table 数据处理](./docs/01_table.md)
- [2. NLP 任务](./docs/02_nlp.md)
- [3. 时序任务](./docs/03_timeseries.md)
- [4. Pipeline 操作](./docs/04_pipeline.md)
- [5. 模型持久化](./docs/05_persistence.md)
- [6. 生产部署](./docs/06_deploy.md)
- [7. 特征存储](./docs/07_storage.md)
- [8. AutoML](./docs/08_automl.md)
- [9. 常用类速查表](./docs/09_cheatsheet.md)
- [10. 完整示例](./docs/10_example.md)
- [11. YOLO 视觉任务](./docs/11_yolo.md)
- [12. OCR 文本识别](./docs/12_ocr.md)
