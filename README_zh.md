# EasyMLOps

[![Python Version](https://img.shields.io/pypi/pyversions/easymlops)](https://pypi.org/project/easymlops/)
[![License](https://img.shields.io/pypi/l/easymlops)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/yourusername/easymlops?style=social)](https://github.com/yourusername/easymlops/stargazers)

[中文版](./README.md) | English

EasyMLOps is an efficient Machine Learning Operations framework that builds modeling tasks through Pipeline approach. It supports model training, prediction, testing, feature storage, monitoring and more. Simply wrap with Flask or FastApi to deploy to production.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [Table Data Processing](#table-data-processing)
  - [NLP Text Processing](#nlp-text-processing)
  - [YOLO Vision Tasks](#yolo-vision-tasks)
  - [OCR Text Recognition](#ocr-text-recognition)
- [Modules](#modules)
- [Environment Dependencies](#environment-dependencies)
- [Documentation](#documentation)
- [License](#license)

## Features

- 🚀 **Unified Pipeline Architecture** - Clean API design with chainable calls
- 📊 **Table Data Processing** - Data cleaning, feature encoding, dimensionality reduction, feature selection, factorization machines, classification/regression
- 📝 **NLP Tasks** - Text cleaning, tokenization, feature extraction, text classification, similarity search
- 🖼️ **YOLO Vision Tasks** - Object detection, instance segmentation, image classification, pose estimation, rotated object detection
- 🔤 **OCR Text Recognition** - Supports 80+ languages
- 🤖 **AutoML** - Automated machine learning
- 💾 **Model Persistence** - Easy model save and load

## Installation

### Install from PyPI

```bash
pip install easymlops
```

### Install from source

```bash
pip install -e .
```

## Quick Start

### Table Data Processing

```python
import pandas as pd
from easymlops import TablePipeLine
from easymlops.table.preprocessing import *
from easymlops.table.encoding import *
from easymlops.table.classification import *
from easymlops.table.ensemble import *

# Load data
data = pd.read_csv("./data/demo.csv")
x_train = data[:500]
x_test = data[500:]
y_train = x_train["Survived"]
del x_train["Survived"]
del x_test["Survived"]

# Build Pipeline
table = TablePipeLine()
table.pipe(FixInput()) \
  .pipe(FillNa()) \
  .pipe(Parallel([
      OneHotEncoding(cols=["Pclass", "Sex"]),
      LabelEncoding(cols=["Sex", "Pclass"]),
      TargetEncoding(cols=["Name", "Ticket"], y=y_train)
  ])) \
  .pipe(Parallel([
      LGBMClassification(y=y_train, prefix="lgbm"),
      LogisticRegressionClassification(y=y_train, prefix="lr")
  ]))

# Train and predict
table.fit(x_train)
result = table.transform(x_test)
print(result.head(5))

# Single prediction
table.transform_single(x_test.to_dict("record")[0])
```

#### Factorization Machines (FM/FFM/DeepFM)

```python
from easymlops.table.fm import FMClassification, DeepFMClassification

# FM Classification
fm = FMClassification(
    y=label,
    cols=["feature1", "feature2", "feature3"],
    field_dims=[5, 10, 8],
    embed_dim=8,
    epochs=10
)
pipeline = TablePipeLine().pipe(fm)
pipeline.fit(df).transform(df)

# DeepFM Classification
deepfm = DeepFMClassification(
    y=label,
    cols=["feature1", "feature2", "feature3"],
    field_dims=[5, 10, 8],
    embed_dim=8,
    hidden_layers=[64, 32],
    epochs=10
)
```

#### LightGBM Leaf Node Encoding

```python
from easymlops.table.encoding import LGBMLeafEncoder

encoder = LGBMLeafEncoder(y=label, n_estimators=10, max_depth=5)
pipeline = TablePipeLine().pipe(encoder)
pipeline.fit(df)
result = pipeline.transform(df)

# Get leaf node descriptions
descriptions = encoder.describe()
```

#### Evaluation Metrics

```python
from easymlops.table.utils import (
    calc_precision_recall_at_thresholds,
    plot_pr_curve,
    calc_roc_at_thresholds,
    plot_roc_curve
)

# P-R Curve
result = calc_precision_recall_at_thresholds(y_true, y_pred, bins=10)
plot_pr_curve(y_true, y_preds_dict, save_path="pr_curve.png")

# ROC Curve
result = calc_roc_at_thresholds(y_true, y_pred, bins=10)
plot_roc_curve(y_true, y_preds_dict, save_path="roc_curve.png")
```

### NLP Text Processing

```python
from easymlops import NLPPipeline
from easymlops.nlp.preprocessing import *
from easymlops.nlp.representation import *

nlp = NLPPipeline()
nlp.pipe(FixInput()) \
  .pipe(FillNa()) \
  .pipe(SelectCols(cols=["Name"])) \
  .pipe(ReplaceDigits()) \
  .pipe(RemovePunctuation()) \
  .pipe(Word2VecModel(embedding_size=4))

nlp.fit(x_train)
result = nlp.transform(x_test)
```

### YOLO Vision Tasks

```python
from easymlops.yolo.detection import YOLODetection
from easymlops.yolo.segmentation import YOLOSegmentation
from easymlops.yolo.classification import YOLOClassification
from easymlops.yolo.pose import YOLOPose
from easymlops.yolo.obb import YOLOOBB

# Object Detection
pipe = YOLODetection(model_name="yolo11n", device="cpu")

# Instance Segmentation
pipe = YOLOSegmentation(model_name="yolo11n-seg")

# Image Classification
pipe = YOLOClassification(model_name="yolo11n-cls", imgsz=224)

# Pose Estimation
pipe = YOLOPose(model_name="yolo11n-pose")

# Rotated Object Detection
pipe = YOLOOBB(model_name="yolo11n-obb")

# Train and predict
df = pd.DataFrame({"image_path": ["path/to/image.jpg"]})
result = pipe.fit(df).transform(df)
```

### OCR Text Recognition

```python
from easymlops.ocr import EasyOCRText

df = pd.DataFrame({"image_path": ["path/to/image.jpg"]})
pipe = EasyOCRText(lang="en", gpu=False)
result = pipe.fit(df).transform(df)
```

## Modules

| Module | Description |
|--------|-------------|
| **TablePipeLine** | Table data processing: data cleaning, feature encoding, dimensionality reduction, feature selection, **Factorization Machines (FM/FFM/DeepFM)**, classification/regression, Stacking |
| **NLPPipeline** | NLP tasks: text cleaning, tokenization, feature extraction, text classification, similarity search |
| **TSPipeLine** | Time series tasks: time series processing, DNN models |
| **YOLO Module** | Vision tasks: object detection, instance segmentation, image classification, pose estimation, rotated object detection |
| **OCRPipeLine** | OCR tasks: text detection and recognition, supports 80+ languages |
| **AutoML** | Automated machine learning |

## Environment Dependencies

### YOLO Vision Tasks

```bash
pip install ultralytics-opencv-headless
# or
pip install numpy>=1.24.4,<2 torch>=1.12.0 torchvision>=0.13.0 ultralytics
```

### OCR Text Recognition

```bash
pip install easyocr
pip install "numpy<2" "Pillow<10" "torch>=1.12.0" "torchvision>=0.13.0"
```

## Documentation

For detailed usage instructions, please refer to the documentation in the [skills](./skills) directory:

- [SKILL.md](./skills/SKILL.md) - Main documentation
- [01_table.md](./skills/docs/01_table.md) - Table Data Processing
- [02_nlp.md](./skills/docs/02_nlp.md) - NLP Tasks
- [03_timeseries.md](./skills/docs/03_timeseries.md) - Time Series Tasks
- [04_pipeline.md](./skills/docs/04_pipeline.md) - Pipeline Operations
- [05_persistence.md](./skills/docs/05_persistence.md) - Model Persistence
- [06_deploy.md](./skills/docs/06_deploy.md) - Production Deployment
- [07_storage.md](./skills/docs/07_storage.md) - Feature Storage
- [08_automl.md](./skills/docs/08_automl.md) - AutoML
- [09_cheatsheet.md](./skills/docs/09_cheatsheet.md) - Quick Reference
- [10_example.md](./skills/docs/10_example.md) - Complete Examples
- [11_yolo.md](./skills/docs/11_yolo.md) - YOLO Vision Tasks
- [12_ocr.md](./skills/docs/12_ocr.md) - OCR Text Recognition

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
