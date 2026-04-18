# EasyMLOps

[![Python Version](https://img.shields.io/pypi/pyversions/easymlops)](https://pypi.org/project/easymlops/)
[![License](https://img.shields.io/pypi/l/easymlops)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/zhulei227/easymlops?style=social)](https://github.com/zhulei227/easymlops/stargazers)

[中文版](./README_zh.md) | English

EasyMLOps is an efficient Machine Learning Operations framework that builds modeling tasks through Pipeline approach. It supports model training, prediction, testing, feature storage, monitoring and more. Simply wrap with Flask or FastApi to deploy to production.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Module Details](#module-details)
  - [Table Pipeline](#table-pipeline)
  - [NLP Pipeline](#nlp-pipeline)
  - [Time Series Pipeline](#time-series-pipeline)
  - [YOLO Vision](#yolo-vision)
  - [OCR Module](#ocr-module)
  - [AutoML](#automl)
- [Core Classes Reference](#core-classes-reference)
- [Environment Dependencies](#environment-dependencies)
- [Documentation](#documentation)
- [License](#license)

## Overview

EasyMLOps provides a unified Pipeline-based architecture for machine learning workflows. It simplifies the process of building, training, evaluating, and deploying ML models through a chainable API design.

### Core Concepts

- **Pipe**: Individual processing units that transform data
- **Pipeline**: A chain of pipes that process data sequentially
- **Parallel**: Execute multiple pipes concurrently
- **Stacking**: Combine multiple models for ensemble learning

## Features

- 🚀 **Unified Pipeline Architecture** - Clean API design with chainable calls
- 📊 **Table Data Processing** - Data cleaning, feature encoding, dimensionality reduction, feature selection, factorization machines, classification/regression
- 📝 **NLP Tasks** - Text cleaning, tokenization, feature extraction, text classification, similarity search
- 🖼️ **YOLO Vision Tasks** - Object detection, instance segmentation, image classification, pose estimation, rotated object detection
- 🔤 **OCR Text Recognition** - Supports 80+ languages
- 🤖 **AutoML** - Automated machine learning
- 💾 **Model Persistence** - Easy model save and load

## Architecture

```
easymlops/
├── core/           # Core pipeline base classes
├── table/          # Tabular data processing
│   ├── preprocessing/    # Data preprocessing
│   ├── encoding/         # Feature encoding
│   ├── classification/   # Classification models
│   ├── regression/       # Regression models
│   ├── ensemble/         # Ensemble methods
│   ├── fm/               # Factorization Machines
│   ├── decomposition/    # Dimensionality reduction
│   ├── feature_selection/# Feature selection
│   ├── strategy/         # Modeling strategies
│   ├── storage/          # Feature storage
│   └── utils/            # Utility functions
├── nlp/            # Natural language processing
├── ts/             # Time series
├── yolo/           # YOLO vision tasks
├── ocr/            # OCR text recognition
└── automl/         # AutoML
```

## Installation

### From PyPI

```bash
pip install easymlops
```

### From Source

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

## Module Details

### Table Pipeline

The Table Pipeline is the core module for tabular data processing.

#### 1. Preprocessing (`easymlops.table.preprocessing`)

| Class | Description |
|-------|-------------|
| **PreprocessBase** | Base class for all preprocessing pipes |
| **FixInput** | Ensures input is DataFrame format |
| **ReName** | Rename columns |
| **DropCols** | Remove specified columns |
| **SelectCols** | Select specific columns |
| **DoNoThing** | No-operation pass-through |

#### 2. One-Variable Operations (`easymlops.table.preprocessing.onevar_operation`)

| Class | Description |
|-------|-------------|
| **Replace** | Replace values in a column |
| **ClipString** | Clip string values |
| **FillNa** | Fill missing values |
| **IsNull** | Mark null values |
| **IsNotNull** | Mark non-null values |
| **TransToCategory** | Convert to category type |
| **TransToFloat** | Convert to float type |
| **TransToInt** | Convert to integer type |
| **TransToLower** | Convert to lowercase |
| **TransToUpper** | Convert to uppercase |
| **Abs** | Absolute value |
| **MapValues** | Map values using dictionary |
| **Clip** | Clip values to range |
| **MinMaxScaler** | Min-Max normalization |
| **Normalizer** | L1/L2 normalization |
| **Bins** | Discretize continuous values |
| **Tanh** | Hyperbolic tangent activation |
| **Relu** | Rectified linear unit |
| **Sigmoid** | Sigmoid activation |
| **Swish** | Swish activation |
| **DateMonthInfo** | Extract month from date |
| **DateHourInfo** | Extract hour from date |
| **DateMinuteInfo** | Extract minute from date |
| **DateTotalMinuteInfo** | Extract total minutes from date |

#### 3. Two-Variable Operations (`easymlops.table.preprocessing.bivar_operation`)

| Class | Description |
|-------|-------------|
| **Add** | Add two columns |
| **Sub** | Subtract two columns |
| **Mul** | Multiply two columns |
| **Div** | Divide two columns |
| **Mod** | Modulo operation |
| **Power** | Power operation |
| **Greater** | Greater than comparison |
| **Less** | Less than comparison |
| **Equal** | Equality comparison |
| **And** | Logical AND |
| **Or** | Logical OR |

#### 4. Multi-Variable Operations (`easymlops.table.preprocessing.mulvar_operation`)

| Class | Description |
|-------|-------------|
| **AddMulti** | Add multiple columns |
| **MulMulti** | Multiply multiple columns |
| **ConcatStr** | Concatenate strings |
| **Coalesce** | Return first non-null value |

#### 5. Feature Encoding (`easymlops.table.encoding`)

| Class | Description |
|-------|-------------|
| **EncodingBase** | Base class for encoding |
| **LabelEncoding** | Label encoding for categorical features |
| **OneHotEncoding** | One-hot encoding |
| **TargetEncoding** | Target encoding with smoothing |
| **CountEncoding** | Count frequency encoding |
| **WOEEncoding** | Weight of Evidence encoding |
| **LGBMLeafEncoder** | LightGBM leaf node encoding |

#### 6. Classification Models (`easymlops.table.classification`)

| Class | Description |
|-------|-------------|
| **ClassificationBase** | Base class for classifiers |
| **LGBMClassification** | LightGBM classifier |
| **LogisticRegressionClassification** | Logistic regression |
| **SVMClassification** | Support Vector Machine |
| **DecisionTreeClassification** | Decision tree |
| **RandomForestClassification** | Random forest |
| **KNeighborsClassification** | K-Nearest Neighbors |
| **GaussianNBClassification** | Gaussian Naive Bayes |
| **MultinomialNBClassification** | Multinomial Naive Bayes |
| **BernoulliNBClassification** | Bernoulli Naive Bayes |

#### 7. Regression Models (`easymlops.table.regression`)

| Class | Description |
|-------|-------------|
| **RegressionBase** | Base class for regressors |
| **LGBMRegression** | LightGBM regressor |
| **LinearRegression** | Linear regression |
| **SVMRegression** | Support Vector Regression |
| **DecisionTreeRegression** | Decision tree regression |
| **RandomForestRegression** | Random forest regression |
| **KNeighborsRegression** | KNN regression |
| **RidgeRegression** | Ridge regression |
| **LassoRegression** | Lasso regression |
| **ElasticNetRegression** | Elastic Net regression |

#### 8. Factorization Machines (`easymlops.table.fm`)

| Class | Description |
|-------|-------------|
| **FMBase** | Base class for FM models |
| **FMClassification** | FM for classification |
| **FMRegression** | FM for regression |
| **FFMClassification** | Field-aware FM for classification |
| **FFMRegression** | Field-aware FM for regression |
| **DeepFMClassification** | DeepFM for classification |
| **DeepFMRegression** | DeepFM for regression |

#### 9. Ensemble Methods (`easymlops.table.ensemble`)

| Class | Description |
|-------|-------------|
| **Parallel** | Run multiple pipes in parallel |
| **Stacking** | Stacking ensemble |

#### 10. Dimensionality Reduction (`easymlops.table.decomposition`)

| Class | Description |
|-------|-------------|
| **PCA** | Principal Component Analysis |
| **SVD** | Singular Value Decomposition |
| **TCA** | Transfer Component Analysis |

#### 11. Feature Selection (`easymlops.table.feature_selection`)

| Class | Description |
|-------|-------------|
| **VarianceThreshold** | Remove low variance features |
| **SelectKBest** | Select top K features |
| **FeatureEmbedding** | Feature importance via embedding |

#### 12. Evaluation Metrics (`easymlops.table.eval`)

| Function | Description |
|----------|-------------|
| calc_precision_recall_at_thresholds | Calculate precision/recall at thresholds |
| calc_precision_recall_at_quantiles | Calculate precision/recall at quantiles |
| plot_pr_curve | Plot Precision-Recall curve |
| calc_roc_at_thresholds | Calculate ROC metrics at thresholds |
| plot_roc_curve | Plot ROC curve with AUC |

### NLP Pipeline

The NLP Pipeline handles natural language processing tasks.

#### 1. Preprocessing (`easymlops.nlp.preprocessing`)

| Class | Description |
|-------|-------------|
| **NLPPreprocessBase** | Base class for NLP preprocessing |
| **FixInput** | Ensure DataFrame input |
| **FillNa** | Fill missing text values |
| **ReplaceDigits** | Replace digits with token |
| **RemovePunctuation** | Remove punctuation |
| **RemoveStopwords** | Remove stopwords |
| **RemoveHtml** | Remove HTML tags |
| **LowerCase** | Convert to lowercase |
| **RemoveUrl** | Remove URLs |
| **RemoveEmail** | Remove email addresses |

#### 2. Representation (`easymlops.nlp.representation`)

| Class | Description |
|-------|-------------|
| **NLPRepresentationBase** | Base class for text representation |
| **TfidfModel** | TF-IDF vectorization |
| **Word2VecModel** | Word2Vec embeddings |
| **BertModel** | BERT embeddings |

#### 3. Text Classification (`easymlops.nlp.text_classification`)

| Class | Description |
|-------|-------------|
| **TextClassificationBase** | Base class for text classification |
| **TextCNN** | Text CNN classifier |
| **TextRNN** | Text RNN classifier |
| **HAN** | Hierarchical Attention Network |

#### 4. Text Regression (`easymlops.nlp.text_regression`)

| Class | Description |
|-------|-------------|
| **TextRegressionBase** | Base class for text regression |
| **TextCNNRegression** | Text CNN regressor |

#### 5. Similarity Search (`easymlops.nlp.similarity`)

| Class | Description |
|-------|-------------|
| **CosineSimilarity** | Cosine similarity search |
| **EuclideanSimilarity** | Euclidean distance search |

### Time Series Pipeline

The Time Series Pipeline handles time series forecasting.

#### Core Classes (`easymlops.ts.core`)

| Class | Description |
|-------|-------------|
| **TSPipeLine** | Time series pipeline |
| **TSPipeObjectBase** | Base class for TS pipes |

#### DNN Models (`easymlops.ts.dnn`)

| Class | Description |
|-------|-------------|
| **TSCNNRegression** | CNN for time series regression |

### YOLO Vision

The YOLO module provides computer vision capabilities.

#### Vision Tasks (`easymlops.yolo`)

| Class | Description |
|-------|-------------|
| **YOLODetection** | Object detection |
| **YOLOSegmentation** | Instance segmentation |
| **YOLOClassification** | Image classification |
| **YOLOPose** | Pose estimation |
| **YOLOOBB** | Oriented bounding box detection |

#### Training (`easymlops.yolo.train`)

| Class | Description |
|-------|-------------|
| **YOLOTrain** | YOLO model training |

### OCR Module

The OCR module provides optical character recognition.

#### OCR Classes (`easymlops.ocr`)

| Class | Description |
|-------|-------------|
| **EasyOCRText** | EasyOCR text recognition |
| **OCRPipeLine** | OCR pipeline |

### AutoML

The AutoML module provides automated machine learning capabilities.

#### Core Classes (`easymlops.automl`)

| Class | Description |
|-------|-------------|
| **AutoMLSession** | AutoML session manager |
| **AutoMLPipe** | AutoML pipeline |
| **AutoMLPipeV250429** | AutoML pipeline v2 |

#### LLM Integration (`easymlops.automl.llms`)

| Class | Description |
|-------|-------------|
| **LLMBase** | Base class for LLMs |
| **ChatGPT** | OpenAI ChatGPT integration |
| **DeepSeek** | DeepSeek integration |

#### Tools (`easymlops.automl.tools`)

| Class | Description |
|-------|-------------|
| **AutoMLTool** | Base class for AutoML tools |
| **CodeTool** | Code execution tool |
| **SearchTool** | Web search tool |

## Advanced Usage Examples

### Factorization Machines (FM/FFM/DeepFM)

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

### LightGBM Leaf Node Encoding

```python
from easymlops.table.encoding import LGBMLeafEncoder

encoder = LGBMLeafEncoder(y=label, n_estimators=10, max_depth=5)
pipeline = TablePipeLine().pipe(encoder)
pipeline.fit(df)
result = pipeline.transform(df)

# Get leaf node descriptions
descriptions = encoder.describe()
```

### Evaluation Metrics

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

## Core Classes Reference

### Pipeline Base Classes

| Class | Description |
|-------|-------------|
| **PipeObjectBase** | Base class for all pipe objects |
| **TablePipeObjectBase** | Base class for table pipes |
| **NLPPipeObjectBase** | Base class for NLP pipes |
| **TSPipeObjectBase** | Base class for time series pipes |

### Model Persistence

All models support save/load functionality:

```python
# Save model
pipeline.save("./model.pkl")

# Load model
pipeline = TablePipeLine.load("./model.pkl")
```

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
