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
│   ├── preprocessing/      # Data preprocessing
│   ├── encoding/           # Feature encoding
│   ├── classification/     # Classification models
│   ├── regression/         # Regression models
│   ├── ensemble/           # Ensemble methods
│   ├── fm/                 # Factorization Machines
│   ├── decomposition/      # Dimensionality reduction
│   ├── feature_selection/  # Feature selection
│   ├── strategy/           # Modeling strategies
│   ├── storage/            # Feature storage
│   ├── utils/              # Utility functions
│   ├── sqls/               # SQL operations
│   └── perfopt/            # Performance optimization
├── nlp/                   # Natural language processing
│   ├── preprocessing/      # Text cleaning, tokenization
│   ├── representation/     # Text embedding (TFIDF, Word2Vec, etc.)
│   ├── text_classification/ # Text classification (CNN, RNN, HAN)
│   ├── text_regression/    # Text regression
│   └── similarity/         # Similarity search (Faiss, ES)
├── ts/                    # Time series processing
│   ├── core/              # Core pipeline base classes
│   ├── statistical/       # Statistical models (ARIMA, SARIMA, GARCH)
│   ├── deep_learning/     # Deep learning models (N-BEATS, N-HiTS, DeepAR, GP)
│   ├── transformer/       # Transformer models (TFT, Informer, Autoformer, etc.)
│   ├── state_space/       # State space models (DeepState, Mamba, Liquid S4)
│   ├── generative/       # Generative models (VAE, Normalizing Flow, Diffusion)
│   └── dnn/               # DNN models (CNN for time series)
├── yolo/                  # YOLO vision tasks
│   ├── detection/          # Object detection
│   ├── segmentation/       # Instance segmentation
│   ├── classification/     # Image classification
│   ├── pose/               # Pose estimation
│   └── obb/                # Oriented bounding box
├── ocr/                   # OCR text recognition
│   └── easyocr/            # EasyOCR integration
├── automl/                # AutoML
│   ├── llms/               # LLM integrations
│   ├── tools/              # Tool managers
│   └── sessions/           # Session managers
└── storage/               # Storage backends
    └── feature_storage/    # Feature store
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

#### 10. Dimensionality Reduction (`easymlops.table.decomposition`)

| Class | Description |
|-------|-------------|
| **Decomposition** | Base class for dimensionality reduction |
| **PCADecomposition** | Principal Component Analysis |
| **NMFDecomposition** | Non-negative Matrix Factorization |
| **KernelPCADecomposition** | Kernel PCA |
| **FastICADecomposition** | Fast Independent Component Analysis |
| **DictionaryLearningDecomposition** | Dictionary Learning |
| **MiniBatchDictionaryLearningDecomposition** | Mini-batch Dictionary Learning |
| **LDADecomposition** | Latent Dirichlet Allocation |
| **TSNEDecomposition** | t-SNE |
| **MDSDecomposition** | Multidimensional Scaling |
| **IsomapDecomposition** | Isomap |
| **SpectralEmbeddingDecomposition** | Spectral Embedding |
| **LocallyLinearEmbeddingDecomposition** | Locally Linear Embedding |
| **TCADecomposition** | Transfer Component Analysis |

#### 11. Feature Selection (`easymlops.table.feature_selection`)

| Class | Description |
|-------|-------------|
| **FilterBase** | Base class for filter-based feature selection |
| **MissRateFilter** | Filter by missing rate |
| **VarianceFilter** | Filter by variance |
| **PersonCorrFilter** | Filter by Pearson correlation |
| **Chi2Filter** | Filter by chi-square test |
| **PValueFilter** | Filter by p-value |
| **MutualInfoFilter** | Filter by mutual information |
| **IVFilter** | Filter by Information Value |
| **PSIFilter** | Filter by Population Stability Index |
| **EmbedBase** | Base class for embedding-based feature selection |
| **LREmbed** | Feature importance via Logistic Regression |
| **LGBMEmbed** | Feature importance via LightGBM |

#### 12. Evaluation Metrics (`easymlops.table.utils`)

| Function | Description |
|----------|-------------|
| calc_precision_recall_at_thresholds | Calculate precision/recall at thresholds |
| calc_precision_recall_at_quantiles | Calculate precision/recall at quantiles |
| plot_pr_curve | Plot Precision-Recall curve |
| calc_roc_at_thresholds | Calculate ROC metrics at thresholds |
| plot_roc_curve | Plot ROC curve with AUC |

#### 13. SQL Operations (`easymlops.table.sqls`)

| Class | Description |
|-------|-------------|
| **SQL** | Execute SQL queries on DataFrame |

#### 14. Performance Optimization (`easymlops.table.perfopt`)

| Class | Description |
|-------|-------------|
| **ReduceMemUsage** | Reduce memory usage of DataFrame |
| **Dense2Sparse** | Convert dense DataFrame to sparse format |

### NLP Pipeline

The NLP Pipeline handles natural language processing tasks.

#### 1. Preprocessing (`easymlops.nlp.preprocessing`)

| Class | Description |
|-------|-------------|
| **PreprocessBase** | Base class for NLP preprocessing |
| **Lower** | Convert to lowercase |
| **Upper** | Convert to uppercase |
| **RemoveDigits** | Remove digits |
| **ReplaceDigits** | Replace digits with token |
| **RemovePunctuation** | Remove punctuation |
| **ReplacePunctuation** | Replace punctuation |
| **Replace** | Replace specified characters |
| **RemoveWhitespace** | Remove whitespace |
| **ExpandWhitespace** | Expand whitespace |
| **RemoveStopWords** | Remove stopwords |
| **ExtractKeyWords** | Extract keywords |
| **AppendKeyWords** | Append keywords |
| **ExtractChineseWords** | Extract Chinese words |
| **ExtractNGramWords** | Extract n-gram words |
| **ExtractJieBaWords** | Extract words using jieba |
| **VocabIndex** | Convert words to vocabulary indices |
| **ExtractJieBaWordsWithSentSplit** | Extract words with sentence splitting |
| **VocabIndexWithSentSplit** | Convert words to indices with sentence splitting |

#### 2. Representation (`easymlops.nlp.representation`)

| Class | Description |
|-------|-------------|
| **RepresentationBase** | Base class for text representation |
| **BagOfWords** | Bag of Words model |
| **TFIDF** | TF-IDF vectorization |
| **LdaTopicModel** | LDA Topic Model |
| **LsiTopicModel** | LSI Topic Model |
| **Word2VecModel** | Word2Vec embeddings |
| **Doc2VecModel** | Doc2Vec embeddings |
| **FastTextModel** | FastText embeddings |

#### 3. Text Classification (`easymlops.nlp.text_classification`)

| Class | Description |
|-------|-------------|
| **TextClassificationBase** | Base class for text classification |
| **TextCNNClassification** | Text CNN classifier |
| **TextRNNClassification** | Text RNN classifier |
| **HANClassification** | Hierarchical Attention Network |

#### 4. Text Regression (`easymlops.nlp.text_regression`)

| Class | Description |
|-------|-------------|
| **TextRegressionBase** | Base class for text regression |
| **TextCNNRegression** | Text CNN regressor |

#### 5. Similarity Search (`easymlops.nlp.similarity`)

| Class | Description |
|-------|-------------|
| **ElasticSearchSimilarity** | ElasticSearch similarity search |
| **FaissSimilarity** | Faiss similarity search |

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

#### Statistical Models (`easymlops.ts.statistical`)

| Class | Description |
|-------|-------------|
| **ArimaRegression** | ARIMA time series forecasting |
| **SarimaRegression** | SARIMA seasonal time series forecasting |
| **ArimaxRegression** | ARIMAX time series with exogenous variables |
| **SarimaxRegression** | SARIMAX seasonal time series with exogenous variables |
| **GarchRegression** | GARCH volatility model |

#### Deep Learning Models (`easymlops.ts.deep_learning`)

| Class | Description |
|-------|-------------|
| **NBeatsRegression** | N-BEATS time series forecasting |
| **NHiTSRegression** | N-HiTS time series forecasting |
| **DeepARRegression** | DeepAR time series forecasting |
| **GPRegression** | Gaussian Process time series forecasting |

#### Transformer Models (`easymlops.ts.transformer`)

| Class | Description |
|-------|-------------|
| **TFTRegression** | Temporal Fusion Transformer |
| **InformerRegression** | Informer time series forecasting |
| **AutoformerRegression** | Autoformer time series forecasting |
| **FEDformerRegression** | FEDformer time series forecasting |
| **PatchTSTRegression** | PatchTST time series forecasting |
| **TimesNetRegression** | TimesNet time series forecasting |
| **iTransformerRegression** | iTransformer time series forecasting |

#### State Space Models (`easymlops.ts.state_space`)

| Class | Description |
|-------|-------------|
| **DeepStateRegression** | Deep State time series forecasting |
| **MambaRegression** | Mamba state space time series forecasting |
| **LiquidS4Regression** | Liquid S4 state space time series forecasting |

#### Generative Models (`easymlops.ts.generative`)

| Class | Description |
|-------|-------------|
| **VAERegression** | VAE variational autoencoder time series |
| **NormalizingFlowRegression** | Normalizing Flow time series |
| **DiffusionRegression** | Diffusion time series forecasting |

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
| **AutoMLTab** | AutoML tabular data processing |
| **AutoML** | Main AutoML class |

#### LLM Integration (`easymlops.automl.llms`)

| Class | Description |
|-------|-------------|
| **LLM** | Base class for LLMs |
| **OllamaLLM** | Ollama LLM integration |
| **SparkLLM** | Spark LLM integration |
| **ZhiPuLLM** | ZhiPu LLM integration |
| **KimiLLM** | Kimi LLM integration |

#### Tools (`easymlops.automl.tools`)

| Class | Description |
|-------|-------------|
| **LLMToolManager** | LLM tool manager |

#### Sessions (`easymlops.automl.sessions`)

| Class | Description |
|-------|-------------|
| **LLMSessionManager** | LLM session manager |

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
