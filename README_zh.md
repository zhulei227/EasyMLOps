# EasyMLOps

[![Python Version](https://img.shields.io/pypi/pyversions/easymlops)](https://pypi.org/project/easymlops/)
[![License](https://img.shields.io/pypi/l/easymlops)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/zhulei227/easymlops?style=social)](https://github.com/zhulei227/easymlops/stargazers)

[English](./README.md) | 中文版

EasyMLOps 是一个高效的机器学习运营框架，通过 Pipeline 方式构建建模任务，支持模型训练、预测、测试、特征存储、监控等功能。通过外套 Flask 或 FastApi 即可直接部署生产环境。

## 目录

- [概述](#概述)
- [特性](#特性)
- [架构](#架构)
- [安装](#安装)
- [快速开始](#快速开始)
- [模块详情](#模块详情)
  - [表格处理 Pipeline](#表格处理-pipeline)
  - [NLP Pipeline](#nlp-pipeline)
  - [时序 Pipeline](#时序-pipeline)
  - [YOLO 视觉](#yolo-视觉)
  - [OCR 模块](#ocr-模块)
  - [AutoML](#automl)
- [核心类参考](#核心类参考)
- [环境依赖](#环境依赖)
- [文档](#文档)
- [许可证](#许可证)

## 概述

EasyMLOps 提供统一的基于 Pipeline 的机器学习工作流架构。通过链式 API 设计，简化了构建、训练、评估和部署 ML 模型的过程。

### 核心概念

- **Pipe**: 处理和转换数据的独立单元
- **Pipeline**: 按顺序处理数据的管道链
- **Parallel**: 并行执行多个管道

## 特性

- 🚀 **统一的 Pipeline 架构** - 简洁的 API 设计，支持链式调用
- 📊 **表格数据处理** - 数据清洗、特征编码、特征降维、特征选择、因子分解机、分类回归
- 📝 **NLP 任务** - 文本清洗、分词、特征提取、文本分类、相似度检索
- 🖼️ **YOLO 视觉任务** - 目标检测、实例分割、图像分类、姿态估计、旋转目标检测
- 🔤 **OCR 文本识别** - 支持 80+ 语言
- 🤖 **AutoML** - 自动化机器学习
- 💾 **模型持久化** - 便捷的模型保存与加载

## 架构

```
easymlops/
├── core/           # 核心 Pipeline 基类
├── table/          # 表格数据处理
│   ├── preprocessing/      # 数据预处理
│   ├── encoding/           # 特征编码
│   ├── classification/     # 分类模型
│   ├── regression/         # 回归模型
│   ├── ensemble/           # 集成方法
│   ├── fm/                 # 因子分解机
│   ├── decomposition/      # 特征降维
│   ├── feature_selection/  # 特征选择
│   ├── strategy/           # 建模策略
│   ├── storage/            # 特征存储
│   ├── utils/              # 工具函数
│   ├── sqls/               # SQL 操作
│   └── perfopt/            # 性能优化
├── nlp/                   # 自然语言处理
│   ├── preprocessing/      # 文本清洗、分词
│   ├── representation/    # 文本向量化 (TFIDF, Word2Vec等)
│   ├── text_classification/ # 文本分类 (CNN, RNN, HAN)
│   ├── text_regression/   # 文本回归
│   └── similarity/         # 相似度检索 (Faiss, ES)
├── ts/                    # 时序数据处理
├── yolo/                  # YOLO 视觉任务
│   ├── detection/          # 目标检测
│   ├── segmentation/       # 实例分割
│   ├── classification/     # 图像分类
│   ├── pose/              # 姿态估计
│   └── obb/               # 旋转目标检测
├── ocr/                   # OCR 文本识别
│   └── easyocr/           # EasyOCR 集成
├── automl/                # AutoML
│   ├── llms/              # LLM 集成
│   ├── tools/              # 工具管理器
│   └── sessions/          # 会话管理器
└── storage/               # 存储后端
    └── feature_storage/    # 特征存储
```

## 安装

### 从 PyPI 安装

```bash
pip install easymlops
```

### 从源码安装

```bash
pip install -e .
```

## 快速开始

### 表格数据处理

```python
import pandas as pd
from easymlops import TablePipeLine
from easymlops.table.preprocessing import *
from easymlops.table.encoding import *
from easymlops.table.classification import *
from easymlops.table.ensemble import *

# 加载数据
data = pd.read_csv("./data/demo.csv")
x_train = data[:500]
x_test = data[500:]
y_train = x_train["Survived"]
del x_train["Survived"]
del x_test["Survived"]

# 构建 Pipeline
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

# 训练和预测
table.fit(x_train)
result = table.transform(x_test)
print(result.head(5))

# 单条预测
table.transform_single(x_test.to_dict("record")[0])
```

## 模块详情

### 表格处理 Pipeline

表格处理是核心模块，用于表格数据的处理和建模。

#### 1. 预处理 (`easymlops.table.preprocessing`)

| 类名 | 说明 |
|------|------|
| **PreprocessBase** | 所有预处理管道的基类 |
| **FixInput** | 确保输入为 DataFrame 格式 |
| **ReName** | 重命名列 |
| **DropCols** | 删除指定列 |
| **SelectCols** | 选择特定列 |
| **DoNoThing** | 无操作直通 |

#### 2. 单变量操作 (`easymlops.table.preprocessing.onevar_operation`)

| 类名 | 说明 |
|------|------|
| **Replace** | 替换列中的值 |
| **ClipString** | 裁剪字符串值 |
| **FillNa** | 填充缺失值 |
| **IsNull** | 标记空值 |
| **IsNotNull** | 标记非空值 |
| **TransToCategory** | 转换为类别类型 |
| **TransToFloat** | 转换为浮点类型 |
| **TransToInt** | 转换为整数类型 |
| **TransToLower** | 转换为小写 |
| **TransToUpper** | 转换为大写 |
| **Abs** | 绝对值 |
| **MapValues** | 使用字典映射值 |
| **Clip** | 裁剪值到范围 |
| **MinMaxScaler** | 最小最大归一化 |
| **Normalizer** | L1/L2 归一化 |
| **Bins** | 离散化连续值 |
| **Tanh** | 双曲正切激活函数 |
| **Relu** | 线性整流函数 |
| **Sigmoid** | Sigmoid 激活函数 |
| **Swish** | Swish 激活函数 |
| **DateMonthInfo** | 从日期提取月份 |
| **DateHourInfo** | 从日期提取小时 |
| **DateMinuteInfo** | 从日期提取分钟 |
| **DateTotalMinuteInfo** | 从日期提取总分钟数 |

#### 3. 双变量操作 (`easymlops.table.preprocessing.bivar_operation`)

| 类名 | 说明 |
|------|------|
| **Add** | 两列相加 |
| **Sub** | 两列相减 |
| **Mul** | 两列相乘 |
| **Div** | 两列相除 |
| **Mod** | 取模运算 |
| **Power** | 幂运算 |
| **Greater** | 大于比较 |
| **Less** | 小于比较 |
| **Equal** | 相等比较 |
| **And** | 逻辑与 |
| **Or** | 逻辑或 |

#### 4. 多变量操作 (`easymlops.table.preprocessing.mulvar_operation`)

| 类名 | 说明 |
|------|------|
| **AddMulti** | 多列相加 |
| **MulMulti** | 多列相乘 |
| **ConcatStr** | 字符串拼接 |
| **Coalesce** | 返回第一个非空值 |

#### 5. 特征编码 (`easymlops.table.encoding`)

| 类名 | 说明 |
|------|------|
| **EncodingBase** | 编码基类 |
| **LabelEncoding** | 标签编码 |
| **OneHotEncoding** | 独热编码 |
| **TargetEncoding** | 目标编码（带平滑） |
| **CountEncoding** | 计数频率编码 |
| **WOEEncoding** | 证据权重编码 |
| **LGBMLeafEncoder** | LightGBM 叶子节点编码 |

#### 6. 分类模型 (`easymlops.table.classification`)

| 类名 | 说明 |
|------|------|
| **ClassificationBase** | 分类器基类 |
| **LGBMClassification** | LightGBM 分类器 |
| **LogisticRegressionClassification** | 逻辑回归 |
| **SVMClassification** | 支持向量机 |
| **DecisionTreeClassification** | 决策树 |
| **RandomForestClassification** | 随机森林 |
| **KNeighborsClassification** | K近邻 |
| **GaussianNBClassification** | 高斯朴素贝叶斯 |
| **MultinomialNBClassification** | 多项式朴素贝叶斯 |
| **BernoulliNBClassification** | 伯努利朴素贝叶斯 |

#### 7. 回归模型 (`easymlops.table.regression`)

| 类名 | 说明 |
|------|------|
| **RegressionBase** | 回归器基类 |
| **LGBMRegression** | LightGBM 回归器 |
| **LinearRegression** | 线性回归 |
| **SVMRegression** | 支持向量回归 |
| **DecisionTreeRegression** | 决策树回归 |
| **RandomForestRegression** | 随机森林回归 |
| **KNeighborsRegression** | K近邻回归 |
| **RidgeRegression** | 岭回归 |
| **LassoRegression** | Lasso 回归 |
| **ElasticNetRegression** | 弹性网络回归 |

#### 8. 因子分解机 (`easymlops.table.fm`)

| 类名 | 说明 |
|------|------|
| **FMBase** | FM 模型基类 |
| **FMClassification** | FM 分类 |
| **FMRegression** | FM 回归 |
| **FFMClassification** | 域感知 FM 分类 |
| **FFMRegression** | 域感知 FM 回归 |
| **DeepFMClassification** | DeepFM 分类 |
| **DeepFMRegression** | DeepFM 回归 |

#### 9. 集成方法 (`easymlops.table.ensemble`)

| 类名 | 说明 |
|------|------|
| **Parallel** | 并行运行多个管道 |

#### 10. 特征降维 (`easymlops.table.decomposition`)

| 类名 | 说明 |
|------|------|
| **Decomposition** | 降维基类 |
| **PCADecomposition** | 主成分分析 |
| **NMFDecomposition** | 非负矩阵分解 |
| **KernelPCADecomposition** | 核PCA |
| **FastICADecomposition** | 快速独立成分分析 |
| **DictionaryLearningDecomposition** | 字典学习 |
| **MiniBatchDictionaryLearningDecomposition** | 迷你批字典学习 |
| **LDADecomposition** | 潜在狄利克雷分配 |
| **TSNEDecomposition** | t-SNE |
| **MDSDecomposition** | 多维缩放 |
| **IsomapDecomposition** | Isomap |
| **SpectralEmbeddingDecomposition** | 谱嵌入 |
| **LocallyLinearEmbeddingDecomposition** | 局部线性嵌入 |
| **TCADecomposition** | 迁移成分分析 |

#### 11. 特征选择 (`easymlops.table.feature_selection`)

| 类名 | 说明 |
|------|------|
| **FilterBase** | 过滤法特征选择基类 |
| **MissRateFilter** | 按缺失率过滤 |
| **VarianceFilter** | 按方差过滤 |
| **PersonCorrFilter** | 按皮尔逊相关系数过滤 |
| **Chi2Filter** | 按卡方检验过滤 |
| **PValueFilter** | 按p值过滤 |
| **MutualInfoFilter** | 按互信息过滤 |
| **IVFilter** | 按信息价值过滤 |
| **PSIFilter** | 按群体稳定性指数过滤 |
| **EmbedBase** | 嵌入法特征选择基类 |
| **LREmbed** | 基于逻辑回归的特征重要性 |
| **LGBMEmbed** | 基于LightGBM的特征重要性 |

#### 12. 评估指标 (`easymlops.table.utils`)

| 函数 | 说明 |
|------|------|
| calc_precision_recall_at_thresholds | 计算不同阈值下的精确率/召回率 |
| calc_precision_recall_at_quantiles | 计算不同分位数的精确率/召回率 |
| plot_pr_curve | 绘制精确率-召回率曲线 |
| calc_roc_at_thresholds | 计算不同阈值下的 ROC 指标 |
| plot_roc_curve | 绘制 ROC 曲线并计算 AUC |

#### 13. SQL 操作 (`easymlops.table.sqls`)

| 类名 | 说明 |
|------|------|
| **SQL** | 在 DataFrame 上执行 SQL 查询 |

#### 14. 性能优化 (`easymlops.table.perfopt`)

| 类名 | 说明 |
|------|------|
| **ReduceMemUsage** | 减少 DataFrame 内存占用 |
| **Dense2Sparse** | 将密集型 DataFrame 转换为稀疏格式 |

### NLP Pipeline

NLP Pipeline 用于处理自然语言处理任务。

#### 1. 预处理 (`easymlops.nlp.preprocessing`)

| 类名 | 说明 |
|------|------|
| **PreprocessBase** | NLP 预处理基类 |
| **Lower** | 转换为小写 |
| **Upper** | 转换为大写 |
| **RemoveDigits** | 移除数字 |
| **ReplaceDigits** | 替换数字为标记 |
| **RemovePunctuation** | 移除标点符号 |
| **ReplacePunctuation** | 替换标点符号 |
| **Replace** | 替换指定字符 |
| **RemoveWhitespace** | 移除空白字符 |
| **ExpandWhitespace** | 展开空白字符 |
| **RemoveStopWords** | 移除停用词 |
| **ExtractKeyWords** | 提取关键词 |
| **AppendKeyWords** | 追加关键词 |
| **ExtractChineseWords** | 提取中文词语 |
| **ExtractNGramWords** | 提取 n-gram 词 |
| **ExtractJieBaWords** | 使用 jieba 提取词语 |
| **VocabIndex** | 将词语转换为词汇索引 |
| **ExtractJieBaWordsWithSentSplit** | 分句后提取词语 |
| **VocabIndexWithSentSplit** | 分句后转换为词汇索引 |

#### 2. 表示 (`easymlops.nlp.representation`)

| 类名 | 说明 |
|------|------|
| **RepresentationBase** | 文本表示基类 |
| **BagOfWords** | 词袋模型 |
| **TFIDF** | TF-IDF 向量化 |
| **LdaTopicModel** | LDA 主题模型 |
| **LsiTopicModel** | LSI 主题模型 |
| **Word2VecModel** | Word2Vec 词嵌入 |
| **Doc2VecModel** | Doc2Vec 词嵌入 |
| **FastTextModel** | FastText 词嵌入 |

#### 3. 文本分类 (`easymlops.nlp.text_classification`)

| 类名 | 说明 |
|------|------|
| **TextClassificationBase** | 文本分类基类 |
| **TextCNNClassification** | TextCNN 分类器 |
| **TextRNNClassification** | TextRNN 分类器 |
| **HANClassification** | 层次注意力网络 |

#### 4. 文本回归 (`easymlops.nlp.text_regression`)

| 类名 | 说明 |
|------|------|
| **TextRegressionBase** | 文本回归基类 |
| **TextCNNRegression** | TextCNN 回归器 |

#### 5. 相似度检索 (`easymlops.nlp.similarity`)

| 类名 | 说明 |
|------|------|
| **ElasticSearchSimilarity** | ElasticSearch 相似度检索 |
| **FaissSimilarity** | Faiss 相似度检索 |

### 时序 Pipeline

时序 Pipeline 用于处理时间序列预测任务。

#### 核心类 (`easymlops.ts.core`)

| 类名 | 说明 |
|------|------|
| **TSPipeLine** | 时序 Pipeline |
| **TSPipeObjectBase** | 时序管道基类 |

#### DNN 模型 (`easymlops.ts.dnn`)

| 类名 | 说明 |
|------|------|
| **TSCNNRegression** | 时序 CNN 回归 |

### YOLO 视觉

YOLO 模块提供计算机视觉能力。

#### 视觉任务 (`easymlops.yolo`)

| 类名 | 说明 |
|------|------|
| **YOLODetection** | 目标检测 |
| **YOLOSegmentation** | 实例分割 |
| **YOLOClassification** | 图像分类 |
| **YOLOPose** | 姿态估计 |
| **YOLOOBB** | 旋转目标检测 |

#### 训练 (`easymlops.yolo.train`)

| 类名 | 说明 |
|------|------|
| **YOLOTrain** | YOLO 模型训练 |

### OCR 模块

OCR 模块提供光学字符识别功能。

#### OCR 类 (`easymlops.ocr`)

| 类名 | 说明 |
|------|------|
| **EasyOCRText** | EasyOCR 文本识别 |
| **OCRPipeLine** | OCR Pipeline |

### AutoML

AutoML 模块提供自动化机器学习能力。

#### 核心类 (`easymlops.automl`)

| 类名 | 说明 |
|------|------|
| **AutoMLTab** | AutoML 表格数据处理 |
| **AutoML** | AutoML 主类 |

#### LLM 集成 (`easymlops.automl.llms`)

| 类名 | 说明 |
|------|------|
| **LLM** | LLM 基类 |
| **OllamaLLM** | Ollama LLM 集成 |
| **SparkLLM** | Spark LLM 集成 |
| **ZhiPuLLM** | 智谱 LLM 集成 |
| **KimiLLM** | Kimi LLM 集成 |

#### 工具 (`easymlops.automl.tools`)

| 类名 | 说明 |
|------|------|
| **LLMToolManager** | LLM 工具管理器 |

#### 会话 (`easymlops.automl.sessions`)

| 类名 | 说明 |
|------|------|
| **LLMSessionManager** | LLM 会话管理器 |

## 高级用法示例

### 因子分解机 (FM/FFM/DeepFM)

```python
from easymlops.table.fm import FMClassification, DeepFMClassification

# FM 分类
fm = FMClassification(
    y=label,
    cols=["feature1", "feature2", "feature3"],
    field_dims=[5, 10, 8],
    embed_dim=8,
    epochs=10
)
pipeline = TablePipeLine().pipe(fm)
pipeline.fit(df).transform(df)

# DeepFM 分类
deepfm = DeepFMClassification(
    y=label,
    cols=["feature1", "feature2", "feature3"],
    field_dims=[5, 10, 8],
    embed_dim=8,
    hidden_layers=[64, 32],
    epochs=10
)
```

### LightGBM 叶子节点编码

```python
from easymlops.table.encoding import LGBMLeafEncoder

encoder = LGBMLeafEncoder(y=label, n_estimators=10, max_depth=5)
pipeline = TablePipeLine().pipe(encoder)
pipeline.fit(df)
result = pipeline.transform(df)

# 获取叶子节点描述
descriptions = encoder.describe()
```

### 评估指标

```python
from easymlops.table.utils import (
    calc_precision_recall_at_thresholds,
    plot_pr_curve,
    calc_roc_at_thresholds,
    plot_roc_curve
)

# P-R 曲线
result = calc_precision_recall_at_thresholds(y_true, y_pred, bins=10)
plot_pr_curve(y_true, y_preds_dict, save_path="pr_curve.png")

# ROC 曲线
result = calc_roc_at_thresholds(y_true, y_pred, bins=10)
plot_roc_curve(y_true, y_preds_dict, save_path="roc_curve.png")
```

### NLP 文本处理

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

### YOLO 视觉任务

```python
from easymlops.yolo.detection import YOLODetection
from easymlops.yolo.segmentation import YOLOSegmentation
from easymlops.yolo.classification import YOLOClassification
from easymlops.yolo.pose import YOLOPose
from easymlops.yolo.obb import YOLOOBB

# 目标检测
pipe = YOLODetection(model_name="yolo11n", device="cpu")

# 实例分割
pipe = YOLOSegmentation(model_name="yolo11n-seg")

# 图像分类
pipe = YOLOClassification(model_name="yolo11n-cls", imgsz=224)

# 姿态估计
pipe = YOLOPose(model_name="yolo11n-pose")

# 旋转目标检测
pipe = YOLOOBB(model_name="yolo11n-obb")

# 训练和预测
df = pd.DataFrame({"image_path": ["path/to/image.jpg"]})
result = pipe.fit(df).transform(df)
```

### OCR 文本识别

```python
from easymlops.ocr import EasyOCRText

df = pd.DataFrame({"image_path": ["path/to/image.jpg"]})
pipe = EasyOCRText(lang="en", gpu=False)
result = pipe.fit(df).transform(df)
```

## 核心类参考

### Pipeline 基类

| 类名 | 说明 |
|------|------|
| **PipeObjectBase** | 所有管道对象的基类 |
| **TablePipeObjectBase** | 表格管道的基类 |
| **NLPPipeObjectBase** | NLP 管道的基类 |
| **TSPipeObjectBase** | 时序管道的基类 |

### 模型持久化

所有模型都支持保存/加载功能：

```python
# 保存模型
pipeline.save("./model.pkl")

# 加载模型
pipeline = TablePipeLine.load("./model.pkl")
```

## 环境依赖

### YOLO 视觉任务

```bash
pip install ultralytics-opencv-headless
# 或
pip install numpy>=1.24.4,<2 torch>=1.12.0 torchvision>=0.13.0 ultralytics
```

### OCR 文本识别

```bash
pip install easyocr
pip install "numpy<2" "Pillow<10" "torch>=1.12.0" "torchvision>=0.13.0"
```

## 文档

详细使用说明请参考 [skills](./skills) 目录下的文档：

- [SKILL.md](./skills/SKILL.md) - 主文档
- [01_table.md](./skills/docs/01_table.md) - Table 数据处理
- [02_nlp.md](./skills/docs/02_nlp.md) - NLP 任务
- [03_timeseries.md](./skills/docs/03_timeseries.md) - 时序任务
- [04_pipeline.md](./skills/docs/04_pipeline.md) - Pipeline 操作
- [05_persistence.md](./skills/docs/05_persistence.md) - 模型持久化
- [06_deploy.md](./skills/docs/06_deploy.md) - 生产部署
- [07_storage.md](./skills/docs/07_storage.md) - 特征存储
- [08_automl.md](./skills/docs/08_automl.md) - AutoML
- [09_cheatsheet.md](./skills/docs/09_cheatsheet.md) - 常用类速查表
- [10_example.md](./skills/docs/10_example.md) - 完整示例
- [11_yolo.md](./skills/docs/11_yolo.md) - YOLO 视觉任务
- [12_ocr.md](./skills/docs/12_ocr.md) - OCR 文本识别

## 许可证

本项目基于 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件
