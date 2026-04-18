# Table 数据处理 (TablePipeLine)

## 1.1 数据预处理 (preprocessing)

### 基础操作

```python
from easymlops.table.preprocessing import *

# 创建 Pipeline
table = TablePipeLine()

# 固定输入列顺序和数据类型（推荐作为第一个pipe）
table.pipe(FixInput())

# 类型转换
table.pipe(TransToCategory(cols=["Cabin", "Embarked"]))  # 转换为类别
table.pipe(TransToFloat(cols=["Age", "Fare"]))           # 转换为浮点
table.pipe(TransToInt(cols=["Pclass"]))                  # 转换为整数

# 大小写转换
table.pipe(TransToLower(cols=["Name", "Sex"]))           # 转小写
table.pipe(TransToUpper(cols=["Ticket"]))                # 转大写

# 列操作
table.pipe(ReName(cols=[("PassengerId", "乘客ID")]))     # 重命名
table.pipe(SelectCols(cols=["Name", "Age", "Sex"]))      # 选择列
table.pipe(DropCols(cols=["Cabin", "Ticket"]))           # 删除列

# 执行 pipeline
x_train_processed = table.fit(x_train).transform(x_test)
```

### 空值处理 (FillNa)

```python
from easymlops.table.preprocessing import FillNa

# 方式1: 均值填充
table.pipe(FillNa(cols=["Age"], strategy="mean"))

# 方式2: 中位数填充
table.pipe(FillNa(cols=["Age"], strategy="median"))

# 方式3: 众数填充
table.pipe(FillNa(cols=["Embarked"], strategy="mode"))

# 方式4: 指定值填充
table.pipe(FillNa(cols=["Age"], strategy="constant", fill_value=0))
```

### 一元操作

```python
from easymlops.table.preprocessing import Replace, Clip, MinMaxScaler, Normalizer, Bins

# 替换值
table.pipe(Replace(cols=["Sex"], old="male", new="M"))

# 数值裁剪
table.pipe(Clip(cols=["Age"], lower=0, upper=100))

# 归一化
table.pipe(MinMaxScaler(cols=["Age", "Fare"]))

# 标准化
table.pipe(Normalizer(cols=["Age", "Fare"]))

# 分箱
table.pipe(Bins(cols=["Age"], n_bins=5, strategy="equal_width"))
```

### 二元操作

```python
from easymlops.table.preprocessing import Add, Subtract, Multiply, Divide

# 两列运算
table.pipe(Add(cols=["col1", "col2"], new_col="sum"))
table.pipe(Subtract(cols=["col1", "col2"], new_col="diff"))
table.pipe(Multiply(cols=["col1", "col2"], new_col="product"))
table.pipe(Divide(cols=["col1", "col2"], new_col="ratio"))

# 日期操作
table.pipe(DateDayDiff(cols=["date1", "date2"], new_col="days_diff"))
```

### 更多预处理操作

```python
from easymlops.table.preprocessing import DoNoThing, ClipString, IsNull, IsNotNull
from easymlops.table.preprocessing import Abs, MapValues
from easymlops.table.preprocessing import Tanh, Relu, Sigmoid, Swish
from easymlops.table.preprocessing import DateMonthInfo, DateHourInfo, DateMinuteInfo, DateTotalMinuteInfo
from easymlops.table.preprocessing import CrossCategoryWithNumber, CrossNumberWithNumber
from easymlops.table.preprocessing import Sum, Mean, Median

# 空操作
table.pipe(DoNoThing())

# 字符串裁剪
table.pipe(ClipString(cols=["Name"], start=0, end=10))

# 空值判断
table.pipe(IsNull(cols=["Age"], new_col="is_age_null"))
table.pipe(IsNotNull(cols=["Age"], new_col="is_age_not_null"))

# 绝对值
table.pipe(Abs(cols=["diff"]))

# 值映射
table.pipe(MapValues(cols=["Sex"], mapping={"male": 0, "female": 1}))

# 激活函数
table.pipe(Tanh(cols=["feature"]))
table.pipe(Relu(cols=["feature"]))
table.pipe(Sigmoid(cols=["feature"]))
table.pipe(Swish(cols=["feature"]))

# 日期特征提取
table.pipe(DateMonthInfo(cols=["date"], new_col="month"))
table.pipe(DateHourInfo(cols=["date"], new_col="hour"))
table.pipe(DateMinuteInfo(cols=["date"], new_col="minute"))
table.pipe(DateTotalMinuteInfo(cols=["date"], new_col="total_minute"))

# 类别与数值交叉
table.pipe(CrossCategoryWithNumber(cols=["City", "Price"], new_col="city_price"))

# 数值与数值交叉
table.pipe(CrossNumberWithNumber(cols=["Price", "Quantity"], new_col="price_qty"))

# 多列汇总操作
table.pipe(Sum(cols=["col1", "col2", "col3"], new_col="total"))
table.pipe(Mean(cols=["col1", "col2", "col3"], new_col="avg"))
table.pipe(Median(cols=["col1", "col2", "col3"], new_col="mid"))
```

### 自定义Pipe模块

```python
from easymlops.table.preprocessing.extract_lastname import ExtractLastName

# 提取姓氏（自定义Pipe模块）
table.pipe(ExtractLastName(name_col="Name", new_col="LastName"))
```

### LightGBM叶子节点编码

```python
from easymlops.table.encoding import LGBMLeafEncoder
from easymlops.table.core.pipeline import TablePipeLine

# 使用LightGBM决策树进行叶子节点特征编码
# 输出为稀疏向量：每个叶子节点对应一个维度，命中为1，未命中为0
encoder = LGBMLeafEncoder(
    y=label,           # 标签数据，用于训练决策树
    n_estimators=10,  # 决策树数量
    max_depth=5,       # 决策树最大深度
    num_leaves=31,     # 每棵树的叶子节点数量
    learning_rate=0.1  # 学习率
)

pipeline = TablePipeLine()
pipeline.pipe(encoder)
pipeline.fit(df)

# 获取叶子节点描述信息
descriptions = encoder.describe()
# 输出示例：
# Leaf 0: feature1<=-0.938 and feature2<=10.5
# Leaf 1: feature1>-0.938
# Leaf 2: feature1<=-0.938 and feature2>10.5 and feature2<=17.5
# ...

# 转换结果
result = pipeline.transform(df)
# 输出形状: (n_samples, n_leaves)
```

**参数说明：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| y | array-like | None | 标签数据，用于训练决策树 |
| n_estimators | int | 10 | 决策树数量 |
| max_depth | int | 5 | 决策树最大深度 |
| learning_rate | float | 0.1 | 学习率 |
| num_leaves | int | 31 | 每棵树的叶子节点数量 |
| min_child_samples | int | 20 | 叶子节点最小样本数 |
| cols | str/list | "all" | 要操作的列 |

**describe() 方法：**
- 返回字典，key为叶子节点索引，value为该叶子节点对应的特征组合条件描述
- 描述格式：
  - 数值特征：`feature_name<=5 and feature_name>6`
  - 离散特征（类别<=20）：`feature_name in (1,2,3)` 或 `feature_name not in (4,5)`

**保存与加载：**
- 通过 TablePipeLine 的 save/load 方法进行模型持久化
- 加载后可直接使用，无需重新 fit

---

## 1.2 特征编码 (encoding)

```python
from easymlops.table.encoding import LabelEncoding, OneHotEncoding, TargetEncoding, WOEEncoding

# 标签编码
table.pipe(LabelEncoding(cols=["Sex", "Embarked"]))

# 独热编码
table.pipe(OneHotEncoding(cols=["Sex", "Embarked"]))

# 目标编码 (需要指定 label 参数)
table.pipe(TargetEncoding(cols=["Sex"], label="Survived"))

# WOE编码
table.pipe(WOEEncoding(cols=["Sex"], label="Survived"))
```

---

## 1.3 特征降维 (decomposition)

```python
from easymlops.table.decomposition import PCADecomposition, NMFDecomposition, LDADecomposition

# PCA降维
table.pipe(PCADecomposition(n_components=10))

# NMF非负矩阵分解
table.pipe(NMFDecomposition(n_components=10))

# LDA主题模型
table.pipe(LDADecomposition(n_components=10))
```

### 更多降维方法

```python
from easymlops.table.decomposition import *

# Kernel PCA
table.pipe(KernelPCADecomposition(n_components=10))

# FastICA
table.pipe(FastICADecomposition(n_components=10))

# Dictionary Learning
table.pipe(DictionaryLearningDecomposition(n_components=10))

# t-SNE
table.pipe(TSNEDecomposition(n_components=2))

# MDS
table.pipe(MDSDecomposition(n_components=2))

# Isomap
table.pipe(IsomapDecomposition(n_components=2))

# Spectral Embedding
table.pipe(SpectralEmbeddingDecomposition(n_components=2))

# Locally Linear Embedding
table.pipe(LocallyLinearEmbeddingDecomposition(n_components=2))
```

---

## 1.4 特征选择 (feature_selection)

### 过滤式

```python
from easymlops.table.feature_selection import MissRateFilter, VarianceFilter, PersonCorrFilter
from easymlops.table.feature_selection import Chi2Filter, IVFilter, PSIFilter

# 缺失率过滤
table.pipe(MissRateFilter(threshold=0.5))

# 方差过滤
table.pipe(VarianceFilter(threshold=0.0))

# Pearson相关系数过滤
table.pipe(PersonCorrFilter(threshold=0.8))

# 卡方检验
table.pipe(Chi2Filter(k=10))

# P值过滤
table.pipe(PValueFilter(k=10))

# 互信息过滤
table.pipe(MutualInfoFilter(k=10))

# IV值过滤
table.pipe(IVFilter(threshold=0.02))

# PSI稳定性过滤
table.pipe(PSIFilter(threshold=0.1))
```

### 嵌入式

```python
from easymlops.table.feature_selection import LREmbed, LGBMEmbed

# 逻辑回归嵌入式特征选择
table.pipe(LREmbed(label="Survived", C=1.0))

# LightGBM嵌入式特征选择
table.pipe(LGBMEmbed(label="Survived", n_estimators=100))
```

### TCA 迁移学习

```python
from easymlops.table.decomposition import TCADecomposition

# TCA (Transfer Component Analysis) 迁移学习降维
table.pipe(TCADecomposition(
    n_components=10,
    kernel_type="linear",
    mu=1.0
))
```

### 扩展模块 (extend)

```python
from easymlops.table.extend import Normalization, MapValues as ExtendMapValues

# 高级归一化
table.pipe(Normalization(cols=["feature1", "feature2"]))

# 扩展的值映射
table.pipe(ExtendMapValues(cols=["col"], mapping={...}))
```

### 性能优化 (perfopt)

```python
from easymlops.table.perfopt import ReduceMemUsage, Dense2Sparse

# 内存优化：通过修改数据类型减少内存使用量
# 自动检测并转换整数和浮点数的最小数据类型
table.pipe(ReduceMemUsage())

# 稠密转稀疏：将稠密矩阵转换为稀疏矩阵
# 适用于0值较多的数据，可显著减少内存使用
table.pipe(Dense2Sparse())
```

### 评估 (eval)

```python
from easymlops.table.eval import Eval

# 使用SQL表达式对数据进行计算评估
# 格式："(a+b)/c as col1, c//d as col2"
table.pipe(Eval(sql="(a+b)/c as col1, c//d as col2"))

# 模型评估
table.pipe(Eval(
    label="Survived",
    metrics=["accuracy", "precision", "recall", "f1"]
))
```

### 嵌入式

```python
from easymlops.table.feature_selection import LGBMFeatureSelection
# 使用LGBM进行特征重要性选择
```

### 工具模块 (utils)

```python
from easymlops.table.utils import EvalFunction, FasterLgbMulticlassPredictor
from easymlops.table.utils import FasterLgbPredictorSingle
from easymlops.table.utils import CpuMemDetector
from easymlops.table.utils import PandasUtils
from easymlops.table.utils import calc_precision_recall_at_thresholds, calc_precision_recall_at_quantiles, plot_pr_curve
from easymlops.table.utils import calc_roc_at_thresholds, plot_roc_curve

# 评估函数
eval_func = EvalFunction(
    y_true,
    y_pred,
    metric="accuracy"
)

# 阈值评估：计算不同阈值下的precision、recall、F1
result = calc_precision_recall_at_thresholds(y_true, y_pred, bins=10)
# 输出：
#    threshold  cumulative_samples  tp  fp  fn  tn  precision  recall       f1
# 0       0.95                   1   1   0   7   7  1.000000   0.125 0.222222
# 1       0.81                   3   3   0   5   7  1.000000   0.375 0.545455
# ...

# 基于分位数的阈值评估
result2 = calc_precision_recall_at_quantiles(y_true, y_pred, quantiles=10)

# 绘制多模型P-R曲线
y_preds_dict = {
    "Model_A": y_pred_a,
    "Model_B": y_pred_b,
    "Model_C": y_pred_c
}
best_scores = plot_pr_curve(y_true, y_preds_dict, bins=10, save_path="pr_curve.png")
# 输出最佳F1分数和对应阈值：
# {'Model_A': {'best_f1': 0.85, 'best_threshold': 0.5, ...}, ...}

# 计算ROC指标
result_roc = calc_roc_at_thresholds(y_true, y_pred, bins=10)
# 输出 DataFrame: threshold, fpr, tpr, tnr, fnr

# 绘制多模型ROC曲线
y_preds_dict_roc = {
    "Model_A": y_pred_a,
    "Model_B": y_pred_b,
    "Model_C": y_pred_c
}
aucs = plot_roc_curve(y_true, y_preds_dict_roc, bins=10, save_path="roc_curve.png")
# 输出每个模型的AUC值：
# {'Model_A': 0.92, 'Model_B': 0.85, 'Model_C': 0.78}

# 快速LGB单类预测器
predictor = FasterLgbPredictorSingle(
    model_path="model.lgb",
    feature_names=[...]
)

# CPU/内存检测
detector = CpuMemDetector()
cpu_count = detector.get_cpu_count()
mem_info = detector.get_mem_info()

# Pandas工具
df_cleaned = PandasUtils.drop_duplicates(df)
```

### 回调模块 (callback)

```python
from easymlops.table.callback import *

# 批量单条转换
run_batch_single_transform(module, data)

# 形状检查
check_shape(module, batch_transform, single_transform)

# 列检查
check_columns(module, batch_transform, single_transform)

# 数据类型检查
check_data_type(module, batch_transform, single_transform)

# 数据一致性检查
check_data_same(module, batch_transform, single_transform)

# 转换函数检查
check_transform_function(module, data)

# 泄漏检查
leak_check_type_is_same(module, type1, type2)
leak_check_value_is_same(module, ser1, ser2)

# Pipeline检查
check_transform_function_pipeline(pipeline, x, sample=1000)

# 批量转换一致性检查
check_two_batch_transform_same(pipeline, cur_batch, pre_batch)

# 空值检查
check_null_value(pipeline, x, sample=100)

# 极端值检查
check_extreme_value(pipeline, x, sample=100)

# 数据类型反转检查
check_inverse_dtype(pipeline, x, sample=100)

# int转float检查
check_int_trans_float(pipeline, x, sample=100)
```

---

## 1.5 分类模型 (classification)

```python
from easymlops.table.classification import *

# LightGBM分类
table.pipe(LGBMClassification(
    label="Survived",
    n_estimators=100,
    learning_rate=0.1,
    num_leaves=31
))

# 逻辑回归
table.pipe(LogisticRegressionClassification(label="Survived"))

# SVM分类
table.pipe(SVMClassification(label="Survived"))

# 决策树
table.pipe(DecisionTreeClassification(label="Survived"))

# 随机森林
table.pipe(RandomForestClassification(label="Survived", n_estimators=100))

# K近邻
table.pipe(KNeighborsClassification(label="Survived", n_neighbors=5))

# 朴素贝叶斯
table.pipe(GaussianNBClassification(label="Survived"))
table.pipe(MultinomialNBClassification(label="Survived"))
table.pipe(BernoulliNBClassification(label="Survived"))
```

---

## 1.6 回归模型 (regression)

```python
from easymlops.table.regression import *

# LightGBM回归
table.pipe(LGBMRegression(
    label="SalePrice",
    n_estimators=100,
    learning_rate=0.1
))

# 线性回归
table.pipe(LinearRegression(label="SalePrice"))

# 岭回归
table.pipe(RidgeRegression(label="SalePrice"))

# RidgeCV回归
table.pipe(RidgeCVRegression(label="SalePrice"))

# SVM回归
table.pipe(SVMRegression(label="SalePrice"))

# 逻辑回归
table.pipe(LogisticRegression(label="SalePrice"))
```

---

## 1.7 集成学习 (ensemble)

```python
from easymlops.table.ensemble import Parallel

# 并行训练多个模型
table.pipe(Parallel([
    ("lgbm", LGBMClassification(label="Survived")),
    ("rf", RandomForestClassification(label="Survived")),
], n_jobs=-1))
```

## 1.8 因子分解机 (Factorization Machine)

```python
from easymlops.table.fm import (
    FMClassification, FMRegression,
    FFMClassification, FFMRegression,
    DeepFMClassification, DeepFMRegression
)

# FM (Factorization Machine) 分类
table.pipe(FMClassification(
    y=label,
    cols=["feature1", "feature2", "feature3"],
    field_dims=[5, 10, 8],
    embed_dim=8,
    epochs=10,
    learning_rate=0.01
))

# FFM (Field-aware Factorization Machine) 分类
table.pipe(FFMClassification(
    y=label,
    cols=["feature1", "feature2", "feature3"],
    field_dims=[5, 10, 8],
    embed_dim=8,
    epochs=10
))

# DeepFM 分类 (结合FM和深度神经网络)
table.pipe(DeepFMClassification(
    y=label,
    cols=["feature1", "feature2", "feature3"],
    field_dims=[5, 10, 8],
    embed_dim=8,
    hidden_layers=[64, 32],
    epochs=10
))

# FM 回归
table.pipe(FMRegression(
    y=label,
    cols=["feature1", "feature2", "feature3"],
    field_dims=[5, 10, 8],
    embed_dim=8,
    epochs=10
))
```

---

## 1.9 Stacking 集成

```python
from easymlops.table.strategy import HillClimbingStackingRegression

# Hill Climbing + Stacking
table.pipe(HillClimbingStackingRegression(
    label="target",
    base_models=[...],
    meta_model=...
))
```

### 其他策略模块

```python
from easymlops.table.strategy import AutoResidualRegression, FactorStrategy, VarRuler

# 自动残差回归
table.pipe(AutoResidualRegression(label="target"))

# 因子策略
table.pipe(FactorStrategy())

# 变异性规则
table.pipe(VarRuler())
```

### LGBM回归层

```python
from easymlops.table.strategy import LGBMRegressionLayers, AutoResidualRegression

# LGBM回归层（用于深度学习中的自定义层）
table.pipe(LGBMRegressionLayers(
    label="target",
    n_estimators=100,
    learning_rate=0.1,
    num_leaves=31,
    output_name="lgbm_output"
))

# 自动残差回归（LGBMRegressionLayers的子类）
table.pipe(AutoResidualRegression(
    label="target",
    base_model=None,
    n_estimators=100
))
```

### 快速预测器

```python
from easymlops.table.strategy import FasterLgbSinglePredictor

# 快速LGB单类预测器
predictor = FasterLgbSinglePredictor(
    model_path="model.lgb",
    input_names=["feature1", "feature2"],
    output_name="prediction"
)

# 预测
result = predictor.predict(data)
```

### 爬山法 (Hill Climbing)

```python
from easymlops.table.strategy.hill_climbing import Climber, ClimberCV, HillClimbingStackingRegression

# 爬山法基类
climber = Climber(
    X=X_train,
    y=y_train,
    metric="rmse",
    n_folds=5,
    random_state=42
)

# CV爬山法
climber_cv = ClimberCV(
    X=X_train,
    y=y_train,
    metric="rmse",
    n_folds=5,
    random_state=42
)

# Hill Climbing Stacking
hc_stacking = HillClimbingStackingRegression(
    label="target",
    n_folds=5,
    metric="rmse",
    random_state=42
)
```

### 工具类

```python
from easymlops.table.strategy.hill_climbing.utils import COLORS

# 颜色工具类（用于命令行输出）
print(COLORS.RED + "错误信息" + COLORS.RESET)
print(COLORS.GREEN + "成功信息" + COLORS.RESET)
print(COLORS.YELLOW + "警告信息" + COLORS.RESET)
```
