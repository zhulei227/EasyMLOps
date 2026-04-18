# 时序任务 (TSPipeLine)

```python
from easymlops import TSPipeLine
from easymlops.ts.dnn import TSCNNRegression

ts = TSPipeLine()

# 时序CNN回归
ts.pipe(TSCNNRegression(
    label="target",
    seq_len=30,
    features=[...],
    ...
))

# 训练
ts.fit(x_train, y=y_train)

# 预测
predictions = ts.transform(x_test)
```

## 时序数据处理

TSPipeLine 继承自 TablePipeLine，因此也可以使用表格数据的预处理方法：

```python
from easymlops.table.preprocessing import *

# 使用表格预处理
ts.pipe(FixInput())
ts.pipe(FillNa(strategy="mean"))
ts.pipe(MinMaxScaler(cols=[...]))
```

## 回调函数

```python
from easymlops.ts.core.callback import *

# 形状检查
check_shape(ts_pipe, batch_transform, single_transform)

# 列检查
check_columns(ts_pipe, batch_transform, single_transform)

# 数据类型检查
check_data_type(ts_pipe, batch_transform, single_transform)

# 数据一致性检查
check_data_same(ts_pipe, batch_transform, single_transform)
```

## 时间序列模型

easymlops.ts 模块实现了多种时间序列预测模型。

### 统计模型 (`easymlops.ts.statistical`)

| 类名 | 说明 |
|------|------|
| **ArimaRegression** | ARIMA 时间序列预测 |
| **SarimaRegression** | SARIMA 季节性时间序列预测 |
| **ArimaxRegression** | ARIMAX 带外生变量的时间序列预测 |
| **SarimaxRegression** | SARIMAX 带外生变量的季节性预测 |
| **GarchRegression** | GARCH 波动率模型 |

### 深度学习模型 (`easymlops.ts.deep_learning`)

| 类名 | 说明 |
|------|------|
| **NBeatsRegression** | N-BEATS 时间序列预测 |
| **NHiTSRegression** | N-HiTS 时间序列预测 |
| **DeepARRegression** | DeepAR 时间序列预测 |
| **GPRegression** | 高斯过程时间序列预测 |

### Transformer模型 (`easymlops.ts.transformer`)

| 类名 | 说明 |
|------|------|
| **TFTRegression** | Temporal Fusion Transformer |
| **InformerRegression** | Informer 时间序列预测 |
| **AutoformerRegression** | Autoformer 时间序列预测 |
| **FEDformerRegression** | FEDformer 时间序列预测 |
| **PatchTSTRegression** | PatchTST 时间序列预测 |
| **TimesNetRegression** | TimesNet 时间序列预测 |
| **iTransformerRegression** | iTransformer 时间序列预测 |

### 状态空间模型 (`easymlops.ts.state_space`)

| 类名 | 说明 |
|------|------|
| **DeepStateRegression** | Deep State 时间序列预测 |
| **MambaRegression** | Mamba 状态空间时间序列预测 |
| **LiquidS4Regression** | Liquid S4 状态空间时间序列预测 |

### 生成模型 (`easymlops.ts.generative`)

| 类名 | 说明 |
|------|------|
| **VAERegression** | VAE 变分自编码器时间序列预测 |
| **NormalizingFlowRegression** | Normalizing Flow 时间序列预测 |
| **DiffusionRegression** | Diffusion 时间序列预测 |

### 核心基类

所有时间序列模型继承自以下基类：

- **PipeBase** (`easymlops.ts.core.pipe`) - 管道基类，提供 save/load 功能

## 使用示例

### 统计模型

```python
from easymlops.ts.statistical.arima import ArimaRegression

# 准备数据
df = pd.DataFrame({
    'value': np.cumsum(np.random.randn(200)) + 100
})

# 创建并训练模型
pipe = ArimaRegression(input_cols=["value"], time_period=30, forecast_horizon=7)
pipe.fit(df)

# 预测
result = pipe.transform(df)

# 保存和加载模型
pipe.save("arima_model.pkl")
pipe.load("arima_model.pkl")
```

### 深度学习模型

```python
from easymlops.ts.deep_learning.nbeats import NBeatsRegression

# 准备数据
df = pd.DataFrame({
    'value': np.cumsum(np.random.randn(200)) + 100
})

# 创建并训练模型
pipe = NBeatsRegression(
    input_cols=["value"],
    time_period=30,
    forecast_horizon=7,
    epochs=5
)
pipe.fit(df)

# 预测
result = pipe.transform(df)

# 保存和加载模型
pipe.save("nbeats_model.pkl")
pipe.load("nbeats_model.pkl")
```

### 带外生变量的模型

```python
from easymlops.ts.statistical.arima import ArimaxRegression

# 准备数据
df = pd.DataFrame({
    'value': np.cumsum(np.random.randn(200)) + 100,
    'feature1': np.random.randn(200) * 10,
    'feature2': np.random.randn(200) * 5
})

# 创建并训练模型 (使用外生变量)
pipe = ArimaxRegression(
    input_cols=["value"],
    exog_cols=["feature1", "feature2"],
    time_period=30,
    forecast_horizon=7
)
pipe.fit(df)

# 预测
result = pipe.transform(df)
```

## 测试脚本

测试脚本位于 `tests/` 目录：

| 测试文件 | 说明 |
|----------|------|
| `test_ts_statistical.py` | 统计模型测试 |
| `test_ts_deep_learning.py` | 深度学习模型测试 |
| `test_ts_transformer.py` | Transformer模型测试 |
| `test_ts_state_space.py` | 状态空间模型测试 |
| `test_ts_generative.py` | 生成模型测试 |
| `test_ts_all.py` | 所有模型汇总测试 |

## 测试运行

```bash
# 运行统计模型测试
python tests/test_ts_statistical.py

# 运行所有测试
python tests/test_ts_all.py
```

## 环境依赖

| 模型类别 | 依赖 |
|----------|------|
| 统计模型 | pandas, numpy, statsmodels |
| 深度学习模型 | pandas, numpy, torch, statsmodels |
| Transformer模型 | pandas, numpy, torch, statsmodels |
| 状态空间模型 | pandas, numpy, torch, statsmodels |
| 生成模型 | pandas, numpy, torch, statsmodels |

## 注意事项

1. 统计模型测试可以直接运行
2. 深度学习/Transformer等模型需要正确的 PyTorch 环境
3. GARCH 模型需要 numba 依赖
4. 所有模型都支持 save/load 功能进行模型持久化
