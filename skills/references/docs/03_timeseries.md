# 时序任务 (TSPipeLine)

> 以下内容已对照 easymlops 0.2.4 实测校正（分组面板、持久化、单条推理均验证通过）。

## 分组面板能力（0.2.4 核心增强）

时序模块全面支持面板（多实体）数据：预处理/特征/统计模型均提供 `group_cols`（分组，低基数维度如渠道/省市）与 `time_col`（组内时间排序）参数。

- 批量 `transform` 是**纯函数**（不推进缓存）；单条 `transform_single` 流式演进，两者语义一致（预测类模型除外，见下）
- 组内计算不跨组（fillna/lag/rolling 均按组隔离）；未见新组自动 warmup 或回退
- `sort_rows=False`（默认）保持原行序与 index，行数守恒，兼容 auto_test

### 分组预处理与特征

| 类 | 说明 |
|----|------|
| `TSDiff` / `TSPctChange` / `TSLogReturn` | 可逆变化量（差分/百分比/对数收益），组边界用 fit 时保存的 seed 还原 |
| `TSBoxCox` | Box-Cox/Yeo-Johnson 幂变换（可逆；lambda 自动估计并裁剪至 [-2,2]，随 save/load 持久化） |
| `TSZScoreScaler` / `TSMinMaxScaler` | 可逆标准化（`per_group=True` 时按组学统计量） |
| `TSFillNa` | 时序填充（ffill/linear/value，组内不跨组） |
| `FixTimeInput` / `TimeResample` / `TimeGapsFill` | 时间解析/重采样/网格补齐（后两者会改变行数，勿纳入 auto_test） |
| `LagFeature` / `RollingFeature` / `EWMAFeature` / `ExpandingFeature` | 滞后/滚动/指数加权/累积特征（组内因果） |
| `SeasonalMean` / `MovingAverageTrend` / `FourierFeature` / `STLDecompose` | 季节项/趋势/傅里叶/STL 分解 |
| `RollingCorrCov` | 双列滚动相关/协方差 |

### 工具 (`easymlops.ts.utils`)

`split_time_series`（防泄漏切分）、`walk_forward_eval`（滚动回测）、`calc_adf/kpss/acf_pacf`（平稳性）、`calc_mape/smape/mase`（评估）、`calc_sample_entropy/calc_hurst`（复杂度/持续性）。

```python
from easymlops.ts.utils import split_time_series

# 防泄漏切分: 组内按时间切, 同一实体不跨 train/test 边界
train, test = split_time_series(data, time_col="date", group_cols=["group"], test_size=0.3)
```

## 统计模型 (`easymlops.ts.statistical`)

| 类名 | 说明 |
|------|------|
| **ArimaRegression** | ARIMA（真实签名: `output_col, input_cols, p, d, q, seasonal_*, seasonal_period, exog_cols, group_cols, time_col`） |
| **SarimaRegression** | SARIMA 季节性（seasonal_* 默认开启, 默认周期 12） |
| **ArimaxRegression** | ARIMAX 带外生变量 |
| **SarimaxRegression** | SARIMAX 带外生变量的季节性预测 |
| **GarchRegression** | GARCH 波动率（依赖 arch, 0.2.4 起已声明） |

0.2.4 起全部迁移到分组框架：提供 `group_cols` 时**每个实体独立拟合一个模型**，forecast 延续各自序列；`transform_single` 为该组 forecast 1 步。

```python
from easymlops.ts.statistical import ArimaRegression
from easymlops.ts.utils import split_time_series

train, test = split_time_series(data, time_col="date", group_cols=["group"], test_size=0.3)

# 分组 ARIMA: 面板数据每组独立建模
pipe = ArimaRegression(input_cols=["value"], p=1, d=1, q=1,
                       group_cols=["group"], time_col="date")
pipe.fit(train)
result = pipe.transform(test)      # 追加 arima_predict 列, 行数守恒

# 持久化: 模型对象随 save/load 持久化, load 后无需重新 fit(0.2.4 起)
pipe.save("arima.pkl")
loaded = ArimaRegression(input_cols=["value"], p=1, d=1, q=1,
                         group_cols=["group"], time_col="date")
loaded.load("arima.pkl")

# 单条推理: 该组 forecast 1 步
pipe.transform_single(dict(test.iloc[0]))["arima_predict"]
```

## 深度学习模型（torch，0.2.4 已实测签名/持久化/单条推理）

| 模块 | 类 |
|------|----|
| `easymlops.ts.deep_learning` | **NBeatsRegression / NHiTSRegression / DeepARRegression / GPRegression** |
| `easymlops.ts.transformer` | **TFTRegression / InformerRegression / AutoformerRegression / FEDformerRegression / PatchTSTRegression / TimesNetRegression / iTransformerRegression** |
| `easymlops.ts.state_space` | **DeepStateRegression / MambaRegression / LiquidS4Regression** |
| `easymlops.ts.generative` | **VAERegression / NormalizingFlowRegression / DiffusionRegression** |

```python
from easymlops.ts.deep_learning.nbeats import NBeatsRegression

pipe = NBeatsRegression(input_cols=["value"], time_period=30,
                        forecast_horizon=7, epochs=5)   # epochs 控制训练轮数
pipe.fit(df)
result = pipe.transform(df)        # 追加 nbeats_predict 列（首步预测, 标量广播, 行数守恒）
pipe.save("nbeats.pkl")            # state_dict 随 save/load 持久化
pipe.transform_single(dict(df.iloc[-1]))["nbeats_predict"]
```

要点：
- 批量 `transform` 是多步 forecast（取输入序列末窗），与逐条 `transform_single`（以 fit 序列为历史起点推进窗口）语义不同；预测类 pipe 不纳入 auto_test
- VAE/Diffusion 的 transform 含随机采样，复现需固定 `torch.manual_seed`

## GroupForecast：深度模型的面板分组

深度模型自身不支持 `group_cols`；面板数据用 `GroupForecast` 包装（每组 deepcopy 一份模板模型独立拟合），支持 save/load 与单条推理：

```python
from easymlops.ts.core.group_forecast import GroupForecast
from easymlops.ts.transformer.transformer import TFTRegression

gf = GroupForecast(model=TFTRegression(input_cols=["value"], time_period=30,
                                       forecast_horizon=7, epochs=5),
                   group_cols=["group"], time_col="date")
gf.fit(train)
out = gf.transform(test)           # 追加 tft_predict 列
gf.save("gf.pkl")                  # 各组模型权重全部持久化
```

## 回调检查

```python
from easymlops.ts.core.callback import check_shape, check_columns, check_data_type
# 注意: 预测类 pipe 的批量(多步 forecast)与单条(1 步)语义不同, 不适用数值一致性检查
```

## 测试

```bash
python tests/test_ts_all.py    # 门禁: 11 个套件全量验收, 任一失败退出码为 1
```

| 套件 | 覆盖 |
|------|------|
| `test_ts_ops.py` (P0) | 基础特征/可逆变换/分组边界 |
| `test_ts_ops_p1.py` (P1) | 重采样/扩展特征/平稳性/Parallel 分支 |
| `test_ts_ops_p2.py` (P2) | 变化量/幂变换/二阶滚动/STL/熵 |
| `test_ts_ops_p3.py` (P3) | 统计模型分组迁移对拍 statsmodels |
| `test_ts_ops_p4.py` (P4) | 深度模型权重持久化/单条推理/input_cols 解析 |
| `test_ts_ops_p5.py` (P5) | GroupForecast 分组预测包装器 |
| `test_ts_statistical/deep_learning/transformer/state_space/generative.py` | 各模型套件 |

## 环境依赖

| 模型类别 | 依赖 |
|----------|------|
| 预处理/特征 | pandas, numpy（已随包声明） |
| 统计模型 | statsmodels（已声明）, GARCH 另需 arch（已声明） |
| 深度/Transformer/状态空间/生成模型 | torch（可选依赖, 需自行安装） |
