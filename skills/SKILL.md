---
name: easymlops
description: "This skill should be used when building machine learning modeling tasks with the EasyMLOps framework (pip install easymlops), including business task playbooks. Covers Pipeline-based table data modeling (cleaning, encoding, feature selection, decomposition, FM/DeepFM, classification/regression), NLP tasks, time series, YOLO vision, OCR, AutoML, model persistence, and FastAPI deployment. Business playbooks with iterative optimization workflows: 核保风控 underwriting, 理赔风控 claims risk/fraud, 反欺诈反洗钱 anti-fraud/AML, 易续保 renewal, 客户流失 churn, 推荐系统/交叉销售 recommendation/cross-sell CTR, 催收评分 collection, 营销响应 marketing uplift, 车险定价/赔付金额/医疗费用预测 pricing, 保费收入/赔付率/需求销量时序预测 forecasting. Triggers include EasyMLOps, TablePipeLine, FM, WOE, IV, 核保, 理赔, 续保, 流失, 反欺诈, 推荐系统, 特征工程, 模型部署."
---

# EasyMLOps 机器学习 Pipeline 框架

EasyMLOps 是一个基于 Pipeline 模式构建的机器学习建模任务框架，提供从数据处理、特征工程、模型训练到生产部署的完整流程支持。只依赖 `pip install easymlops` 即可使用核心表格功能；视觉/OCR 任务需要额外依赖（见下文）。

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
| **AutoML** | 自动化机器学习（含 LLM 集成） |

## 典型工作流程

1. 数据输入 → `FixInput()` 确保输入为 DataFrame 格式
2. 预处理 → `FillNa()`、列选择/重命名、类型转换
3. 特征工程 → 用 `Parallel([...])` 并行执行多种编码，多路结果自动横向拼接
4. 模型训练 → `pipeline.fit(x_train)`，模型 pipe 需传入 `y=y_train` 或 `label=...`
5. 预测/评估 → `transform` / `transform_single`（单条预测），配合 PR/ROC 曲线评估
6. 持久化 → `pipeline.save("./model.pkl")` / `TablePipeLine.load("./model.pkl")`
7. 部署 → 用 Flask/FastAPI 封装 `transform_single` 实现服务化上线
8. 特征存储与监控 → 通过 storage 模块存储特征

## 最小示例（表格分类）

```python
import pandas as pd
from easymlops import TablePipeLine
from easymlops.table.preprocessing import *
from easymlops.table.encoding import *
from easymlops.table.classification import *
from easymlops.table.ensemble import Parallel

data = pd.read_csv("./data/demo.csv")
x_train, x_test = data[:500], data[500:]
y_train = x_train["Survived"]
del x_train["Survived"]; del x_test["Survived"]

table = TablePipeLine()
table.pipe(FixInput()) \
  .pipe(FillNa()) \
  .pipe(Parallel([
      OneHotEncoding(cols=["Pclass", "Sex"]),
      TargetEncoding(cols=["Name", "Ticket"], y=y_train)
  ])) \
  .pipe(LGBMClassification(y=y_train))

table.fit(x_train)
result = table.transform(x_test)
table.transform_single(x_test.to_dict("record")[0])  # 单条预测
```

## 示例脚本（scripts/，全部实测跑通）

可在安装 easymlops 后直接运行，产物（模型 pkl、样例 CSV 副本）写入 `scripts/output/`：

| 脚本 | 内容 | 备注 |
|------|------|------|
| `example_feature_engineering.py` | 特征工程(上)：清洗/派生/编码/特征选择 4 板块 | 无额外依赖 |
| `example_feature_engineering_advanced.py` | 特征工程(下)：降维/SQL/Pipeline操作/质量检查/性能优化/全流程串联 | 无额外依赖 |
| `example_classification.py` | 分类训练/评估/单条预测/save+load 一致性 | 生成 `output/cls_model.pkl` |
| `example_regression.py` | Parallel 并行 LGBM+线性回归，RMSE/R² 评估 | 无额外依赖 |
| `example_timeseries.py` | ARIMA 训练预测 + pickle 持久化 | 需 statsmodels |
| `deploy_fastapi.py` | FastAPI 加载模型提供 /predict、/health | 需先跑分类脚本；需 fastapi、uvicorn |

特征工程样例（两个脚本共 9 大板块，每块均有输出断言）：

1. 数据清洗与类型规范化：`FixInput`/`TransTo*`/`FillNa`(fill_mode+fill_detail)/`DropCols`/`TransToLower`
2. 特征派生：二元运算 `Add/Divide`(left/right_col_name)、多列汇总 `Sum/Mean`(drop_input=False)、日期特征 `DateMonthInfo/DateHourInfo`、交叉特征 `CrossCategoryWithNumber/CrossNumberWithNumber`、`Bins/Clip/Tanh/IsNull`
3. 特征编码：`OneHot/Label`(Parallel 并行)、`TargetEncoding/WOEEncoding`(y)、`LGBMLeafEncoder`（输出整体替换为叶子列）
4. 特征选择：`MissRate/Variance/PersonCorr/Chi2/IV` 过滤器 + `LREmbed/LGBMEmbed` 嵌入式
5. 特征降维：`PCADecomposition`
6. SQL 表达式派生列：`SQL(sql="(age*score)/2 as ...")`
7. Pipeline 操作与质量检查：切片 `p[0]`、`transform_single`、减法运算符（有缺陷，见陷阱）、`check_transform_function_pipeline`
8. 性能优化：`ReduceMemUsage`/`Dense2Sparse`
9. 全流程串联：清洗→派生→编码→选择，10 列扩展到 24 列

运行方式（在 skill 根目录）：

```bash
python scripts/example_classification.py
python scripts/deploy_fastapi.py   # 然后访问 http://127.0.0.1:8901/health 与 /predict
```

测试数据样例（`assets/data/`，由脚本确定性生成，可直接查看）：

- `fe_sample_raw.csv`：特征工程原始样例（数值/类别/日期列，含缺失值、零方差列、全缺失列）
- `fe_sample_full.csv`：全流程特征工程输出样例（10 列 -> 24 列）
- `sample_classification.csv`：二分类数据（8 数值特征 + 类别列 gender）
- `sample_regression.csv`：回归数据（6 数值特征 + 类别列 region）
- `sample_timeseries.csv`：200 点随机游走序列

## 离线安装

skill 已随附 wheel（`assets/easymlops-0.2.4-py3-none-any.whl`，与 PyPI 最新版本一致），可直接本地化安装：

```bash
pip install <本skill目录>/assets/easymlops-0.2.4-py3-none-any.whl
```

注意：wheel 不携带依赖，核心功能仍需安装 lightgbm、scikit-learn、pandas；0.2.4 起 psutil、matplotlib、arch（GARCH）已在依赖声明中；时序深度模型另需 torch，部署另需 fastapi、uvicorn。

## 额外依赖（按需安装）

```bash
# YOLO 视觉任务
pip install "numpy>=1.24.4,<2" "torch>=1.12.0" "torchvision>=0.13.0" ultralytics

# OCR 任务
pip install easyocr
pip install "numpy<2" "Pillow<10" "torch>=1.12.0" "torchvision>=0.13.0"
```

注意：YOLO 场景要求 numpy 1.x（<2）；OCR 场景要求 Pillow<10。

## 参考文档（references/docs/）

以下文档已全部对照 easymlops 0.2.4 实测校正（含导入、构造参数、方法存在性），可直接按文档编写代码。torch 深度模型签名与持久化亦在 0.2.4 实测通过（见文末版本实测陷阱第 7 条）。

按需加载对应文档，不要一次性全部读取：

| 任务场景 | 应读取的文档 |
|---------|-------------|
| 表格数据建模（预处理/编码/模型/评估） | [01_table.md](./references/docs/01_table.md) |
| NLP 任务 | [02_nlp.md](./references/docs/02_nlp.md) |
| 时序任务 | [03_timeseries.md](./references/docs/03_timeseries.md) |
| Pipeline 链式调用、切片、组合、回调检查 | [04_pipeline.md](./references/docs/04_pipeline.md) |
| 模型保存与加载 | [05_persistence.md](./references/docs/05_persistence.md) |
| Flask/FastAPI 生产部署 | [06_deploy.md](./references/docs/06_deploy.md) |
| 特征存储 | [07_storage.md](./references/docs/07_storage.md) |
| AutoML 与 LLM 集成 | [08_automl.md](./references/docs/08_automl.md) |
| 常用类速查表（所有 Pipe 类及参数） | [09_cheatsheet.md](./references/docs/09_cheatsheet.md) |
| 端到端完整示例 | [10_example.md](./references/docs/10_example.md) |
| YOLO 视觉任务 | [11_yolo.md](./references/docs/11_yolo.md) |
| OCR 文本识别 | [12_ocr.md](./references/docs/12_ocr.md) |

## 业务任务 Playbook（references/playbooks/，迭代优化指导）

面向具体业务场景的建模迭代闭环指导，每篇含：标签/样本 checklist、实测 API 的 Pipeline 模板、迭代路径、工具缺口（外补方案）。业务规则留作业务方填写。

| Playbook | 任务 |
|----------|------|
| [00_framework.md](./references/playbooks/00_framework.md) | 通用迭代优化框架（7 步闭环 + 工具缺口总表） |
| [01_underwriting.md](./references/playbooks/01_underwriting.md) | 核保风控（WOE/IV + 双模型 + 阈值分流） |
| [02_claims_risk.md](./references/playbooks/02_claims_risk.md) | 理赔风控（表格+NLP 双路 + 相似案件检索） |
| [03_renewal_churn.md](./references/playbooks/03_renewal_churn.md) | 易续保 + 客户流失预警（同族合并） |
| [04_recommendation_cross_sell.md](./references/playbooks/04_recommendation_cross_sell.md) | 推荐系统 + 交叉销售（FM/CTR + Faiss 召回） |
| [05_fraud_aml.md](./references/playbooks/05_fraud_aml.md) | 反欺诈/反洗钱（极不平衡，重采样外补） |
| [06_collection_limit.md](./references/playbooks/06_collection_limit.md) | 催收评分/额度管理（两段式，vintage 外补） |
| [07_marketing_response.md](./references/playbooks/07_marketing_response.md) | 营销响应（uplift 两模型法外补方案） |
| [08_pricing_amount.md](./references/playbooks/08_pricing_amount.md) | 车险定价/赔付金额/医疗费用（回归两段式，GLM 缺口） |
| [09_forecasting_ts.md](./references/playbooks/09_forecasting_ts.md) | 保费/赔付率/需求时序预测（含股价预测定位与限制说明） |
| [10_claims_nlp_ocr.md](./references/playbooks/10_claims_nlp_ocr.md) | 理赔文本分类 + 单证发票 OCR 审核 |

## MLE 方法论参考（references/mly/，源自 Andrew Ng《Machine Learning Yearning》）

在做建模任务的策略决策（数据集划分、评估指标设计、误差分析、瓶颈诊断、流水线架构取舍）时，按需加载对应文档；不要一次性全部读取：

| 决策场景 | 应读取的文档 |
|---------|-------------|
| 全书结构与核心思想速览、方法论→easymlops 映射 | [mly/00_overview.md](./references/mly/00_overview.md) |
| 开发/测试集划分、单一数字评估指标、优化/满足指标 | [mly/01_datasets_and_metrics.md](./references/mly/01_datasets_and_metrics.md) |
| 误差分析（错误样本分桶、Eyeball/Blackbox dev set、错标清理） | [mly/02_error_analysis.md](./references/mly/02_error_analysis.md) |
| 偏差/方差诊断、学习曲线、与人类水平对比 | [mly/03_bias_variance_learning_curves.md](./references/mly/03_bias_variance_learning_curves.md) |
| 训练/测试分布不一致、数据加权、人工数据合成、三分诊断 | [mly/04_data_mismatch.md](./references/mly/04_data_mismatch.md) |
| 优化验证测试、端到端 vs 分阶段流水线、分部件误差归因 | [mly/05_pipeline_architecture.md](./references/mly/05_pipeline_architecture.md) |

注：以上为原书方法论的自有语言提炼笔记（原书 CC BY-NC-SA，禁止商用），细节以原书（mlyearning.org 可免费获取）为准。与 playbooks/00_framework.md 的 7 步迭代闭环配套使用：①② 对应 01/04 文档，④⑤ 对应 02/03 文档，⑥⑦ 对应 05 文档。

## 诊断工具扩展（0.2.4 wheel 已包含以下全部能力）

为补齐 playbooks 反复外补的评估/诊断缺口，本仓库源码新增以下能力（`tests/test_diag_tools.py` 全部实测通过，含 auto_test 验收）：

| 能力 | 导入 | 说明 |
|------|------|------|
| PSI 分布漂移 | `from easymlops.table.utils import calc_psi` | 同分布 <0.1 稳定，>0.25 显著漂移 |
| KS 区分度 | `calc_ks(y_true, y_score)` | 返回 `{"ks", "ks_threshold"}` |
| Lift 表 | `calc_lift(y_true, y_score, bins=10)` | 按分数降序等频分箱，含 bad_rate/lift/cum_gain |
| 评分卡刻度 | `prob2score(prob, base_score=600, pdo=20, base_odds=1/19)` | 分数越高风险越低 |
| 重采样 | `resample(x, y, method="oversample"/"undersample"/"smote")` | 返回对齐的 (x,y)，**在构造 pipeline 前调用**（y 在构造时绑定）；smote 需 imblearn（懒加载） |
| 样本权重 | `LGBMClassification(y=y, weight=w)` / `LGBMRegression(y=y, weight=w)` | 长度需与 y 一致 |
| 概率校准 | `from easymlops.table.calibration import IsotonicCalibrationPipe` | 推荐用法：主 pipeline 先 fit，再 `calib.fit(table.transform(x_val)); table.pipe(calib)`（无偏）；简便用法 y=y_train 直接挂入（有偏） |
| 漂移监控 | `from easymlops.table.drift import DriftMonitorPipe` | 透传型 pipe，fit 记录基准分布，transform 逐列算 PSI：`pipe.last_psi_report_` / `pipe.last_drift_cols_`；可挂主链或 `set_branch_pipe`；也适用于模型分数监控（cols=[正类列名]） |

注意事项：
- pipe 的 `name` 默认是**类对象**而非字符串，按名检索 pipe 会失败；用 `[m for m in pipeline.models if isinstance(m, XxxPipe)]` 或整数索引。
- 仓库源码在 pandas<1.5 环境的兼容修复已合入（`copy_on_write` 设置改为防御式）。
- 校准/监控 pipe 的 y 必须与**实际流入该 pipe 的数据行**配对（`pipeline.fit(x)` 时校准 pipe 收到的是上游 transform 后的训练数据）。

## 关键注意事项

- 模型类 pipe（如 `LGBMClassification`）在 fit 时需要知道标签：构造时传入 `y=y_train`（`label=` 参数不被接受）。
- `Parallel` 内多个 pipe 的输出会自动按列拼接，模型 pipe 用 `prefix` 参数区分输出列名（如 `prefix="lgbm"`）。
- 使用 `easymlops.table.callback` 中的 `check_shape` / `check_columns` / `check_data_type` / `check_transform_function_pipeline` 验证 pipeline 正确性。
- 部署时用 `transform_single` 处理单条请求；保存的 pkl 在加载后无需重新 fit。

## 版本实测陷阱（对照 easymlops 0.2.4 验证，docs 部分描述有漂移）

以下差异已在真实环境验证，编写代码时以本节为准：

1. **`FillNa` 参数是 `fill_mode` 而非文档写的 `strategy`**，且 `fill_mode="mean"/"median"` 必须配合 `cols=[数值列列表]` 使用；`cols="all"` 遇到字符串列会因对字符串求均值而崩溃。指定值填充用 `fill_detail={"col": value}`。
2. **`MissRateFilter`/`VarianceFilter` 的参数是 `max_threshold`/`min_threshold`**，不是 `threshold`/`k`。语义：保留缺失率/方差落在 [min, max] 区间的列。两者对字符串列执行统计会报错，必须用 `cols=` 限定数值列。级联多个过滤器时，后一个过滤器的 `cols` 不能引用被前一个过滤掉的列。
3. **`FixInput` 默认 `reduce_mem_usage=True` 会把 float64 降为 float16**，可能导致数值方差溢出为 `inf`，进而被 VarianceFilter 误删；建模前建议 `FixInput(reduce_mem_usage=False)`。
4. **模型类不接受 `n_estimators`/`learning_rate`/`label=` 等构造参数**（会被 `**kwargs` 透传到 `object.__init__` 而报 TypeError）；且默认 `use_faster_predictor=True` 与 lightgbm 4.x 不兼容（要求模型版本 v3），需显式传 `use_faster_predictor=False`。输出列为各类别概率，列名是类别 id（如 "0"/"1"）。
5. **`TablePipeLine.load` 是实例方法**：`p = TablePipeLine(); p.load("model.pkl")`，不能 `TablePipeLine.load(...)`；`save()` 存的是参数字典而非整个对象。
6. **`ArimaRegression` 实际签名为 `(output_col, input_cols, p, d, q, seasonal_p/d/q, seasonal_period, exog_cols, group_cols, time_col)`**，文档中的 `time_period`/`forecast_horizon` 不存在；`transform` 会 forecast `len(s)` 步并追加 `output_col` 列（默认 `arima_predict`）。提供 `group_cols` 时每个组独立拟合一个模型（面板数据），`transform_single` 为该组 forecast 1 步。
7. **TS 预测模型的 `save/load` 会持久化拟合结果**：统计模型（ArimaRegression 家族/GarchRegression）持久化模型对象，torch 深度模型（NBeats/NHiTS/DeepAR/GP/TFT 家族/DeepState/Mamba/LiquidS4/VAE/NF/Diffusion）经 `TorchForecastMixin` 持久化 state_dict，load 后无需重新 fit 即可 transform。批量 transform 是多步 forecast（深度模型取输入序列末窗、统计模型从训练末端 forecast），与逐条 `transform_single`（以 fit 序列为历史起点推进窗口、输出首步预测）语义不同，预测类 pipe 不纳入 auto_test。VAE/Diffusion 的 transform 含随机采样，复现需固定 torch 种子。**深度模型不支持 `group_cols`**：面板数据用 `easymlops.ts.core.group_forecast.GroupForecast(model=模板模型, group_cols=..., time_col=...)` 包装（每组 deepcopy 一份模板独立拟合），同样支持 save/load 与单条推理。
8. **字符串列必须在模型 pipe 之前完成编码**（如 `OneHotEncoding`/`LabelEncoding`），否则 lightgbm 报 "pandas dtypes must be int, float or bool"。`OneHotEncoding(drop_col=True)` 为默认但原列可能保留，依赖 `LabelEncoding` 原地覆盖来消除原字符串列。
9. **`TablePipeLine` 无 `get_pipe_names()`/`pipes` 属性**，`transform` 的 `stop_at`/`start_from` 参数按 pipe 名匹配可能静默失效，调试时不要依赖它做分段排查。
10. **依赖**：0.2.4 起 `psutil`、`matplotlib`、`arch`（GARCH）已在依赖声明中；部署仍需自行安装 `fastapi`、`uvicorn`；TS 深度模型需 `torch`（可选依赖）。
11. **`Sum`/`Mean`/`Median` 默认 `drop_input=True`**，transform 后只输出汇总列、输入列被消费；需保留输入列时显式传 `drop_input=False`。二元运算（`Add` 等）则保留输入列。
12. **`LGBMLeafEncoder` 会把输出整体替换为叶子列**（`leaf_0..N`），不要与其它特征混在同一条管道尾部。
13. **运算符**：`-` 减法可用但会破坏原 pipeline 内部状态（input_col_names 被清空）且减法产物 transform 有缺陷；`+` 组合不支持（TypeError）。
14. **`Bins` 不限定 `cols` 时会对全部列（含字符串）执行分箱**而报错，需 `Bins(cols=[数值列], n_bins=...)`。
15. **`transform_single`** 的 record 字段需与 fit 输入列一致；复杂管道（含交叉特征等）可能触发 `input_col_names=None` 缺陷，单条推理建议用单行 DataFrame + `transform`。
16. **实测不稳定的 API（避免使用）**：`PSIFilter`（IndexError）、`DateDayDiff`（pandas 2 下报错）、`MapValues`（map_detail 对未覆盖值 KeyError）、`get_pipe/get_pipe_names/pipes`（不存在）。
