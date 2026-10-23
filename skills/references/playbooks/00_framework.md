# 业务任务迭代优化框架（Playbook 总纲）

本目录是面向具体业务任务的建模迭代指导。所有代码均映射到 easymlops 0.2.4 **实测可用**的 API；工具缺口显式标注"外补"。业务规则（标签口径、阈值策略）以 checklist 形式给出，由业务方填写。

## 通用迭代闭环（7 步）

```text
① 标签与样本设计 → ② 探查与清洗 → ③ 基线模型 → ④ 特征迭代
→ ⑤ 模型迭代 → ⑥ 阈值/校准/稳定性 → ⑦ 部署与监控
         ↑______________④~⑦ 不达标则回到④______________|
```

### ① 标签与样本设计（业务方 checklist）

- [ ] 标签定义：正样本的业务口径（如"出险"/"续保"/"点击"）、观察期(window)与表现期(performance)长度
- [ ] 样本切分：**必须按时间切分**（train 取早于 test 的时段），防止时间泄漏
- [ ] 样本偏差：是否存在 accept-only（核保通过后才有表现）、 survivorship 问题
- [ ] 外部补充：极不平衡场景的重采样工具（imblearn 等，easymlops 无内置）

### ② 探查与清洗 → easymlops 映射

```python
from easymlops import TablePipeLine
from easymlops.table.preprocessing import FixInput, FillNa, DropCols
clean = TablePipeLine()
clean.pipe(FixInput(reduce_mem_usage=False)) \
     .pipe(FillNa(cols=[数值列], fill_mode="mean")) \
     .pipe(FillNa(cols=[类别列], fill_detail={...})) \
     .pipe(DropCols(cols=[ID列]))
```

### ③ 基线模型（先跑通再优化）

```python
from easymlops.table.preprocessing import FixInput, FillNa
from easymlops.table.encoding import LabelEncoding
from easymlops.table.classification import LogisticRegressionClassification, LGBMClassification
base = TablePipeLine()
base.pipe(FixInput(reduce_mem_usage=False)) \
    .pipe(FillNa(cols=[数值列], fill_mode="mean")) \
    .pipe(LabelEncoding(cols=[类别列])) \
    .pipe(LGBMClassification(y=y_train, use_faster_predictor=False))
base.fit(x_train)
```

### ④ 特征迭代（按性价比排序）

| 手段 | API |
|------|-----|
| 类别编码 | `OneHotEncoding` / `TargetEncoding(y=y)` / `WOEEncoding(y=y)` |
| 交叉派生 | `Add/Divide/...`(left/right_col_name)、`CrossCategoryWithNumber`、`CrossNumberWithNumber`、`SQL(sql="...as ...")` |
| 特征筛选 | `MissRateFilter/VarianceFilter`（无监督）、`PersonCorrFilter/IVFilter`（y）、`LGBMEmbed`（重要性） |
| 叶子编码 | `LGBMLeafEncoder(y=y, cols=[...])`（注意：输出整体替换为叶子列） |
| 并行多路 | `Parallel([...])`（输出自动横向拼接，`prefix` 区分模型输出） |

### ⑤ 模型迭代

- LR（可解释）→ LGBM（效果）→ 并行多模型（`Parallel`）→ `HillClimbingStackingRegression(y=y)` 加权融合
- 回归任务用 `LGBMRegression(y=y, use_faster_predictor=False)`

### ⑥ 阈值/校准/稳定性

```python
from easymlops.table.utils import calc_precision_recall_at_thresholds, calc_roc_at_thresholds
table = calc_precision_recall_at_thresholds(y_true, y_prob, bins=10)  # 选业务阈值
```

- 时间稳定性：新旧样本各跑一遍 pipeline，比对分布（`PSIFilter` 的 oot 思路，或自算 PSI）
- 质量检查：`from easymlops.table.callback import check_transform_function_pipeline`

### ⑦ 部署与监控

```python
pipe.save("model.pkl")            # save 存参数字典
from easymlops import TablePipeLine
online = TablePipeLine(); online.load("model.pkl")
online.transform_single(record)   # 单条推理（record 字段须与 fit 输入列一致）
```

- 服务化模板见 `scripts/deploy_fastapi.py`
- 上线后监控：分数分布漂移（PSI）、特征缺失率、口径一致性

## 工具缺口总表（各 playbook 引用）

> 2026-10 更新：以下缺口已在本仓库源码内置（`tests/test_diag_tools.py` 实测通过），
> 优先用内置实现，外补库仅在其覆盖不足时使用：
> - **重采样/类不平衡** → `easymlops.table.utils.resample`（oversample/undersample/smote）+ `LGBMClassification/LGBMRegression(weight=)` 样本权重
> - **KS/评分卡刻度/lift** → `calc_ks` / `calc_lift` / `prob2score`（`easymlops.table.utils`）
> - **PSI 漂移监控** → `easymlops.table.drift.DriftMonitorPipe`（透传逐列监控，替代实测不稳定的 `PSIFilter`）+ `calc_psi`

| 缺口 | 影响任务 | 外补建议 |
|------|---------|---------|
| ~~重采样/类不平衡~~（已内置 `resample`/`weight`） | 反欺诈、反洗钱 | 内置无法覆盖时再用 imblearn |
| ~~KS/评分卡刻度/lift~~（已内置 `calc_ks`/`calc_lift`/`prob2score`） | 核保、催收、反欺诈 | 评分卡变量细分箱仍可用 toad/scorecardpy |
| uplift 建模 | 营销响应 | 手工两模型法(T-learner) 或 causalml |
| 生存分析 | 续保时点、LTV | lifelines |
| walk-forward / 回测 | 时序预测、股价 | 自写滚动循环 / vectorbt |
| 图特征 | 反欺诈、理赔关联 | networkx / 图数据库 |
| SHAP 归因 | 全部 | shap（对 LGBM 模型对象） |
| NDCG/MRR 排序指标、A/B 平台 | 推荐 | 自实现 / 实验平台 |
| ~~PSI 漂移监控~~（已内置 `DriftMonitorPipe`/`calc_psi`） | 全部上线监控 | 复杂监控平台另建 |
| 概率校准 | 定价、定价类回归校准 | 已内置 `IsotonicCalibrationPipe`（`easymlops.table.calibration`） |
