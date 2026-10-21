# Playbook：反欺诈 / 反洗钱（极不平衡二分类）

## 业务目标

从交易/投保/理赔中识别可疑行为，输出可疑度分数供人工调查。核心挑战：**极不平衡（可疑样本常 <0.1%）+ 对抗性强 + 误报成本高**。

## ① 标签与样本 checklist

- [ ] 标签来自人工调查确认（注意确认周期长、标签延迟）
- [ ] 极不平衡：重采样外补 imblearn（easymlops 无内置）
- [ ] 评估：accuracy 无意义；用 PR-AUC、top-k 命中率
- [ ] 按交易/报案时间切分；对抗场景需定期重训

## ② 特征设计 checklist

- [ ] 频次：单位时间交易次数/金额（`Divide`、`SQL(sql="...")`）
- [ ] 集中度：对手方/地区/设备集中度（`CrossCategoryWithNumber`）
- [ ] 偏离度：当前金额/历史均值、夜间交易占比
- [ ] 图特征（共用设备/账户环路）→ **外补** networkx
- [ ] 泄漏红线：调查结论、人工评级字段严禁入模

## ③ Pipeline 模板（实测 API）

```python
from easymlops import TablePipeLine
from easymlops.table.preprocessing import FixInput, FillNa
from easymlops.table.encoding import LabelEncoding
from easymlops.table.feature_selection import LGBMEmbed, VarianceFilter
from easymlops.table.classification import LGBMClassification

aml = TablePipeLine()
aml.pipe(FixInput(reduce_mem_usage=False)) \
   .pipe(FillNa(cols=[数值列], fill_mode="mean")) \
   .pipe(LabelEncoding(cols=[类别列])) \
   .pipe(VarianceFilter(min_threshold=0.0)) \
   .pipe(LGBMEmbed(y=y_train)) \
   .pipe(LGBMClassification(y=y_train, use_faster_predictor=False))
aml.fit(x_train)
```

- 规则+模型双轨：规则层（黑名单）独立于模型层；训练前外部重采样，评估用原始分布

## ④ 阈值与告警分级

```python
from easymlops.table.utils import calc_precision_recall_at_thresholds
pr = calc_precision_recall_at_thresholds(y_true, y_prob, bins=20)
```

- 按调查人力定 top-k：先看 top 1% 命中率，再定告警线
- 三级告警：高（必查）/ 中（抽查）/ 低（记录）
- 月度监控分数漂移与命中率，下降即重训（save/load）

## ⑤ 迭代路径

1. 规则打底 → 2. LGBM 基线 → 3. 频次/集中度/偏离度特征族
4. `LGBMEmbed` 收敛特征 → 5. 月度重训 → 6. 误报回流做难例特征

## ⑥ 工具缺口

重采样（imblearn）、图特征（networkx）、实时流式评分、可解释报告（shap）——外补。
