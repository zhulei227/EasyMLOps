# Playbook：核保风控（自动核保 / 拒保与人工核保分流）

## 业务目标

用历史承保+赔付数据预测新单风险等级，实现"自动通过 / 人工核保 / 拒保"分流。本质是**表格二分类 + 评分卡式阈值管理**，与 easymlops 能力高度契合（内置 WOE/IV 即风控血统）。

## ① 标签与样本 checklist

- [ ] 标签口径：观察期内是否"出险/赔付/高风险事件"（口径由产品定义）
- [ ] 表现期足够长（如保单年度），剔除标签未成熟样本
- [ ] **accept-only 偏差**：只有核保通过的单子有表现——被拒样本需拒绝推断（外补），或先用通过件建模并在监控中校准
- [ ] 时间切分：按保单生效日切 train/oot（out-of-time），禁用随机切分
- [ ] 类别不平衡：出险率通常很低，重采样外补 imblearn；评估用 AUC/PR 而非 accuracy

## ② 特征清单 checklist

- [ ] 投保人：年龄/性别/职业/地区/收入 proxy
- [ ] 保单：险种/保额/缴费方式/渠道
- [ ] 健康告知/历史赔付——**只能用核保时点前可得的数据**（泄漏红线：理赔金额、结案结论等理赔后字段严禁入模）
- [ ] 交叉派生：`Add/Divide`（如 保额/收入）、`CrossCategoryWithNumber`（地区×险种）

## ③ Pipeline 模板（实测 API）

```python
from easymlops import TablePipeLine
from easymlops.table.preprocessing import FixInput, FillNa
from easymlops.table.encoding import WOEEncoding, OneHotEncoding, LabelEncoding
from easymlops.table.feature_selection import IVFilter
from easymlops.table.classification import LGBMClassification, LogisticRegressionClassification
from easymlops.table.ensemble import Parallel

uw = TablePipeLine()
uw.pipe(FixInput(reduce_mem_usage=False)) \
  .pipe(FillNa(cols=[数值列], fill_mode="mean")) \
  .pipe(Parallel([
      OneHotEncoding(cols=[低基数类别列]),
      WOEEncoding(y=y_train, cols=[高基数类别列]),   # WOE 是风控标准编码
  ])) \
  .pipe(IVFilter(y=y_train, min_threshold=0.02)) \
  .pipe(Parallel([
      LogisticRegressionClassification(y=y_train, prefix="lr"),   # 可解释基线
      LGBMClassification(y=y_train, prefix="lgbm", use_faster_predictor=False),
  ]))
uw.fit(x_train)
pred = uw.transform(x_test)   # 输出 lr_* 与 lgbm_* 概率列
```

## ④ 迭代优化路径（典型 5 轮）

1. **基线**：LabelEncoding + LGBM，拿到 AUC 基准
2. **编码升级**：高基数列换 `WOEEncoding(y=y)`（风控语义），低基数 OneHot
3. **特征筛选**：`IVFilter(y=y, min_threshold=0.02)` 剔弱特征 → 防过拟合
4. **双模型融合**：LR（合规可解释）+ LGBM（效果）`Parallel` 并行输出，业务侧加权
5. **稳定性验证**：train vs oot 分数分布比对（PSI 思路）；`check_transform_function_pipeline` 回归校验

## ⑤ 评估与阈值

```python
from easymlops.table.utils import calc_precision_recall_at_thresholds, calc_roc_at_thresholds
pr = calc_precision_recall_at_thresholds(y_true, y_prob, bins=10)   # 按阈值看通过率/漏损
roc = calc_roc_at_thresholds(y_true, y_prob, bins=10)
```

- 分流阈值由业务定（自动通过率 vs 漏检率的成本权衡）
- **缺口外补**：KS 统计量、评分卡刻度（PDO/基准分）→ toad / scorecardpy；SHAP 归因 → shap

## ⑥ 部署与监控

- `save/load` + `scripts/deploy_fastapi.py` 模板；`transform_single` 实时核保评分
- 监控：分数 PSI（0.1/0.25 分级告警）、特征缺失率、人工核保率漂移
- `FixInput(reduce_mem_usage=False)` 防止 float16 污染下游统计（SKILL.md 陷阱3）
