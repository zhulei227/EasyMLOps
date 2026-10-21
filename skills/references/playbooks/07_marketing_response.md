# Playbook：营销响应模型（含 uplift 缺口说明）

## 业务目标

预测客户对营销活动（加保/续保催办/邀约）的响应概率，指导触达名单排序与资源分配。

## ① 标签与样本 checklist

- [ ] 响应口径：触达后 N 天内完成目标行为（购买/点击/回复）
- [ ] **必须有随机对照组（RCT）**：无对照无法区分"本来就会买"与"被营销才买"
- [ ] 按活动开始日切分；多活动叠加期样本剔除或标记

## ② 响应率 Pipeline 模板（实测 API）

```python
from easymlops import TablePipeLine
from easymlops.table.preprocessing import FixInput, FillNa
from easymlops.table.encoding import TargetEncoding, LabelEncoding
from easymlops.table.feature_selection import IVFilter
from easymlops.table.classification import LGBMClassification

mkt = TablePipeLine()
mkt.pipe(FixInput(reduce_mem_usage=False)) \
   .pipe(FillNa(cols=[数值列], fill_mode="mean")) \
   .pipe(LabelEncoding(cols=[类别列])) \
   .pipe(TargetEncoding(cols=[渠道, 险种, 客户分层], y=y_train)) \
   .pipe(IVFilter(y=y_train, min_threshold=0.02)) \
   .pipe(LGBMClassification(y=y_train, use_faster_predictor=False))
mkt.fit(x_train)
```

## ③ uplift（增量响应）——工具缺口，外补方案

easymlops 无 uplift 模块。手工两模型法（T-learner）：

- 同一特征对实验组/对照组各训练一个 `LGBMClassification`（各自 save/load）
- `uplift_score = P(响应|触达) - P(响应|未触达)`
- 或外补 causalml（meta-learner）；评估用 Qini 曲线（自实现）

## ④ 迭代路径

1. 响应率基线 → 2. 历史响应编码 → 3. 名单分层（`calc_precision_recall_at_thresholds`）
4. RCT 就绪后升级 uplift → 5. 名单 = 响应率 × uplift 综合排序

## ⑤ 工具缺口

uplift 建模（causalml/两模型法）、Qini 评估（自实现）、A/B 实验平台——外补。
