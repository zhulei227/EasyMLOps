# Playbook：易续(保)模型 + 客户流失预警（同族合并）

## 业务目标

预测保单/客户"是否续保（或是否流失）"，支撑续保催办与挽留资源分配。与流失预警（churn）同构，仅标签口径不同，故合并一篇。本质：**表格二分类 + 时间切分 + 分层运营**。

## ① 标签与样本 checklist

- [ ] 续保口径：宽限期内是否完成续保（注意宽限期长度影响标签成熟）
- [ ] 流失口径：N 个月无交互/无保单（业务定义）
- [ ] **观察点对齐**：特征只取"到期日前 X 天"之前的数据，防止泄漏
- [ ] 多周期客户：一个客户多个保单年度 → 注意样本聚合层级（客户级 vs 保单级）
- [ ] 时间切分按到期日

## ② 特征设计 checklist

- [ ] 承保特征：险种/保费/保额/缴费方式/渠道
- [ ] 行为特征：缴费及时率、客服互动、APP 活跃（`Divide` 派生比率）
- [ ] 历史续保序列：上期是否续保（`CrossCategoryWithNumber` 客户×险种续保率）
- [ ] 价格敏感：保费变化率（本期/上期，`Divide`）
- [ ] 缺口标注：续保**时点**预测（生存分析）与挽留 uplift → 外补 lifelines / causalml

## ③ Pipeline 模板（实测 API）

```python
from easymlops import TablePipeLine
from easymlops.table.preprocessing import FixInput, FillNa
from easymlops.table.encoding import TargetEncoding, OneHotEncoding, LabelEncoding
from easymlops.table.feature_selection import IVFilter, LGBMEmbed
from easymlops.table.classification import LGBMClassification
from easymlops.table.ensemble import Parallel

rn = TablePipeLine()
rn.pipe(FixInput(reduce_mem_usage=False)) \
  .pipe(FillNa(cols=[数值列], fill_mode="mean")) \
  .pipe(Parallel([
      OneHotEncoding(cols=[低基数列]),
      TargetEncoding(cols=[渠道, 险种], y=y_train),   # 渠道/险种续保率编码
  ])) \
  .pipe(IVFilter(y=y_train, min_threshold=0.02)) \
  .pipe(LGBMClassification(y=y_train, use_faster_predictor=False))
rn.fit(x_train)
```

## ④ 迭代路径

1. 基线 LGBM → 2. 续保率编码（历史行为编码通常贡献最大）→ 3. 保费变化率等交叉特征
4. `IVFilter`/`LGBMEmbed` 收敛特征集 → 5. 按到期月分 oot 验证稳定性

## ⑤ 分层运营与阈值

```python
from easymlops.table.utils import calc_precision_recall_at_thresholds
pr = calc_precision_recall_at_thresholds(y_true, y_prob, bins=10)
```

- 按分数分层：高续保倾向（免打扰）/ 中（短信提醒）/ 低（人工电访）
- 分层阈值随催办资源与挽留成本动态调整
- 实时评分：`save/load` + `transform_single`（record 字段须与 fit 输入列一致）

## ⑥ 工具缺口

- 生存分析（何时续保/流失）→ lifelines
- 挽留 uplift（谁被触达才续保）→ causalml 或两模型法
- 保费收入联合预测 → 见 09_forecasting_ts.md
