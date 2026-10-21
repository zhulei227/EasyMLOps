# Playbook：催收评分 / 额度管理

## 业务目标

对逾期/续期失败客户预测"回款概率"，指导催收资源排序与额度调整。本质：**表格二分类（是否回款）+ 回归（回款金额）+ 账龄分段管理**。

## ① 标签与样本 checklist

- [ ] 回款口径：观察期内回款金额>0 / 回款率>X%（业务定义）
- [ ] **账龄分层建模**：M0/M1/M2/M3 分开建模或加账龄特征（行为差异大）
- [ ] vintage/迁徙率分析 → **外补**（自建透视表，工具无内置）
- [ ] 时间切分按逾期起始日
- [ ] 回款金额回归：大量 0 值 → 两段式（是否回款 + 回款多少）

## ② Pipeline 模板（实测 API）

```python
from easymlops import TablePipeLine
from easymlops.table.preprocessing import FixInput, FillNa
from easymlops.table.encoding import TargetEncoding, LabelEncoding
from easymlops.table.classification import LGBMClassification
from easymlops.table.regression import LGBMRegression

# 回款概率（分类）
coll = TablePipeLine()
coll.pipe(FixInput(reduce_mem_usage=False)) \
    .pipe(FillNa(cols=[数值列], fill_mode="mean")) \
    .pipe(LabelEncoding(cols=[类别列])) \
    .pipe(TargetEncoding(cols=[催收员, 渠道], y=y_train)) \
    .pipe(LGBMClassification(y=y_train, use_faster_predictor=False))
coll.fit(x_train)

# 预期回款金额（回归，两段式第二步）
amt = TablePipeLine()
amt.pipe(FixInput(reduce_mem_usage=False)) \
    .pipe(FillNa(cols=[数值列], fill_mode="mean")) \
    .pipe(LGBMRegression(y=y_amount, use_faster_predictor=False))
amt.fit(x_train_paid)   # 仅用有回款样本训练金额模型
```

## ③ 额度管理

额度 = f(风险分, 能力分)。调额建议用分数分箱 + 业务规则，勿直接用原始概率。

## ④ 迭代路径

1. 账龄分段基线 → 2. 历史回款行为编码 → 3. 催收动作特征（触达次数/承诺还款）
4. `IVFilter(y=y)` 收敛 → 5. 按月 oot 验证 → 6. 两段式金额模型

## ⑤ 工具缺口

vintage/迁徙率报表（自建透视表）、账龄滚动率定义（业务方）、SHAP 归因（shap）——外补。
