# Playbook：定价与金额预测（车险定价 / 赔付金额 / 医疗费用）

## 业务目标

回归类任务：预测**期望赔付金额**（车险定价、纯保费）或**医疗费用**，支撑定价、准备金与理赔预算。工具覆盖良好（`LGBMRegression`），缺口是精算模型族。

## ① 标签与样本 checklist

- [ ] 金额口径：年化纯保费 / 案均赔款 / 总医疗费用（含大量 0 值样本）
- [ ] 时间切分按保单年度/就诊日期；通胀/费用上涨需口径对齐
- [ ] 重尾分布：金额右偏，建议 log1p 训练、预测后 expm1 还原
- [ ] 评估：RMSE/MAE（工具外自算）、分位数校准

## ② Pipeline 模板（实测 API）

```python
from easymlops import TablePipeLine
from easymlops.table.preprocessing import FixInput, FillNa
from easymlops.table.encoding import TargetEncoding, LabelEncoding
from easymlops.table.regression import LGBMRegression, LinearRegression
from easymlops.table.ensemble import Parallel

price = TablePipeLine()
price.pipe(FixInput(reduce_mem_usage=False)) \
     .pipe(FillNa(cols=[数值列], fill_mode="mean")) \
     .pipe(LabelEncoding(cols=[类别列])) \
     .pipe(TargetEncoding(cols=[车型, 地区], y=y_train)) \
     .pipe(Parallel([
         LGBMRegression(y=y_log1p, prefix="lgbm", use_faster_predictor=False),
         LinearRegression(y=y_log1p, prefix="lr"),
     ]))
price.fit(x_train)
```

- **两段式**（费用类推荐）：`LGBMClassification`（是否发生费用）× `LGBMRegression`（发生后金额）= 期望费用
- 线性模型保留为 GLM 替代基准（可解释）

## ③ 迭代路径

1. log1p + LGBM 基线 → 2. 地区/车型目标编码 → 3. 车龄×险别交叉特征
4. 两段式拆分 → 5. 与现行定价表对比提升度 → 6. 分位数看尾部风险

## ④ 工具缺口

GLM/Tweedie 损失 → sklearn TweedieRegressor 或 LGBM objective 自配；暴露数(exposure)加权、充足度检验 → 精算流程外补。
