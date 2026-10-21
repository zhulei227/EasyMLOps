# Playbook：时序预测（保费收入 / 赔付率 / 需求销量；附股价预测定位说明）

## 业务目标

用 `TSPipeLine` 统计模型做**保费收入、赔付率、需求/销量**等经营指标的时间序列预测。统计模型（ARIMA 族）仅依赖 statsmodels，实测可跑通。

## ① 数据与口径 checklist

- [ ] 序列粒度：月/周/日；长度建议 >= 2 个完整季节周期
- [ ] 预测步数 = transform 输入的行数（0.2.2 语义），滚动多步预测需自写循环
- [ ] 外生变量（如假期、利率）用 `ArimaxRegression(exog_cols=...)`
- [ ] 评估：留出最后 H 期做 holdout，MAE/MAPE 自算

## ② Pipeline 模板（实测 API）

```python
from easymlops.ts.statistical.arima import ArimaRegression, ArimaxRegression

pipe = ArimaRegression(input_cols=["premium"], p=1, d=1, q=1,
                       seasonal_p=1, seasonal_d=0, seasonal_q=1, seasonal_period=12)
pipe.fit(train_df)          # train_df: DataFrame，含 premium 列
result = pipe.transform(holdout_df)   # 追加 arima_predict 列

ax = ArimaxRegression(input_cols=["loss_ratio"], exog_cols=["holiday", "rate"],
                      p=1, d=1, q=1)
ax.fit(df)
```

- **滚动预测**（walk-forward）：外补自写循环——每次用历史 fit、预测 1 步、滚动推进
- **持久化**：TS 基类 save/load 只存参数不含拟合结果，用 pickle 整体序列化

## ③ 迭代路径

1. 朴素基线（上期值/同期值）→ 2. ARIMA 定阶（p,d,q 网格）→ 3. 季节项 seasonal_*
4. 外生变量 Arimax → 5. 多序列分型号/分地区建模 → 6. 滚动回测

## ④ 股价/收益率预测：定位与限制（重要）

可以"跑通"，但**不建议作为正式业务任务迭代优化**：

- 收益率近似白噪声，统计模型无信息优势；价格序列非平稳
- easymlops 无 walk-forward 验证、面板因子回归、回测框架、交易成本建模
- 若用于教学/练习：`ArimaRegression` 对收益率序列建模 + 自写滚动评估；回测外补 vectorbt/backtrader
- 结论：写进迭代闭环只会产生过拟合的假阳性策略，本 skill 明确不支持

## ⑤ 工具缺口

walk-forward 回测、Prophet/深度时序对比（ts 深度模型需 torch 且未验证）、多序列面板建模——外补。
