# -*- coding: utf-8 -*-
"""示例4：时序任务（ARIMA 统计模型）训练与预测（已验证跑通，easymlops 0.2.4）
注：真实签名为 (output_col, input_cols, p, d, q, seasonal_*, exog_cols,
    group_cols, time_col)；文档中的 time_period/forecast_horizon 参数不存在。
    0.2.4 起支持 group_cols 面板分组，save/load 直接持久化拟合结果。
运行: python scripts/example_timeseries.py
"""
import os
import numpy as np
import pandas as pd
from easymlops.ts.statistical.arima import ArimaRegression

OUT = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT, exist_ok=True)

np.random.seed(42)
n = 200
df = pd.DataFrame({"value": np.cumsum(np.random.randn(n)) + 100})
# 保存样例数据便于单独查看
df.to_csv(os.path.join(OUT, "sample_timeseries.csv"), index=False)

train, test = df[:170], df[170:]

# ---- 单序列 ARIMA ----
pipe = ArimaRegression(input_cols=["value"], p=2, d=1, q=2)
pipe.fit(train)
result = pipe.transform(test)
print("[TS] 输出列:", list(result.columns))
pred = result["arima_predict"].values
actual = test["value"].values
rmse = float(np.sqrt(np.mean((pred - actual) ** 2)))
print("[TS] 预测步数:", len(pred), "| 对比实际RMSE: %.4f" % rmse)
print("[TS] 预测前3个值:", np.round(pred[:3], 2), "| 实际前3个值:", np.round(actual[:3], 2))

# ---- 持久化：0.2.4 起 save/load 直接持久化拟合结果（模型对象），无需 pickle ----
model_path = os.path.join(OUT, "arima_model.pkl")
pipe.save(model_path)
pipe2 = ArimaRegression(input_cols=["value"], p=2, d=1, q=2)
pipe2.load(model_path)
result2 = pipe2.transform(test)
print("[TS] save/load 后预测一致:", bool(np.allclose(result["arima_predict"].values,
                                                     result2["arima_predict"].values)))
os.remove(model_path)

# ---- 分组面板：每个实体独立拟合一个模型（0.2.4 起） ----
panel = pd.DataFrame({
    "group": ["A"] * n + ["B"] * n,
    "value": np.concatenate([np.cumsum(np.random.randn(n)) + 50,
                             np.cumsum(np.random.randn(n)) + 120]),
})
gp = ArimaRegression(input_cols=["value"], p=1, d=1, q=1,
                     group_cols=["group"], time_col=None)
gp.fit(panel)
test_panel = pd.concat([panel[panel["group"] == "A"].tail(5),
                        panel[panel["group"] == "B"].tail(5)])
gout = gp.transform(test_panel)         # 每组独立 forecast
print("[TS] 分组预测行数:", len(gout), "| 各组预测互不相同:",
      gout.groupby("group")["arima_predict"].first().nunique() == 2)
print("TIMESERIES PASSED")
