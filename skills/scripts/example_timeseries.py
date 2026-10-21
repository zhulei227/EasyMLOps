# -*- coding: utf-8 -*-
"""示例4：时序任务（ARIMA 统计模型）训练与预测（已验证跑通，easymlops 0.2.2）
注：0.2.2 实际签名为 (output_col, input_cols, p, d, q, seasonal_*, exog_cols)，
    文档中的 time_period/forecast_horizon 参数不存在。
运行: python scripts/example_timeseries.py
"""
import os
import pickle
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

pipe = ArimaRegression(input_cols=["value"], p=2, d=1, q=2)
pipe.fit(train)
result = pipe.transform(test)
print("[TS] 输出列:", list(result.columns))
pred = result["arima_predict"].values
actual = test["value"].values
rmse = float(np.sqrt(np.mean((pred - actual) ** 2)))
print("[TS] 预测步数:", len(pred), "| 对比实际RMSE: %.4f" % rmse)
print("[TS] 预测前3个值:", np.round(pred[:3], 2), "| 实际前3个值:", np.round(actual[:3], 2))

# 持久化：TS 基类的 save/load 只存参数不含拟合结果(results)，改用 pickle 整体序列化
model_path = os.path.join(OUT, "arima_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(pipe, f)
with open(model_path, "rb") as f:
    pipe2 = pickle.load(f)
result2 = pipe2.transform(test)
print("[TS] pickle 加载后预测一致:", bool(np.allclose(result["arima_predict"].values,
                                                   result2["arima_predict"].values)))
print("TIMESERIES PASSED")
