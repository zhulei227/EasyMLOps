# 模型持久化

## 5.1 保存模型

```python
# 保存整个pipeline（save 存的是参数字典）
table.save("./models/table_pipeline.pkl")

# 注意: 0.2.4 没有 get_pipe() 方法，无法单独取出某个 pipe 保存；
# 需要完整快照（含拟合结果）时用 pickle 序列化整个对象:
import pickle
with open("./models/table_pipeline_full.pkl", "wb") as f:
    pickle.dump(table, f)
```

---

## 5.2 加载模型

```python
# 加载整个pipeline（load 是实例方法，先构造再加载）
from easymlops import TablePipeLine
table = TablePipeLine()
table.load("./models/table_pipeline.pkl")

# 或从 pickle 恢复完整对象（含拟合结果，无需重新 fit）
with open("./models/table_pipeline_full.pkl", "rb") as f:
    table = pickle.load(f)
```

---

## 5.3 模型导出

```python
# ONNX/PMML 导出方法在 0.2.4 中存在但未实测验证，使用前请先确认
table.export_onnx("./models/model.onnx")
table.export_pmml("./models/model.pmml")
```

---

## 5.4 持久化注意事项

1. **序列化问题**: LLM模型等对象在序列化时可能会有问题，需要特殊处理
2. **路径问题**: 保存和加载时使用绝对路径避免路径问题
3. **版本兼容**: 不同版本的easymlops保存的模型可能不兼容

---

## 5.5 TS 预测模型持久化（0.2.4 实测）

TS 预测模型的拟合结果随 save/load 持久化，**无需再用 pickle 整体序列化**：

```python
# 统计模型: 模型对象随参数持久化
from easymlops.ts.statistical import ArimaRegression
pipe = ArimaRegression(input_cols=["value"], p=1, d=1, q=1)
pipe.fit(df)
pipe.save("arima.pkl")
loaded = ArimaRegression(input_cols=["value"], p=1, d=1, q=1)
loaded.load("arima.pkl")     # load 后可直接 transform

# torch 深度模型: state_dict 随参数持久化(TorchForecastMixin)
from easymlops.ts.deep_learning.nbeats import NBeatsRegression
nb = NBeatsRegression(input_cols=["value"], time_period=30, forecast_horizon=7, epochs=5)
nb.fit(df)
nb.save("nbeats.pkl")        # load 后无需重新 fit

# GroupForecast(分组面板): 各组模型权重全部持久化
gf.save("gf.pkl")
```

注意：批量 `transform` 是多步 forecast，与逐条 `transform_single` 语义不同；预测类 pipe 不纳入 auto_test。

```python
# 获取pipeline版本
print(table.version)

# 检查模型兼容性
table.check_compatibility("./models/old_model.pkl")
```
