# 模型持久化

## 5.1 保存模型

```python
# 保存整个pipeline（save 存的是参数字典）
table.save("./models/table_pipeline.pkl")

# 注意: 0.2.2 没有 get_pipe() 方法，无法单独取出某个 pipe 保存；
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
# ONNX/PMML 导出方法在 0.2.2 中存在但未实测验证，使用前请先确认
table.export_onnx("./models/model.onnx")
table.export_pmml("./models/model.pmml")
```

---

## 5.4 持久化注意事项

1. **序列化问题**: LLM模型等对象在序列化时可能会有问题，需要特殊处理
2. **路径问题**: 保存和加载时使用绝对路径避免路径问题
3. **版本兼容**: 不同版本的easymlops保存的模型可能不兼容

```python
# 获取pipeline版本
print(table.version)

# 检查模型兼容性
table.check_compatibility("./models/old_model.pkl")
```
