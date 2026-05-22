# 模型持久化

## 5.1 保存模型

```python
# 保存整个pipeline
table.save("./models/table_pipeline.pkl")

# 单独保存某个pipe
table.get_pipe("LGBMClassification").save("./models/lgbm.pkl")

# 保存为JSON格式
table.save("./models/table_pipeline.json")
```

---

## 5.2 加载模型

```python
# 加载整个pipeline
table = TablePipeLine.load("./models/table_pipeline.pkl")

# 加载单个模型
lgbm = LGBMClassification.load("./models/lgbm.pkl")

# 加载JSON格式
table = TablePipeLine.load("./models/table_pipeline.json")
```

---

## 5.3 模型导出

```python
# 导出为ONNX格式 (如果支持)
table.export_onnx("./models/model.onnx")

# 导出为PMML格式
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
