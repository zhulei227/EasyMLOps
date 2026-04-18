# Pipeline 操作

## 4.1 链式调用

```python
table = TablePipeLine()
table.pipe(FixInput())\
     .pipe(TransToCategory(cols=["Sex"]))\
     .pipe(LabelEncoding())\
     .pipe(LGBMClassification(label="Survived"))
```

---

## 4.2 减法操作 (跳过某个pipe)

```python
# 从已有pipeline中移除某个pipe
table = TablePipeLine() - FillNa() - LabelEncoding()
```

---

## 4.3 运行到指定位置

```python
# 训练到某个pipe
table.fit(x_train, stop_at="LGBMClassification")

# 获取中间结果
intermediate_result = table.transform(x_test, start_from="FixInput", stop_at="LabelEncoding")
```

---

## 4.4 获取特定pipe

```python
# 获取指定位置的pipe
lgbm_pipe = table.get_pipe("LGBMClassification")

# 获取所有pipe名称
print(table.get_pipe_names())
```

---

## 4.5 切片式调用

```python
# 对数据的指定行进行transform
result = table.transform(x_test[start:end])
```

---

## 4.6 Pipeline 组合

```python
# 组合多个pipeline
from easymlops import TablePipeLine

pipeline1 = TablePipeLine().pipe(FixInput()).pipe(FillNa())
pipeline2 = TablePipeLine().pipe(LabelEncoding()).pipe(LGBMClassification())

# 组合
combined = pipeline1 + pipeline2

# 或者
combined = TablePipeLine()
combined.pipes = pipeline1.pipes + pipeline2.pipes
```

---

## 4.7 Pipeline 分拆

```python
# 获取指定范围的pipe
partial_pipeline = table[1:3]  # 获取第1到第2个pipe

# 获取单个pipe
single_pipe = table[0]  # 获取第0个pipe
```

---

## 4.8 回调检查

```python
from easymlops.table.callback import *

# 形状检查
check_shape(pipeline, batch_transform, single_transform)

# 列检查
check_columns(pipeline, batch_transform, single_transform)

# 数据类型检查
check_data_type(pipeline, batch_transform, single_transform)

# 数据一致性检查
check_data_same(pipeline, batch_transform, single_transform)

# Pipeline完整性检查
check_transform_function_pipeline(pipeline, x, sample=1000)
```
