# Pipeline 操作

## 4.1 链式调用

```python
table = TablePipeLine()
table.pipe(FixInput())\
     .pipe(TransToCategory(cols=["Sex"]))\
     .pipe(LabelEncoding())\
     .pipe(LGBMClassification(y=y_train))
```

---

## 4.2 减法操作 (跳过某个pipe)

```python
# 从已有pipeline中移除某个pipe
# 注意: 0.2.4 实测减法运算符可用，但会破坏原 pipeline 内部状态
# （input_col_names 被清空），且减法产物 transform 亦有缺陷，慎用
table = TablePipeLine() - FillNa() - LabelEncoding()
```

---

## 4.3 运行到指定位置

```python
# 0.2.4 提供 stop_at/start_from 参数，但按 pipe 名称分段在实测中可能不生效，
# 使用前请先用小样本自行验证中间结果是否符合预期
table.fit(x_train, stop_at="LGBMClassification")
intermediate_result = table.transform(x_test, start_from="FixInput", stop_at="LabelEncoding")
```

---

## 4.4 获取特定pipe

```python
# 注意: 0.2.4 的 TablePipeLine 没有 get_pipe()/get_pipe_names() 方法（旧版 API）；
# pipe 顺序在构造时确定，调试时建议对 pipeline.transform 的输出列做断言来定位问题
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

# 注意: 0.2.4 实测 TablePipeLine 不支持 + 运算符（TypeError），也不存在 .pipes 属性；
# 组合管道请手动依次调用 .pipe()；减法 - 可用但会破坏原 pipeline 内部状态（见 4.2）
```

---

## 4.7 Pipeline 分拆

```python
# 切片/索引访问在 0.2.4 未验证可用，使用前请先自行确认
partial_pipeline = table[1:3]  # 获取第1到第2个pipe
single_pipe = table[0]         # 获取第0个pipe
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
