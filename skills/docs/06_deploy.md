# 生产部署

## 6.1 Flask 部署示例

```python
from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# 加载模型
with open("table_pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    df = pd.DataFrame([data])
    result = pipeline.transform(df)
    return jsonify({"prediction": result.tolist()})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
```

---

## 6.2 FastAPI 部署示例

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd

app = FastAPI()

# 加载模型
with open("table_pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)

class InputData(BaseModel):
    features: dict

@app.post("/predict")
def predict(data: InputData):
    try:
        df = pd.DataFrame([data.features])
        result = pipeline.transform(df)
        return {"prediction": result.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}
```

---

## 6.3 一致性测试

```python
from easymlops.table.callback import check_two_batch_transform_same

# 检查两次transform结果是否一致
result = check_two_batch_transform_same(
    pipeline, 
    cur_batch_transform, 
    pre_batch_transform, 
    check_col=None,
    check_type=True,
    check_value=True
)

print(f"一致性检查: {result}")
```

---

## 6.4 空值/极端值测试

```python
from easymlops.table.callback import check_null_value, check_extreme_value

# 检查空值
null_result = check_null_value(pipeline, x, sample=100)
print(f"空值检查结果: {null_result}")

# 检查极端值
extreme_result = check_extreme_value(pipeline, x, sample=100)
print(f"极端值检查结果: {extreme_result}")
```

---

## 6.5 数据类型检查

```python
from easymlops.table.callback import check_inverse_dtype, check_int_trans_float

# 检查数据类型反转
dtype_result = check_inverse_dtype(pipeline, x, sample=100)

# 检查int转float问题
int_float_result = check_int_trans_float(pipeline, x, sample=100)
```

---

## 6.6 日志记录

```python
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("easymlops")

# 在pipeline中使用
table = TablePipeLine()
table.set_logger(logger)
table.fit(x_train, y=y_train)
```

---

## 6.7 性能测试

```python
import time
import numpy as np

# 测试预测性能
def benchmark(pipeline, x_test, n_runs=100):
    times = []
    for _ in range(n_runs):
        start = time.time()
        _ = pipeline.transform(x_test)
        times.append(time.time() - start)
    
    return {
        "mean": np.mean(times),
        "std": np.std(times),
        "min": np.min(times),
        "max": np.max(times),
        "p95": np.percentile(times, 95)
    }

# 运行性能测试
results = benchmark(table, x_test)
print(f"性能结果: {results}")
```

---

## 6.8 批量预测

```python
# 批量预测
def batch_predict(pipeline, x_batch, batch_size=1000):
    results = []
    for i in range(0, len(x_batch), batch_size):
        batch = x_batch[i:i+batch_size]
        result = pipeline.transform(batch)
        results.append(result)
    return np.vstack(results)

# 使用
predictions = batch_predict(table, x_test_large)
```
