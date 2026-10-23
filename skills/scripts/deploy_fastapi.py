# -*- coding: utf-8 -*-
"""示例5：FastAPI 模型部署（已验证跑通，easymlops 0.2.4）
前置: 先运行 example_classification.py 生成 scripts/output/cls_model.pkl
依赖: pip install fastapi uvicorn
运行: python scripts/deploy_fastapi.py
测试: GET  http://127.0.0.1:8901/health
      POST http://127.0.0.1:8901/predict  body: {"features": {...9个字段...}}
"""
import os
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from easymlops import TablePipeLine

app = FastAPI(title="EasyMLOps deploy test")

# 加载训练好的 pipeline（save 存的是参数字典，用实例方法 load 恢复）
_model_path = os.path.join(os.path.dirname(__file__), "output", "cls_model.pkl")
pipeline = TablePipeLine()
pipeline.load(_model_path)


class InputData(BaseModel):
    features: dict


@app.post("/predict")
def predict(data: InputData):
    try:
        # 单条请求用 transform_single（返回 dict），避免每次构造 DataFrame
        result = pipeline.transform_single(data.features)
        return {"prediction": {k: (float(v) if hasattr(v, "item") or isinstance(v, (int, float)) else str(v))
                               for k, v in result.items()}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8901)
