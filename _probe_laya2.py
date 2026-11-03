# -*- coding: utf-8 -*-
"""真实 laya 冒烟调试: 打印完整 payload 与异常"""
import json
import os
import sys
import traceback
sys.path.insert(0, r"d:/datas/share/traes/easymlops-trae")

import pandas as pd
from easymlops.llm import DecisionModelPipe

QUESTIONS = [
    {"name": "intent", "type": "choice", "instructions": "user intent",
     "criteria": {"refund": "refund request", "logistics": "delivery issue", "other": "other"}},
]

df = pd.DataFrame({"text": ["I want my money back", "where is my package"]})

pipe = DecisionModelPipe(questions=QUESTIONS, backend="laya", state_col="text")
pipe.fit(df)
try:
    payload = pipe._call_laya("I want my money back")
    print("=== payload keys:", sorted(payload.keys()))
    print("=== answers:", json.dumps(payload.get("answers"), ensure_ascii=False)[:500])
    print("=== probabilities:", json.dumps(payload.get("probabilities"), ensure_ascii=False)[:500])
    out = pipe.transform(df.copy())
    print("=== transform cols:", list(out.columns))
    print(out.head(2).to_string())
except Exception:
    traceback.print_exc()
