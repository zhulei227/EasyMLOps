# -*- coding: utf-8 -*-
# LLM/决策模型验收 (v0.2.12): DecisionModelPipe(mock laya + mock service) /
# LoRAFineTuner(tiny 因果 LM 真实 LoRA 训练, CPU)
import json
import os
import sys
import threading
sys.path.insert(0, "d:/datas/share/traes/easymlops-trae")

import numpy as np
import pandas as pd
from http.server import BaseHTTPRequestHandler, HTTPServer

from easymlops.llm import DecisionModelPipe, LoRAFineTuner
from easymlops import TablePipeLine

PASS, FAIL = [], []
PKL = "d:/datas/share/traes/easymlops-trae/tests/_tmp_llm.pkl"

QUESTIONS = [
    {"name": "intent", "type": "choice", "instructions": "用户意图",
     "criteria": {"refund": "退款", "logistics": "物流", "other": "其他"}},
    {"name": "urgency", "type": "score", "instructions": "紧急程度",
     "criteria": ["lvl0", "lvl1", "lvl2"]},
    {"name": "is_angry", "type": "noul", "instructions": "是否愤怒语气"},
]


def make_mock_payload(state):
    return {
        "answers": {
            "intent": {"choice": "refund"},
            "urgency": {"choice": "lvl1"},
            "is_angry": {"p": 0.87},
        },
        "probabilities": {
            "intent": {"refund": 0.72, "logistics": 0.18, "other": 0.10},
            "urgency": {"lvl0": 0.15, "lvl1": 0.60, "lvl2": 0.25},
            "is_angry": 0.87,
        },
        "confidence": 0.72,
    }


class MockDecisionEngine(object):
    """mock laya Router: 相同 predict 契约"""
    def predict(self, state, questions, **kwargs):
        assert isinstance(questions, dict) and "intent" in questions
        assert questions["intent"]["type"] == "choice"
        return make_mock_payload(state)


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print("[PASS] {} {}".format(name, detail))
    else:
        FAIL.append(name)
        print("[FAIL] {} {}".format(name, detail))


df = pd.DataFrame({
    "text": ["我要退款", "物流太慢了", "你们服务真差", "查询订单"],
    "city": ["bj", "sh", "bj", "sz"],
})


# ============ 1. DecisionModelPipe: 本地后端(mock 引擎注入) ============
print("=== 1. decision pipe local backend (mock engine) ===")
pipe = DecisionModelPipe(questions=QUESTIONS, backend="laya", state_col="text")
pipe.fit(df)                     # 框架 fit(记录列名; laya Router 轻量构建)
pipe._engine = MockDecisionEngine()
out = pipe.transform(df.copy())
expected_cols = {"state", "intent_pred", "intent_refund", "intent_logistics", "intent_other",
                 "urgency_pred", "urgency_level", "urgency_lvl0", "urgency_lvl1",
                 "urgency_lvl2", "is_angry_p"}
check("decision_output_contract", expected_cols.issubset(set(out.columns)),
      "cols={}".format(sorted(c for c in out.columns if c != "state")[:6]))
check("decision_choice_pred", (out["intent_pred"] == "refund").all())
check("decision_choice_probs", np.allclose(out["intent_refund"], 0.72) and
      np.allclose(out["intent_logistics"], 0.18) and
      np.allclose(out["intent_other"], 0.10))
check("decision_score", (out["urgency_pred"] == "lvl1").all() and
      (out["urgency_level"] == 1).all() and np.allclose(out["urgency_lvl1"], 0.60))
check("decision_noul", np.allclose(out["is_angry_p"], 0.87))

# save/load: 配置持久化 + 引擎重建(mock 重新注入)
pipe.save(PKL)
pipe2 = DecisionModelPipe(questions=QUESTIONS)
pipe2.load(PKL)
check("decision_save_load_config", pipe2.backend == "laya" and
      len(pipe2.questions) == 3)
pipe2._engine = MockDecisionEngine()
out2 = pipe2.transform(df.copy())
check("decision_save_load_output", out2["intent_pred"].equals(out["intent_pred"]))

# 单条推理
s = pipe.transform_single({"text": "我要退款"})
check("decision_single", s["intent_pred"] == "refund" and
      abs(s["is_angry_p"] - 0.87) < 1e-9)

# Pipeline 集成
pipeline = TablePipeLine()
pipeline.pipe(DecisionModelPipe(questions=QUESTIONS, backend="laya", state_col="text"))
pipeline.fit(df)
pipeline.models[-1]._engine = MockDecisionEngine()
pout = pipeline.transform(df.copy())
check("decision_pipeline", expected_cols.issubset(set(pout.columns)))

# ============ 2. service 后端(mock OpenAI/Jev 兼容服务) ============
print("=== 2. decision pipe service backend ===")
class _MockSystemOneServer(BaseHTTPRequestHandler):
    def do_POST(self):
        assert self.path == "/v1/systemone"
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])).decode("utf-8"))
        assert "state" in body and "intent" in body["questions"]
        resp = make_mock_payload(body["state"])
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(resp).encode("utf-8"))

    def log_message(self, *args):
        pass

server = HTTPServer(("127.0.0.1", 0), _MockSystemOneServer)
port = server.server_address[1]
threading.Thread(target=server.serve_forever, daemon=True).start()
spipe = DecisionModelPipe(questions=QUESTIONS, backend="service",
                          api_base="http://127.0.0.1:{}".format(port),
                          api_key="test-key", state_col="text")
spipe.fit(df)
sout = spipe.transform(df.copy())
check("service_backend", (sout["intent_pred"] == "refund").all() and
      np.allclose(sout["intent_refund"], 0.72))
ss = spipe.transform_single({"text": "查询"})
check("service_single", ss["urgency_pred"] == "lvl1")
server.shutdown()


# ============ 3. LoRAFineTuner: auto_adapt + tiny 因果 LM 真实 LoRA 训练(CPU) ============
print("=== 3. LoRAFineTuner ===")
try:
    import torch
    import transformers
    import peft
    HAS_FT_DEPS = True
except ImportError:
    HAS_FT_DEPS = False

if HAS_FT_DEPS:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    TINY_DIR = "d:/datas/share/traes/easymlops-trae/tests/_tiny_lm"
    if not os.path.isdir(TINY_DIR):
        from transformers import GPT2Config, GPT2LMHeadModel
        cfg = GPT2Config(vocab_size=256, n_positions=64, n_embd=32, n_layer=2, n_head=2)
        tiny = GPT2LMHeadModel(cfg)
        tok = AutoTokenizer.from_pretrained("gpt2") if False else None
        tiny.save_pretrained(TINY_DIR)
        # 保存一个极简 tokenizer(手写字表)
        vocab = {"<|endoftext|>": 0}
        for i in range(33, 127):
            vocab[chr(i)] = i - 32
        tok_dir = os.path.join(TINY_DIR, "tokenizer.json")
        with open(os.path.join(TINY_DIR, "vocab.json"), "w", encoding="utf-8") as f:
            json.dump(vocab, f)
        with open(os.path.join(TINY_DIR, "merges.txt"), "w", encoding="utf-8") as f:
            f.write("#version: 0.2\n")

    tuner = LoRAFineTuner("tiny-lm", local_path=TINY_DIR, lora_rank=4, lora_alpha=8,
                          device="cpu")
    info = tuner.auto_adapt()
    check("ft_auto_adapt", info["model_type"] == "gpt2" and
          any("c_attn" in t or "proj" in t for t in info["target_modules"]) and
          info["model_class"] == "AutoModelForCausalLM",
          "targets={}".format(info["target_modules"][:5]))

    samples = [
        {"prompt": "route: refund request\n", "completion": "answer: refund\n"},
        {"prompt": "route: slow delivery\n", "completion": "answer: logistics\n"},
        {"prompt": "route: where is my order\n", "completion": "answer: logistics\n"},
        {"prompt": "route: angry complaint\n", "completion": "answer: other\n"},
    ]
    OUT_DIR = "d:/datas/share/traes/easymlops-trae/tests/_lora_out"
    result = tuner.train_sft(samples, output_dir=OUT_DIR, epochs=1, lr=5e-4, batch_size=2)
    check("ft_train_sft", result["steps"] >= 2 and
          os.path.exists(os.path.join(OUT_DIR, "adapter_config.json")))
    # load_adapter 后可推理
    model = tuner.load_adapter(OUT_DIR)
    check("ft_load_adapter", model is not None)
    # merge_and_export
    merged_dir = "d:/datas/share/traes/easymlops-trae/tests/_lora_merged"
    tuner.merge_and_export(merged_dir)
    check("ft_merge_export", os.path.exists(os.path.join(merged_dir, "config.json")))
    # 清理
    import shutil
    shutil.rmtree(OUT_DIR, ignore_errors=True)
    shutil.rmtree(merged_dir, ignore_errors=True)
else:
    print("[SKIP] transformers/peft not installed")

# ============ 4. 真实 laya 冒烟(可选: 需下载 322M checkpoint) ============
print("=== 4. real laya smoke (optional) ===")
try:
    import laya as _laya
    HAS_LAYA = True
except ImportError:
    HAS_LAYA = False

if HAS_LAYA and os.environ.get("EASYMLOPS_TEST_REAL_LAYA") == "1":
    rpipe = DecisionModelPipe(questions=QUESTIONS, backend="laya", state_col="text")
    rpipe.fit(df)
    rout = rpipe.transform(df.iloc[:2].copy())
    check("real_laya_smoke", "intent_pred" in rout.columns and
          rout["intent_pred"].notna().all())
else:
    print("[SKIP] real laya smoke: set EASYMLOPS_TEST_REAL_LAYA=1 to enable "
          "(downloads 322M checkpoint)")

# ============ 汇总 ============
if os.path.exists(PKL):
    os.remove(PKL)
print("")
print("================ SUMMARY ================")
print("PASS: {}, FAIL: {}".format(len(PASS), len(FAIL)))
if FAIL:
    print("FAILED:", FAIL)
    sys.exit(1)
print("ALL TESTS PASSED")
