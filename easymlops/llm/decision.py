# -*- coding: utf-8 -*-
"""
类 Jev 决策模型 Pipe（v0.2.12 新增）。

对接 "System One 决策模型" 生态（Jev/OneJev/Jeff/Laya）：输入 state（文本/JSON/
对话）+ typed questions（choice/score/noul），单次前向返回**校准概率**——非自回归、
无文本生成、无解析幻觉，毫秒级。适合 Agent 流水线中的路由/分流/护栏/分类判断。

后端:
    - "laya":    本地 laya 包（Router，421M/322M 非 AR 决策模型，CPU 可跑，Apache-2.0）
    - "service": Jev 兼容 HTTP API（POST {api_base}/v1/systemone），适配
                 Jeff-serve / laya serve / OneJev 等服务化部署

输出契约（每行）:
    choice 题目: {name}_pred(选中标签) + {name}_{opt}(各选项概率列)
    score  题目: {name}_pred(等级标签) + {name}_level(等级序号) + {name}_{lvl}(各等级概率)
    noul   题目: {name}_p(P(true) 校准概率)

Args:
    questions: 题目列表 [{"name": "intent", "type": "choice",
                          "instructions": "...", "criteria": {"optA": "...", ...}},
                         {"name": "urgency", "type": "score", "criteria": ["lvl0", ...]},
                         {"name": "is_fraud", "type": "noul", "instructions": "..."}]
    backend: "laya" / "service"
    state_col: 输入 state 文本列名（默认 "state"）
    api_base/api_key/model: service 后端参数
    min_confidence: laya 置信度门控（低于则 answers 带弃权标记，透传）
    **kwargs: 其他父类参数

Example:
    >>> from easymlops.llm import DecisionModelPipe
    >>> pipe = DecisionModelPipe(
    ...     questions=[{"name": "intent", "type": "choice",
    ...                 "instructions": "用户意图",
    ...                 "criteria": {"refund": "退款", "logistics": "物流", "other": "其他"}}],
    ...     backend="laya", state_col="text")
    >>> pipe.fit(df)
    >>> result = pipe.transform(df)   # 追加 intent_pred / intent_refund / ... 列
"""

import pandas as pd

from easymlops.ocr.core.pipe import OCRPipeObjectBase
from easymlops.table.core import dataframe_type, dict_type

__all__ = ["DecisionModelPipe"]

_QTYPES = ("choice", "score", "noul")


class DecisionModelPipe(OCRPipeObjectBase):
    """类 Jev 决策模型 Pipe（见模块 docstring）。"""

    def __init__(self, questions, backend="laya", state_col="state",
                 api_base=None, api_key="EMPTY", model=None, timeout=120,
                 min_confidence=None, model_path=None, **kwargs):
        super().__init__(model_name=model or "decision-model",
                         model_path=model_path, **kwargs)
        self.questions = []
        for q in questions or []:
            if q.get("type") not in _QTYPES:
                raise Exception("question type must be one of {}, got {}".format(
                    _QTYPES, q.get("type")))
            if q["type"] == "choice" and not (q.get("criteria") or {}):
                raise Exception("choice question '{}' requires criteria dict".format(q["name"]))
            if q["type"] == "score" and not (q.get("criteria") or []):
                raise Exception("score question '{}' requires criteria list".format(q["name"]))
            self.questions.append(dict(q))
        self.backend = backend
        if backend not in ("laya", "service"):
            raise Exception("backend must be 'laya' or 'service'")
        self.state_col = state_col
        self.api_base = (api_base or "").rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = int(timeout)
        self.min_confidence = min_confidence
        self._engine = None

    # ---- 引擎 ----

    def _build_engine(self):
        if self.backend == "laya":
            from laya import Router
            router = Router()
            return router
        return None   # service 后端无本地引擎

    def _get_engine(self):
        if self.backend == "laya" and self._engine is None:
            self._engine = self._build_engine()
        return self._engine

    def _laya_questions(self):
        """转 laya 的 {q_id: def} 格式。"""
        return {q["name"]: {"type": q["type"],
                            "instructions": q.get("instructions", ""),
                            **({"criteria": q["criteria"]} if q.get("criteria") else {})}
                for q in self.questions}

    def _call_laya(self, state: str) -> dict:
        router = self._get_engine()
        if self.min_confidence is not None:
            return router.predict(state, self._laya_questions(),
                                  min_confidence=self.min_confidence)
        return router.predict(state, self._laya_questions())

    def _call_service(self, state: str) -> dict:
        import requests
        payload = {"state": state, "questions": self._laya_questions()}
        headers = {"Authorization": "Bearer {}".format(self.api_key)}
        resp = requests.post(self.api_base + "/v1/systemone", json=payload,
                             headers=headers, timeout=self.timeout)
        if resp.status_code != 200:
            raise Exception("decision service error {}: {}".format(
                resp.status_code, resp.text[:200]))
        return resp.json()

    def _predict_state(self, state) -> dict:
        if state is None:
            state = ""
        if not isinstance(state, str):
            import json
            state = json.dumps(state, ensure_ascii=False)
        if self.backend == "laya":
            return self._call_laya(state)
        return self._call_service(state)


    # ---- 结果展开 ----

    @staticmethod
    def _safe_get(payload, *keys, default=None):
        cur = payload
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    def _expand(self, row_out: dict, payload: dict):
        """把 system_one payload 的 answers/probabilities 展开为输出列（容错降级）。"""
        answers = payload.get("answers", {}) or {}
        probabilities = payload.get("probabilities", {}) or {}
        for q in self.questions:
            name, qtype = q["name"], q["type"]
            answer = answers.get(name)
            probs = probabilities.get(name)
            if qtype == "choice":
                options = list(q["criteria"].keys())
                pred = None
                if isinstance(answer, dict):
                    pred = answer.get("choice", answer.get("label"))
                elif isinstance(answer, str):
                    pred = answer
                row_out["{}_pred".format(name)] = pred
                prob_map = probs if isinstance(probs, dict) else None
                for opt in options:
                    value = None
                    if prob_map and opt in prob_map:
                        value = prob_map[opt]
                    elif isinstance(probs, (list, tuple)) and opt in options:
                        value = probs[options.index(opt)]
                    row_out["{}_{}".format(name, opt)] = \
                        float(value) if isinstance(value, (int, float)) else None
            elif qtype == "score":
                levels = list(q["criteria"])
                pred = None
                if isinstance(answer, dict):
                    pred = answer.get("choice", answer.get("level", answer.get("label")))
                elif isinstance(answer, str):
                    pred = answer
                row_out["{}_pred".format(name)] = pred
                row_out["{}_level".format(name)] = \
                    levels.index(pred) if pred in levels else None
                prob_map = probs if isinstance(probs, dict) else None
                for lvl in levels:
                    value = None
                    if prob_map and lvl in prob_map:
                        value = prob_map[lvl]
                    elif isinstance(probs, (list, tuple)) and lvl in levels:
                        value = probs[levels.index(lvl)]
                    row_out["{}_{}".format(name, lvl)] = \
                        float(value) if isinstance(value, (int, float)) else None
            else:   # noul
                value = None
                if isinstance(answer, dict):
                    value = answer.get("p", answer.get("prob", answer.get("score")))
                elif isinstance(answer, (int, float)):
                    value = answer
                if value is None and isinstance(probs, (int, float)):
                    value = probs
                row_out["{}_p".format(name)] = \
                    float(value) if isinstance(value, (int, float)) else None
        return row_out

    def _output_columns(self):
        cols = []
        for q in self.questions:
            name, qtype = q["name"], q["type"]
            cols.append("{}_pred".format(name))
            if qtype == "choice":
                cols.extend("{}_{}".format(name, opt) for opt in q["criteria"].keys())
            elif qtype == "score":
                cols.append("{}_level".format(name))
                cols.extend("{}_{}".format(name, lvl) for lvl in q["criteria"])
            else:
                cols.append("{}_p".format(name))
        return cols

    # ---- 框架接口 ----

    def udf_fit(self, s: dataframe_type, **kwargs):
        if self.state_col not in s.columns:
            raise Exception("state col '{}' not in input".format(self.state_col))
        if self.backend == "laya":
            self._get_engine()   # 触发 Router 构建(模型懒加载至首次 predict)
        return self

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        if self.state_col not in s.columns:
            raise Exception("state col '{}' not in input".format(self.state_col))
        results_list = []
        for _, row in s.iterrows():
            row_out = {"state": row[self.state_col]}
            try:
                payload = self._predict_state(row[self.state_col])
                self._expand(row_out, payload)
            except Exception as exc:
                for col in self._output_columns():
                    row_out[col] = None
                row_out["{}_error".format(self.questions[0]["name"])
                         if self.questions else "error"] = str(exc)[:200]
            results_list.append(row_out)
        return pd.DataFrame(results_list)

    def udf_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        state = s.get(self.state_col, s.get("state", s.get("path")))
        if state is None:
            raise Exception("state col '{}' not in input".format(self.state_col))
        row_out = {"state": state}
        payload = self._predict_state(state)
        self._expand(row_out, payload)
        return row_out

    def transform_single(self, s: dict_type, **kwargs) -> dict_type:
        # 决策语义为"一行输入 -> 展开的多列输出", 绕过框架 extract_dict 逻辑
        return self.udf_transform_single(dict(s), **kwargs)

    def udf_get_params(self) -> dict_type:
        return {"questions": self.questions, "backend": self.backend,
                "state_col": self.state_col, "api_base": self.api_base,
                "api_key": self.api_key, "model": self.model,
                "timeout": self.timeout, "min_confidence": self.min_confidence}

    def udf_set_params(self, params: dict_type):
        self.questions = params["questions"]
        self.backend = params["backend"]
        self.state_col = params["state_col"]
        self.api_base = (params.get("api_base") or "").rstrip("/")
        self.api_key = params.get("api_key", self.api_key)
        self.model = params.get("model", self.model)
        self.timeout = params.get("timeout", self.timeout)
        self.min_confidence = params.get("min_confidence", self.min_confidence)
        self._engine = None   # load 后按恢复的配置重建
