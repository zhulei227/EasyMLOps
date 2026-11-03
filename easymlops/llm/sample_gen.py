# -*- coding: utf-8 -*-
"""
Jev 契约训练样本生成器（v0.2.12 新增）：把带标签的表格数据转换为决策模型微调样本
（{"state", "questions", "answers"}，即 train_decision/train_decision 的输入格式）。

从数据自动派生:
    - choice 题目: criteria 取自标签列的实际取值（可附说明模板）
    - score 题目: criteria 等级列表（按取值排序或显式指定）
    - noul  题目: 标签列布尔化（truthy→P(true)=1）

Example:
    >>> from easymlops.llm import DecisionSampleGenerator
    >>> gen = DecisionSampleGenerator(
    ...     questions=[{"name": "intent", "type": "choice", "instructions": "用户意图"}],
    ...     text_col="text")
    >>> samples = gen.generate(df, label_map={"intent": "label_col"})
    >>> train, val = gen.split(samples, val_ratio=0.2)
    >>> tuner.train_decision(train, output_dir="adapter")
"""

import random

__all__ = ["DecisionSampleGenerator"]


class DecisionSampleGenerator(object):
    """从带标签表格数据生成 Jev 契约微调样本（详见模块 docstring）。"""

    def __init__(self, questions, text_col="text", random_state=42,
                 include_instructions=True):
        self.questions = []
        for q in questions or []:
            if q.get("type") not in ("choice", "score", "noul"):
                raise Exception("question type must be choice/score/noul")
            if q["type"] == "choice" and "criteria" in q and \
                    not isinstance(q["criteria"], dict):
                raise Exception("choice criteria must be a dict {label: description}")
            self.questions.append(dict(q))
        self.text_col = text_col
        self.random_state = int(random_state)
        self.include_instructions = bool(include_instructions)

    # ---- 派生 ----

    def _derive_questions(self, df, label_map):
        """从数据派生完整 questions 定义（补全 criteria）。"""
        derived = {}
        for q in self.questions:
            name, qtype = q["name"], q["type"]
            col = label_map.get(name)
            if col is None:
                if q["type"] == "choice" and "criteria" not in q:
                    raise Exception(
                        "choice question '{}' needs criteria or a label_map column".format(name))
                derived[name] = {k: v for k, v in q.items() if k != "name"}
                continue
            values = df[col].dropna().unique().tolist()
            if qtype == "choice":
                if "criteria" in q:
                    # 显式 criteria = 封闭标签空间(数据中的未知标签在生成时跳过)
                    criteria = dict(q["criteria"])
                else:
                    criteria = {str(v): str(v) for v in sorted(values)}
                derived[name] = {"type": "choice",
                                 "instructions": q.get("instructions", name),
                                 "criteria": criteria}
            elif qtype == "score":
                criteria = q.get("criteria") or [str(v) for v in sorted(values)]
                derived[name] = {"type": "score",
                                 "instructions": q.get("instructions", name),
                                 "criteria": criteria}
            else:
                derived[name] = {"type": "noul",
                                 "instructions": q.get("instructions", name)}
        return derived

    @staticmethod
    def _to_answer(qtype, qdef, value):
        """把标签值转换为该题型的答案。"""
        if value is None or (isinstance(value, float) and value != value):
            return None
        if qtype == "choice":
            label = str(value)
            if label not in qdef["criteria"]:
                return None
            return {"choice": label}
        if qtype == "score":
            label = str(value)
            if label not in qdef["criteria"]:
                return None
            return {"choice": label}
        # noul: 布尔/0-1/yes-no
        if isinstance(value, str):
            v = value.strip().lower()
            if v in ("1", "true", "yes", "y", "是"):
                p = 1.0
            elif v in ("0", "false", "no", "n", "否"):
                p = 0.0
            else:
                return None
        else:
            p = float(value)
            p = 1.0 if p >= 0.5 else 0.0
        return {"p": p}

    # ---- 主入口 ----

    def generate(self, df, label_map):
        """
        从表格数据生成 Jev 契约样本。

        Args:
            df: 含文本列与标签列的 DataFrame
            label_map: {question_name: 标签列名}；未出现在 label_map 中的题目
                       保留其构造时给出的 criteria（生成时 answers 置空由用户补齐
                       ——或直接从数据派生）

        Returns:
            list: [{"state": ..., "questions": {...}, "answers": {...}}]
        """
        if self.text_col not in df.columns:
            raise Exception("text col '{}' not in input".format(self.text_col))
        qdefs = self._derive_questions(df, label_map or {})
        samples = []
        skipped = 0
        for _, row in df.iterrows():
            state = row[self.text_col]
            if state is None or (isinstance(state, float) and state != state):
                skipped += 1
                continue
            answers = {}
            valid = True
            for q in self.questions:
                name, qtype = q["name"], q["type"]
                col = (label_map or {}).get(name)
                if col is None:
                    answers[name] = {}   # 无标签来源: 留空占位
                    continue
                answer = self._to_answer(qtype, qdefs[name], row[col])
                if answer is None:
                    valid = False
                    break
                answers[name] = answer
            if not valid:
                skipped += 1
                continue
            samples.append({"state": str(state),
                            "questions": qdefs,
                            "answers": answers})
        self.last_skipped_ = skipped
        self.last_questions_ = qdefs
        return samples

    def split(self, samples, val_ratio=0.2):
        """随机切分 train/val（固定 random_state 可复现）。"""
        if not 0 < val_ratio < 1:
            raise Exception("val_ratio must be in (0, 1)")
        indexed = list(samples)
        rng = random.Random(self.random_state)
        rng.shuffle(indexed)
        val_size = max(1, int(len(indexed) * val_ratio)) if indexed else 0
        return indexed[val_size:], indexed[:val_size]
