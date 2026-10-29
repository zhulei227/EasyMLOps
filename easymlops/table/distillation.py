# -*- coding: utf-8 -*-
"""
知识蒸馏 pipe（v0.2.7 新增，源自 Kaggle: knowledge distillation）。

约定:
    - 教师(LGBM 等已绑定 y 的模型 pipe)对训练数据给出软目标
    - 分类: 学生(TabularMLPClassification)以 alpha*软标签 + (1-alpha)*硬标签 训练
      (软标签经 temperature 软化)，软标签交叉熵损失
    - 回归: 学生(TabularMLPRegression)直接以教师预测值为目标(MSE)
    - transform/transform_single 委托学生；save/load 持久化学生
      (教师仅 fit 时需要，不持久化)
"""
import copy

import numpy as np
import pandas as pd

from easymlops.table.core import *

dataframe_type = pd.DataFrame
dict_type = dict


class DistillationPipe(TablePipeObjectBase):
    """
    知识蒸馏 pipe（v0.2.7 新增）。

    Args:
        teacher_pipe: 已配置好(构造时绑定 y)的教师模型 pipe（如 LGBMClassification）
        student_params: 学生 TabularMLP 的构造参数
                        （如 {"hidden_dims": (128, 64), "epochs": 30}；不含 y）
        alpha: 软/硬目标混合系数（1=纯教师软标签，0=纯硬标签）
        temperature: 软标签软化温度（>1 更平滑）
        task: "classification" / "regression"
        **kwargs: 其他父类参数

    Example:
        >>> from easymlops.table.distillation import DistillationPipe
        >>> from easymlops.table.classification import LGBMClassification
        >>> distill = DistillationPipe(
        ...     teacher_pipe=LGBMClassification(y=y_train, use_faster_predictor=False),
        ...     student_params={"hidden_dims": (128, 64), "epochs": 30},
        ...     alpha=0.7, temperature=2.0)
        >>> distill.fit(train_df)
        >>> probs = distill.transform(test_df)
    """

    def __init__(self, teacher_pipe=None, student_params=None, alpha=0.5,
                 temperature=1.0, task="classification", **kwargs):
        super().__init__(**kwargs)
        self.teacher_pipe = teacher_pipe
        self.student_params = student_params or {}
        self.alpha = float(alpha)
        self.temperature = float(temperature)
        self.task = task
        self.model_ = None

    def _soften(self, probs: np.ndarray) -> np.ndarray:
        """按温度软化概率: p_i^(1/T) 归一化。"""
        if self.temperature == 1.0:
            return probs
        logits = np.log(np.clip(probs, 1e-9, None)) / self.temperature
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)

    def udf_fit(self, s: dataframe_type, **kwargs):
        if self.teacher_pipe is None:
            raise Exception("DistillationPipe requires teacher_pipe (with y bound)")
        from easymlops.table.nn import TabularMLPClassification, TabularMLPRegression

        teacher = copy.deepcopy(self.teacher_pipe)
        teacher.fit(s, **kwargs)

        if self.task == "classification":
            student = TabularMLPClassification(y=self.teacher_pipe.y, **self.student_params)
            student._learn_prep(s)
            num_x = student._num_array(s)
            cat_x = student._cat_array(s)
            hard_ids = student.y.to_numpy(dtype=np.int64)
            onehot = np.eye(student.num_class, dtype=np.float32)[hard_ids]
            probs = teacher.transform(s)
            ordered = probs[[student.id2label[i] for i in range(student.num_class)]] \
                .to_numpy(dtype=np.float32)
            soft = self._soften(ordered)
            targets = self.alpha * soft + (1.0 - self.alpha) * onehot
            student._train(num_x, cat_x, targets, student.num_class)
            # 绕过框架 fit 路径, 手动补记列信息(before_transform 依赖)
            student.input_col_names = s.columns.tolist()
            self.model_ = student
        else:
            student = TabularMLPRegression(y=pd.Series(dtype=np.float64),
                                           **self.student_params)
            teacher_preds = teacher.transform(s)
            pred_col = teacher_preds.columns[0]
            student.y = pd.Series(teacher_preds[pred_col].to_numpy(dtype=np.float64))
            student.udf_fit(s)
            student.input_col_names = s.columns.tolist()
            self.model_ = student
        return self

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        if self.model_ is None:
            raise Exception("DistillationPipe is not fitted, call fit before transform")
        return self.model_.transform(s, **kwargs)

    def udf_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        if self.model_ is None:
            raise Exception("DistillationPipe is not fitted, call fit before transform")
        return self.model_.transform_single(s, **kwargs)

    def udf_get_params(self) -> dict_type:
        # 教师不持久化(仅 fit 时需要); 学生模型随参数持久化
        return {"task": self.task, "alpha": self.alpha, "temperature": self.temperature,
                "student_params": self.student_params,
                "model_params": self.model_.get_params() if self.model_ is not None else None}

    def udf_set_params(self, params: dict_type):
        self.task = params["task"]
        self.alpha = params["alpha"]
        self.temperature = params["temperature"]
        self.student_params = params["student_params"]
        if params["model_params"] is not None:
            from easymlops.table.nn import TabularMLPClassification, TabularMLPRegression
            self.model_ = TabularMLPClassification() if self.task == "classification" \
                else TabularMLPRegression()
            self.model_.set_params(params["model_params"])
            self.model_.device = "cpu"
        else:
            self.model_ = None
