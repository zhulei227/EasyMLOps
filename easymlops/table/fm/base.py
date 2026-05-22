# -*- coding: utf-8 -*-
from easymlops.table.core import *
import pandas as pd
import numpy as np


pd.options.mode.copy_on_write = True


class FMBase(TablePipeObjectBase):
    """
    因子分解机基类。
    
    提供因子分解机模型的通用框架，包括FM、FFM、DeepFM等。
    
    Args:
        y: 目标变量
        cols: 用于模型训练的列
        task_type: 任务类型，"classification" 或 "regression"
        **kwargs: 其他父类参数
    """
    
    def __init__(self, y: series_type = None, cols="all", task_type="classification",
                 field_dims=None, embed_dim=8, **kwargs):
        super().__init__(**kwargs)
        self.y = copy.deepcopy(y)
        self.cols = cols if cols is not None else []
        self.task_type = task_type
        self.field_dims = field_dims
        self.embed_dim = embed_dim
        self.model = None
        self.id2label = {}
        self.label2id = {}
        self.num_class = None
        self.cols_ = None
        
        if self.y is not None and task_type == "classification":
            for idx, label in enumerate(self.y.value_counts().index):
                self.id2label[idx] = label
                self.label2id[label] = idx
            self.y = self.y.apply(lambda x: self.label2id.get(x))
            self.num_class = len(self.id2label)

    def before_fit(self, s: dataframe_type, **kwargs) -> dataframe_type:
        s = super().before_fit(s, **kwargs)
        assert self.y is not None
        if len(self.cols) == 0:
            self.cols = s.columns.tolist()
        if self.field_dims is None:
            self.field_dims = [s[col].nunique() for col in self.cols]
        return s

    def before_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        s = super().before_transform(s, **kwargs)
        return s

    def transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        s_ = self.after_transform(self.udf_transform(self.before_transform(s, **kwargs), **kwargs), **kwargs)
        return s_

    def transform_single(self, s: dict_type, **kwargs) -> dict_type:
        s_ = copy.deepcopy(s)
        s_ = self.after_transform_single(
            self.udf_transform_single(self.before_transform_single(s_, **kwargs), **kwargs), **kwargs)
        return s_

    def udf_fit(self, s, **kwargs):
        raise Exception("need to implement")

    def udf_transform(self, s, **kwargs):
        raise Exception("need to implement")

    def udf_transform_single(self, s: dict_type, **kwargs):
        raise Exception("need to implement")

    def udf_get_params(self) -> dict_type:
        return {"cols": self.cols, "task_type": self.task_type, "field_dims": self.field_dims,
                "embed_dim": self.embed_dim, "id2label": self.id2label, "label2id": self.label2id,
                "num_class": self.num_class}

    def udf_set_params(self, params: dict):
        self.cols = params["cols"]
        self.task_type = params["task_type"]
        self.field_dims = params["field_dims"]
        self.embed_dim = params["embed_dim"]
        self.id2label = params.get("id2label", {})
        self.label2id = params.get("label2id", {})
        self.num_class = params.get("num_class", None)
