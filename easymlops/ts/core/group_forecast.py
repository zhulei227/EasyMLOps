# -*- coding: utf-8 -*-
"""
分组预测包装器: 把任意单序列预测 pipe 扩展为分组(面板)版本。

设计:
    - 每组独立 deepcopy 一份模板模型并 fit, 组间完全隔离(等价于逐组手工建模)
    - 批量 transform 按组分发, 各组输出按原始行位置写回(行数守恒)
    - transform_single 按记录的组键分发到对应模型
    - 模板模型须为"非分组"预测 pipe(group_cols=None), 即
      ArimaRegression 家族 / GarchRegression / 全部 torch 深度模型均可
    - get_params 持久化模板与各组模型(含权重), load 后无需重新 fit
"""
import copy

import pandas as pd

from easymlops.ts.core.group_pipe import GroupTSBase

dataframe_type = pd.DataFrame
dict_type = dict


class GroupForecast(GroupTSBase):
    """
    分组预测包装器(每组一份独立模型)。

    Args:
        model: 已配置好超参的预测 pipe 实例(模板, 须为非分组模型)
        group_cols: 分组列名, None/[] = 全局单序列(此时等价于直接使用模板模型)
        time_col: 时间列名, 组内按其排序后送入模型
        sort_rows: 是否物理重排输出(默认 False, 保持原行序)
    """

    def __init__(self, model=None, group_cols=None, time_col=None,
                 sort_rows=False, **kwargs):
        super().__init__(group_cols=group_cols, time_col=time_col,
                         sort_rows=sort_rows, **kwargs)
        self.model = model
        self.group_models_ = {}   # 组键 -> 拟合好的模型副本, 随 save/load 持久化

    def _template_class(self):
        if self.model is None:
            raise Exception("GroupForecast requires a template model, got None")
        return self.model.__class__

    def udf_fit(self, s: dataframe_type, **kwargs):
        self._template_class()   # 前置校验
        self.group_models_ = {}
        for key, view in self._iter_group_views(s):
            m = copy.deepcopy(self.model)
            m.fit(view, **kwargs)
            self.group_models_[key] = m
        self._warn_if_too_many_groups(len(self.group_models_))
        return self

    def _model_for(self, key):
        m = self.group_models_.get(key)
        if m is None:
            raise Exception("group {} not fitted, call fit before transform".format(key))
        return m

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        result = s.copy()
        for key, view in self._iter_group_views(s):
            out = self._model_for(key).transform(view, **kwargs)
            # 只写回模型新增的输出列(输入列保持原样)
            for col in out.columns:
                if col not in view.columns:
                    result.loc[view.index, col] = out[col].to_numpy()
        return self._finalize(result)

    def udf_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        return self._model_for(self._key_from_record(s)).transform_single(s, **kwargs)

    def udf_get_params(self) -> dict_type:
        return {"model_class": self.model.__class__ if self.model is not None else None,
                "model_params": self.model.get_params() if self.model is not None else None,
                "group_models_params": {key: m.get_params()
                                        for key, m in self.group_models_.items()}}

    def udf_set_params(self, params: dict_type):
        model_class = params["model_class"]
        if model_class is not None:
            self.model = model_class()
            self.model.set_params(params["model_params"])
        else:
            self.model = None
        self.group_models_ = {}
        for key, model_params in params["group_models_params"].items():
            m = model_class()
            m.set_params(model_params)
            self.group_models_[key] = m
