# -*- coding: utf-8 -*-
"""
文本特征 pipe（v0.2.5 新增，源自 Kaggle 优胜方案高频技术挖掘：TF-IDF/SVD 文本特征）。

约定:
    - 惰性导入 sklearn（TfidfVectorizer / TruncatedSVD）
    - fit 时记录文本列缺失填充值与词表，transform 追加压缩特征列
"""
import numpy as np
import pandas as pd

from easymlops.table.core import *

dataframe_type = pd.DataFrame
dict_type = dict


class TextTfidfSvd(TablePipeObjectBase):
    """
    文本列 → TF-IDF → TruncatedSVD 压缩特征（v0.2.5 新增）。

    对自由文本列（如备注、描述）生成低维稠密向量并追加为数值特征列，
    使表格管道（LGBM 等）能够消费文本信息。

    Args:
        cols: 文本列名列表（必填，至少 1 列）
        n_components: SVD 输出维度（默认 8）
        tfidf_params: 透传 TfidfVectorizer 的参数（如 {"max_features": 2000, "ngram_range": (1, 2)}）
        drop_origin_cols: 是否删除原文本列（默认 False）
        prefix: 输出列前缀（默认 f"{col}_svd_"）
        fillna_value: 文本缺失填充值

    Example:
        >>> from easymlops.table.text_features import TextTfidfSvd
        >>> pipe = TextTfidfSvd(cols=["remark"], n_components=8)
        >>> pipe.fit(train_df)
        >>> result = pipe.transform(test_df)   # 追加 remark_svd_0..7 列
    """

    def __init__(self, cols, n_components=8, tfidf_params=None, drop_origin_cols=False,
                 prefix=None, fillna_value="", **kwargs):
        super().__init__(**kwargs)
        self.cols = [cols] if isinstance(cols, str) else list(cols)
        if len(self.cols) == 0:
            raise Exception("TextTfidfSvd requires at least one text col")
        self.n_components = int(n_components)
        self.tfidf_params = tfidf_params or {}
        self.drop_origin_cols = drop_origin_cols
        self.prefix = prefix
        self.fillna_value = fillna_value
        self.tfidf_models_ = {}   # col -> TfidfVectorizer
        self.svd_models_ = {}     # col -> TruncatedSVD
        self.output_cols_ = []    # 生成的特征列名

    def _output_prefix(self, col):
        return self.prefix if self.prefix is not None else "{}_svd_".format(col)

    def udf_fit(self, s: dataframe_type, **kwargs):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        for col in self.cols:
            if col not in s.columns:
                raise Exception("col {} not in input".format(col))
            texts = s[col].fillna(self.fillna_value).astype(str)
            tfidf = TfidfVectorizer(**self.tfidf_params)
            matrix = tfidf.fit_transform(texts)
            n_components = min(self.n_components, matrix.shape[1] - 1, len(texts) - 1)
            n_components = max(1, n_components)
            svd = TruncatedSVD(n_components=n_components, random_state=42)
            svd.fit(matrix)
            self.tfidf_models_[col] = tfidf
            self.svd_models_[col] = svd
            self.output_cols_.extend(
                ["{}{}".format(self._output_prefix(col), i) for i in range(n_components)])
        return self

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        if not self.tfidf_models_:
            raise Exception("TextTfidfSvd is not fitted, call fit before transform")
        result = s.copy()
        for col in self.cols:
            if col not in result.columns:
                raise Exception("col {} not in input".format(col))
            texts = result[col].fillna(self.fillna_value).astype(str)
            matrix = self.tfidf_models_[col].transform(texts)
            reduced = self.svd_models_[col].transform(matrix)
            for i in range(reduced.shape[1]):
                result["{}{}".format(self._output_prefix(col), i)] = reduced[:, i]
        if self.drop_origin_cols:
            result = result.drop(columns=[c for c in self.cols if c in result.columns])
        return result

    def udf_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        if not self.tfidf_models_:
            raise Exception("TextTfidfSvd is not fitted, call fit before transform")
        result = dict(s)
        for col in self.cols:
            if col not in result:
                raise Exception("col {} not in input".format(col))
            text = self.fillna_value if result[col] is None else str(result[col])
            matrix = self.tfidf_models_[col].transform([text])
            reduced = self.svd_models_[col].transform(matrix)[0]
            for i in range(len(reduced)):
                result["{}{}".format(self._output_prefix(col), i)] = float(reduced[i])
        if self.drop_origin_cols:
            for col in self.cols:
                result.pop(col, None)
        return result

    def udf_get_params(self) -> dict_type:
        return {"cols": self.cols, "n_components": self.n_components,
                "tfidf_params": self.tfidf_params, "drop_origin_cols": self.drop_origin_cols,
                "prefix": self.prefix, "fillna_value": self.fillna_value,
                "tfidf_models_": self.tfidf_models_, "svd_models_": self.svd_models_,
                "output_cols_": self.output_cols_}

    def udf_set_params(self, params: dict_type):
        self.cols = params["cols"]
        self.n_components = params["n_components"]
        self.tfidf_params = params["tfidf_params"]
        self.drop_origin_cols = params["drop_origin_cols"]
        self.prefix = params["prefix"]
        self.fillna_value = params["fillna_value"]
        self.tfidf_models_ = params["tfidf_models_"]
        self.svd_models_ = params["svd_models_"]
        self.output_cols_ = params["output_cols_"]
