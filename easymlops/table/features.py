# -*- coding: utf-8 -*-
"""
表格特征工程 pipe（v0.2.5 新增，源自 Kaggle 优胜方案高频技术挖掘）。

约定:
    - 批量 transform 为纯函数(不改变统计状态)；单条 transform_single 与批量语义一致
    - GroupAggregateFeatures/ClusterFeatures 为有状态 pipe(fit 学统计/中心)，随 save/load 持久化
"""
import numpy as np
import pandas as pd

from easymlops.table.core import *

dataframe_type = pd.DataFrame
dict_type = dict


class GroupAggregateFeatures(TablePipeObjectBase):
    """
    分组聚合特征（v0.2.5 新增）：按 key 列对数值列做 groupby 聚合并回填到每一行
    （Kaggle 表格赛第一大数据增强，如"该用户所在城市的平均收入"）。

    未见过的组回退为全样本统计值（std 例外，回退 0.0；count 回退样本数）。

    Args:
        group_cols: 分组键列，str 或 list（可为多列组合键）
        agg_cols: 参与聚合的数值列；None 表示全部数值列（自动排除 group_cols）
        stats: 聚合统计子集，可选 mean/std/min/max/median/count
        drop_key_cols: transform 后是否删除分组键列（默认保留）
        out_prefix: 输出列前缀，默认 ""（输出列名 f"{col}_{stat}"；与父类 prefix 重命名参数无关）

    Example:
        >>> from easymlops.table.features import GroupAggregateFeatures
        >>> pipe = GroupAggregateFeatures(group_cols=["city"], stats=["mean", "count"])
        >>> pipe.fit(train_df)
        >>> result = pipe.transform(test_df)   # 追加 value_mean / value_count 等列
    """

    def __init__(self, group_cols, agg_cols=None,
                 stats=("mean", "std", "min", "max", "median", "count"),
                 drop_key_cols=False, out_prefix="", **kwargs):
        super().__init__(**kwargs)
        self.group_cols = [group_cols] if isinstance(group_cols, str) else list(group_cols)
        if len(self.group_cols) == 0:
            raise Exception("group_cols must not be empty")
        self.agg_cols = agg_cols
        self.stats = list(stats)
        for stat in self.stats:
            if stat not in ("mean", "std", "min", "max", "median", "count"):
                raise Exception("unsupported stat: {} (mean/std/min/max/median/count)".format(stat))
        self.drop_key_cols = drop_key_cols
        self.out_prefix = out_prefix
        self.agg_tables_ = {}      # stat -> DataFrame(index=组键字符串, columns=agg_cols)
        self.global_defaults_ = {}  # stat -> {col: 未见组的回退值}

    @staticmethod
    def _key_series(s: dataframe_type, group_cols):
        return s[group_cols].astype(str).agg("|".join, axis=1)

    def udf_fit(self, s: dataframe_type, **kwargs):
        if self.agg_cols is None:
            self.agg_cols = [c for c in s.columns
                             if c not in self.group_cols and np.issubdtype(s[c].dtype, np.number)]
        if len(self.agg_cols) == 0:
            raise Exception("no numeric agg_cols found")
        grouped = s.groupby(self.group_cols, dropna=False)[self.agg_cols]
        self.agg_tables_ = {}
        for stat in self.stats:
            table = grouped.agg(stat)
            table.index = self._key_series(table.reset_index(), self.group_cols)
            self.agg_tables_[stat] = table
        # 未见组的回退值: 全样本统计(std→0, count→样本数)
        self.global_defaults_ = {}
        for stat in self.stats:
            if stat == "std":
                self.global_defaults_[stat] = {c: 0.0 for c in self.agg_cols}
            elif stat == "count":
                self.global_defaults_[stat] = {c: float(len(s)) for c in self.agg_cols}
            else:
                self.global_defaults_[stat] = {c: float(s[c].agg(stat)) for c in self.agg_cols}
        return self

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        if not self.agg_tables_:
            raise Exception("GroupAggregateFeatures is not fitted, call fit before transform")
        result = s.copy()
        keys = self._key_series(s, self.group_cols)
        for stat in self.stats:
            table = self.agg_tables_[stat]   # 索引=组键字符串(fit 时已转换)
            for col in self.agg_cols:
                mapped = keys.map(table[col])
                result["{}{}_{}".format(self.out_prefix, col, stat)] = \
                    mapped.fillna(self.global_defaults_[stat][col]).astype(float)
        if self.drop_key_cols:
            result = result.drop(columns=[c for c in self.group_cols if c in result.columns])
        return result

    def udf_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        if not self.agg_tables_:
            raise Exception("GroupAggregateFeatures is not fitted, call fit before transform")
        key = "|".join(str(s.get(c)) for c in self.group_cols)
        for stat in self.stats:
            table = self.agg_tables_[stat]
            for col in self.agg_cols:
                value = table[col].get(key)
                if value is None or (isinstance(value, float) and np.isnan(value)):
                    value = self.global_defaults_[stat][col]
                s["{}{}_{}".format(self.out_prefix, col, stat)] = float(value)
        if self.drop_key_cols:
            for c in self.group_cols:
                s.pop(c, None)
        return s

    def udf_get_params(self) -> dict_type:
        return {"group_cols": self.group_cols, "agg_cols": self.agg_cols, "stats": self.stats,
                "drop_key_cols": self.drop_key_cols, "out_prefix": self.out_prefix,
                "agg_tables_": self.agg_tables_, "global_defaults_": self.global_defaults_}

    def udf_set_params(self, params: dict_type):
        self.group_cols = params["group_cols"]
        self.agg_cols = params["agg_cols"]
        self.stats = params["stats"]
        self.drop_key_cols = params["drop_key_cols"]
        self.out_prefix = params["out_prefix"]
        self.agg_tables_ = params["agg_tables_"]
        self.global_defaults_ = params["global_defaults_"]


class ClusterFeatures(TablePipeObjectBase):
    """
    KMeans 聚类特征（v0.2.5 新增）：追加簇编号列与（可选）到各簇中心的距离列。

    Args:
        cols: 聚类用数值列；None 表示全部数值列（建议上游先做标准化）
        n_clusters: 簇数
        add_cluster_id: 是否追加 "{prefix}cluster_id" 列
        add_distances: 是否追加 "{prefix}dist_{i}" 到各簇中心的距离列
        out_prefix: 输出列前缀
        random_state: KMeans 随机种子
        **kwargs: 其他父类参数
    """

    def __init__(self, cols=None, n_clusters=5, add_cluster_id=True, add_distances=True,
                 out_prefix="kmeans_", random_state=42, method="kmeans", **kwargs):
        super().__init__(**kwargs)
        self.cols = cols
        self.n_clusters = int(n_clusters)
        self.add_cluster_id = add_cluster_id
        self.add_distances = add_distances
        self.out_prefix = out_prefix
        self.random_state = random_state
        self.method = method
        if method not in ("kmeans", "gmm"):
            raise Exception("method must be 'kmeans' or 'gmm'")
        self.cluster_model = None
        self.impute_values_ = {}

    def _numeric_cols(self, s: dataframe_type):
        if self.cols is None or (isinstance(self.cols, list) and len(self.cols) == 0):
            return [c for c in s.columns if np.issubdtype(s[c].dtype, np.number)]
        return list(self.cols)

    def udf_fit(self, s: dataframe_type, **kwargs):
        feature_cols = self._numeric_cols(s)
        if len(feature_cols) == 0:
            raise Exception("no numeric cols for clustering")
        self.feature_cols_ = feature_cols
        self.impute_values_ = {c: float(s[c].mean()) for c in feature_cols}
        x = s[feature_cols].fillna(self.impute_values_).to_numpy(dtype=float)
        if self.method == "kmeans":
            from sklearn.cluster import KMeans
            self.cluster_model = KMeans(n_clusters=self.n_clusters,
                                        random_state=self.random_state, n_init=10)
        else:
            from sklearn.mixture import GaussianMixture
            self.cluster_model = GaussianMixture(n_components=self.n_clusters,
                                                 random_state=self.random_state)
        self.cluster_model.fit(x)
        return self

    def _features(self, x: np.ndarray):
        output = {}
        if self.add_cluster_id:
            output["{}cluster_id".format(self.out_prefix)] = self.cluster_model.predict(x)
        if self.add_distances:
            if self.method == "kmeans":
                distances = self.cluster_model.transform(x)
            else:
                # GMM: 负加权对数概率作为"距离"代理(值越小越接近该簇)
                distances = -self.cluster_model._estimate_weighted_log_prob(x)
            for i in range(self.n_clusters):
                output["{}dist_{}".format(self.out_prefix, i)] = distances[:, i]
        return output

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        if self.cluster_model is None:
            raise Exception("ClusterFeatures is not fitted, call fit before transform")
        x = s[self.feature_cols_].fillna(self.impute_values_).to_numpy(dtype=float)
        features = self._features(x)
        result = s.copy()
        for name, values in features.items():
            result[name] = values
        return result

    def udf_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        if self.cluster_model is None:
            raise Exception("ClusterFeatures is not fitted, call fit before transform")
        x = np.array([[float(s.get(c, self.impute_values_.get(c, 0.0))) for c in self.feature_cols_]])
        features = self._features(x)
        for name, values in features.items():
            s[name] = values[0] if len(np.shape(values[0])) == 0 else values[0][0]
        return s

    def udf_get_params(self) -> dict_type:
        return {"cols": self.cols, "n_clusters": self.n_clusters,
                "add_cluster_id": self.add_cluster_id, "add_distances": self.add_distances,
                "out_prefix": self.out_prefix, "random_state": self.random_state,
                "method": self.method,
                "cluster_model": self.cluster_model, "impute_values_": self.impute_values_,
                "feature_cols_": getattr(self, "feature_cols_", None)}

    def udf_set_params(self, params: dict_type):
        self.cols = params["cols"]
        self.n_clusters = params["n_clusters"]
        self.add_cluster_id = params["add_cluster_id"]
        self.add_distances = params["add_distances"]
        self.out_prefix = params["out_prefix"]
        self.random_state = params["random_state"]
        self.method = params["method"]
        self.cluster_model = params["cluster_model"]
        self.impute_values_ = params["impute_values_"]
        self.feature_cols_ = params["feature_cols_"]


class QuantileClip(TablePipeObjectBase):
    """
    分位数截断 Winsorize（v0.2.5 新增）：fit 时按训练集分位数确定上下界并裁剪，
    抗异常值且无需手工指定边界（对比 Clip 需手写 min/max）。

    Args:
        cols: 要操作的列；None 表示全部数值列
        lower: 下分位数（0~1）
        upper: 上分位数（0~1）

    Example:
        >>> from easymlops.table.features import QuantileClip
        >>> pipe = QuantileClip(cols=["income"], lower=0.01, upper=0.99)
    """

    def __init__(self, cols=None, lower=0.01, upper=0.99, **kwargs):
        super().__init__(**kwargs)
        self.cols = cols
        self.lower = lower
        self.upper = upper
        if not (0 <= self.lower < self.upper <= 1):
            raise Exception("require 0 <= lower < upper <= 1")
        self.quantile_detail_ = {}

    def _numeric_cols(self, s: dataframe_type):
        if self.cols is None or (isinstance(self.cols, list) and len(self.cols) == 0):
            return [c for c in s.columns if np.issubdtype(s[c].dtype, np.number)]
        return list(self.cols)

    def udf_fit(self, s: dataframe_type, **kwargs):
        self.quantile_detail_ = {}
        for col in self._numeric_cols(s):
            values = s[col].dropna()
            if len(values) == 0:
                continue
            self.quantile_detail_[col] = (float(values.quantile(self.lower)),
                                          float(values.quantile(self.upper)))
        return self

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        result = s.copy()
        for col, (lo, hi) in self.quantile_detail_.items():
            if col in result.columns:
                result[col] = result[col].clip(lo, hi)
        return result

    def udf_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        for col, (lo, hi) in self.quantile_detail_.items():
            if col in s and s[col] is not None and s[col] == s[col]:
                s[col] = min(max(s[col], lo), hi)
        return s

    def udf_get_params(self) -> dict_type:
        return {"cols": self.cols, "lower": self.lower, "upper": self.upper,
                "quantile_detail_": self.quantile_detail_}

    def udf_set_params(self, params: dict_type):
        self.cols = params["cols"]
        self.lower = params["lower"]
        self.upper = params["upper"]
        self.quantile_detail_ = params["quantile_detail_"]


class LogTransform(TablePipeObjectBase):
    """
    log1p 变换（v0.2.5 新增）：处理右偏分布；缺失值透传，x <= -1 置为 nan。

    Args:
        cols: 要操作的列；None 表示全部数值列
    """

    def __init__(self, cols=None, **kwargs):
        super().__init__(**kwargs)
        self.cols = cols

    def _numeric_cols(self, s: dataframe_type):
        if self.cols is None or (isinstance(self.cols, list) and len(self.cols) == 0):
            return [c for c in s.columns if np.issubdtype(s[c].dtype, np.number)]
        return list(self.cols)

    def udf_fit(self, s: dataframe_type, **kwargs):
        return self

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        result = s.copy()
        for col in self._numeric_cols(s):
            result[col] = result[col].apply(
                lambda v: np.log1p(v) if v is not None and v == v and v > -1 else np.nan)
        return result

    def udf_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        for col in self._numeric_cols(pd.DataFrame([s])):
            if col in s:
                v = s[col]
                s[col] = float(np.log1p(v)) if (v is not None and v == v and v > -1) else np.nan
        return s


class SqrtTransform(TablePipeObjectBase):
    """
    平方根变换（v0.2.5 新增）：适用于计数/右偏特征；缺失值透传，负值置为 nan。

    Args:
        cols: 要操作的列；None 表示全部数值列
    """

    def __init__(self, cols=None, **kwargs):
        super().__init__(**kwargs)
        self.cols = cols

    def _numeric_cols(self, s: dataframe_type):
        if self.cols is None or (isinstance(self.cols, list) and len(self.cols) == 0):
            return [c for c in s.columns if np.issubdtype(s[c].dtype, np.number)]
        return list(self.cols)

    def udf_fit(self, s: dataframe_type, **kwargs):
        return self

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        result = s.copy()
        for col in self._numeric_cols(s):
            result[col] = result[col].apply(
                lambda v: np.sqrt(v) if v is not None and v == v and v >= 0 else np.nan)
        return result

    def udf_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        for col in self._numeric_cols(pd.DataFrame([s])):
            if col in s:
                v = s[col]
                s[col] = float(np.sqrt(v)) if (v is not None and v == v and v >= 0) else np.nan
        return s


class AnomalyScoreFeatures(TablePipeObjectBase):
    """
    异常分数特征（v0.2.7 新增，源自 Kaggle: anomaly detection features）：
    用 IsolationForest 计算每行的异常分数并追加为特征列
    （分数越大越异常；孤立点往往携带额外信号）。

    Args:
        cols: 参与检测的数值列；None 表示全部数值列
        n_estimators: IsolationForest 树数量
        contamination: 透传 IsolationForest
        out_prefix: 输出列前缀（默认 "anomaly_"）
        random_state: 随机种子

    Example:
        >>> from easymlops.table.features import AnomalyScoreFeatures
        >>> pipe = AnomalyScoreFeatures(cols=["income", "value"])
        >>> pipe.fit(train_df)
        >>> result = pipe.transform(test_df)   # 追加 anomaly_score 列
    """

    def __init__(self, cols=None, n_estimators=100, contamination="auto",
                 out_prefix="anomaly_", random_state=42, **kwargs):
        super().__init__(**kwargs)
        self.cols = cols
        self.n_estimators = int(n_estimators)
        self.contamination = contamination
        self.out_prefix = out_prefix
        self.random_state = random_state
        self.iforest_model = None
        self.impute_values_ = {}

    def _numeric_cols(self, s: dataframe_type):
        if self.cols is None or (isinstance(self.cols, list) and len(self.cols) == 0):
            return [c for c in s.columns if np.issubdtype(s[c].dtype, np.number)]
        return list(self.cols)

    def udf_fit(self, s: dataframe_type, **kwargs):
        from sklearn.ensemble import IsolationForest
        feature_cols = self._numeric_cols(s)
        if len(feature_cols) == 0:
            raise Exception("no numeric cols for IsolationForest")
        self.feature_cols_ = feature_cols
        self.impute_values_ = {c: float(s[c].mean()) for c in feature_cols}
        x = s[feature_cols].fillna(self.impute_values_).to_numpy(dtype=float)
        self.iforest_model = IsolationForest(n_estimators=self.n_estimators,
                                             contamination=self.contamination,
                                             random_state=self.random_state)
        self.iforest_model.fit(x)
        return self

    def _score(self, x: np.ndarray) -> np.ndarray:
        # decision_function 越大越正常, 取负使分数越大越异常
        return -self.iforest_model.decision_function(x)

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        if self.iforest_model is None:
            raise Exception("AnomalyScoreFeatures is not fitted, call fit before transform")
        x = s[self.feature_cols_].fillna(self.impute_values_).to_numpy(dtype=float)
        result = s.copy()
        result["{}score".format(self.out_prefix)] = self._score(x)
        return result

    def udf_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        if self.iforest_model is None:
            raise Exception("AnomalyScoreFeatures is not fitted, call fit before transform")
        x = np.array([[float(s.get(c, self.impute_values_.get(c, 0.0))) for c in self.feature_cols_]])
        s["{}score".format(self.out_prefix)] = float(self._score(x)[0])
        return s

    def udf_get_params(self) -> dict_type:
        return {"cols": self.cols, "n_estimators": self.n_estimators,
                "contamination": self.contamination, "out_prefix": self.out_prefix,
                "random_state": self.random_state,
                "iforest_model": self.iforest_model, "impute_values_": self.impute_values_,
                "feature_cols_": getattr(self, "feature_cols_", None)}

    def udf_set_params(self, params: dict_type):
        self.cols = params["cols"]
        self.n_estimators = params["n_estimators"]
        self.contamination = params["contamination"]
        self.out_prefix = params["out_prefix"]
        self.random_state = params["random_state"]
        self.iforest_model = params["iforest_model"]
        self.impute_values_ = params["impute_values_"]
        self.feature_cols_ = params["feature_cols_"]
