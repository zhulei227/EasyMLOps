from easymlops.table.core import *
from ..encoding import WOEEncoding
from ..preprocessing import FillNa


class FilterBase(TablePipeObjectBase):
    """
    过滤式特征选择的基础类
    """

    def __init__(self, cols="all", as_null_values=None, min_threshold=None, max_threshold=None,
                 skip_check_transform_type=True, skip_check_transform_value=True,
                 **kwargs):
        """

        :param cols: 用于特征选择的cols
        :param as_null_values: 等视为Null的值
        :param min_threshold: 最小阈值,None表示跳过
        :param max_threshold: 最大阈值,None表示跳过
        :param skip_check_transform_type:跳过类型检测
        :param skip_check_transform_value:跳过数值检测
        :param kwargs:
        """
        super().__init__(skip_check_transform_value=skip_check_transform_value,
                         skip_check_transform_type=skip_check_transform_type, **kwargs)
        self.cols = cols
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        if self.min_threshold is None:
            self.min_threshold = np.finfo(np.float64).min
        if self.max_threshold is None:
            self.max_threshold = np.finfo(np.float64).max
        self.as_null_values = as_null_values
        if self.as_null_values is None:
            self.as_null_values = GlobalNullValues
        # filter_value_dist分布
        self.filter_value_dist = dict()
        self.selected_cols = None

    def get_filter_value_dist(self):
        return self.filter_value_dist

    def show_detail(self):
        return pd.DataFrame([self.filter_value_dist])

    def before_fit(self, s: dataframe_type, **kwargs) -> dataframe_type:
        s = super().before_fit(s, **kwargs)
        if str(self.cols).lower() in ["none", "all", "null", "nan"]:
            self.cols = self.input_col_names
        assert type(self.cols) == list and type(self.cols[0]) == str
        return s

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        return s[self.selected_cols]

    def udf_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        return self.extract_dict(s, self.selected_cols)

    def udf_get_params(self):
        return {"cols": self.cols, "min_threshold": self.min_threshold,
                "max_threshold": self.max_threshold, "as_null_values": self.as_null_values,
                "filter_value_dist": self.filter_value_dist, "selected_cols": self.selected_cols}

    def udf_set_params(self, params: dict_type):
        self.cols = params["cols"]
        self.min_threshold = params["min_threshold"]
        self.max_threshold = params["max_threshold"]
        self.as_null_values = params["as_null_values"]
        self.filter_value_dist = params["filter_value_dist"]
        self.selected_cols = params["selected_cols"]


class MissRateFilter(FilterBase):
    """
    缺失率过滤
    """

    def __init__(self, max_threshold=0.95, min_threshold=None, **kwargs):
        super().__init__(max_threshold=max_threshold, min_threshold=min_threshold, **kwargs)

    def udf_fit(self, s: dataframe_type, **kwargs):
        self.filter_value_dist = dict()
        for col in self.cols:
            hint_null_index = self.get_match_values_index(s[col], self.as_null_values)
            miss_rate = np.sum(hint_null_index) / len(s)
            self.filter_value_dist[col] = miss_rate
        self.selected_cols = []
        for col in s.columns:
            miss_rate = self.filter_value_dist.get(col)
            if miss_rate is None or col not in self.cols:
                self.selected_cols.append(col)
            else:
                if self.min_threshold <= miss_rate <= self.max_threshold:
                    self.selected_cols.append(col)
        return self

    def udf_get_params(self):
        return {}


class VarianceFilter(FilterBase):
    """
    方差值过滤
    """

    def __init__(self, max_threshold=None, min_threshold=1e-3, **kwargs):
        super().__init__(max_threshold=max_threshold, min_threshold=min_threshold, **kwargs)

    def udf_fit(self, s: dataframe_type, **kwargs):
        self.filter_value_dist = s[self.cols].var().to_dict()
        self.selected_cols = []
        for col in s.columns:
            miss_rate = self.filter_value_dist.get(col)
            if miss_rate is None or col not in self.cols:
                self.selected_cols.append(col)
            else:
                if self.min_threshold <= miss_rate <= self.max_threshold:
                    self.selected_cols.append(col)
        return self

    def udf_get_params(self):
        return {}


class PersonCorrFilter(FilterBase):
    """
    person相关性绝对值过滤
    """

    def __init__(self, y=None, max_threshold=None, min_threshold=0.01, **kwargs):
        super().__init__(max_threshold=max_threshold, min_threshold=min_threshold, **kwargs)
        self.y = y

    def udf_fit(self, s: dataframe_type, **kwargs):
        assert self.y is not None and len(self.y) == len(s)
        self.filter_value_dist = s[self.cols].corrwith(self.y).to_dict()
        self.selected_cols = []
        for col in s.columns:
            miss_rate = self.filter_value_dist.get(col)
            if miss_rate is None or col not in self.cols:
                self.selected_cols.append(col)
            else:
                if self.min_threshold <= abs(miss_rate) <= self.max_threshold:
                    self.selected_cols.append(col)
        return self

    def udf_get_params(self):
        return {}


class Chi2Filter(FilterBase):
    """
    卡方过滤
    """

    def __init__(self, y=None, max_threshold=None, min_threshold=0.01, **kwargs):
        super().__init__(max_threshold=max_threshold, min_threshold=min_threshold, **kwargs)
        self.y = y

    def udf_fit(self, s: dataframe_type, **kwargs):
        from sklearn.feature_selection import chi2
        assert self.y is not None and len(self.y) == len(s)
        self.filter_value_dist = dict()
        s_ = copy.deepcopy(s[self.cols]).values
        s_ = np.where(s_ < 0, 0, s_)
        for idx, filter_value in enumerate(chi2(s_, self.y)[0]):
            self.filter_value_dist[self.cols[idx]] = filter_value
        del self.y
        self.selected_cols = []
        for col in s.columns:
            miss_rate = self.filter_value_dist.get(col)
            if miss_rate is None or col not in self.cols:
                self.selected_cols.append(col)
            else:
                if self.min_threshold <= abs(miss_rate) <= self.max_threshold:
                    self.selected_cols.append(col)
        return self

    def udf_get_params(self):
        return {}


class PValueFilter(FilterBase):
    """
    P值过滤
    """

    def __init__(self, y=None, max_threshold=None, min_threshold=None, **kwargs):
        super().__init__(max_threshold=max_threshold, min_threshold=min_threshold, **kwargs)
        self.y = y

    def udf_fit(self, s: dataframe_type, **kwargs):
        from sklearn.feature_selection import chi2
        assert self.y is not None and len(self.y) == len(s)
        self.filter_value_dist = dict()
        for idx, filter_value in enumerate(chi2(s[self.cols], self.y)[1]):
            self.filter_value_dist[self.cols[idx]] = filter_value
        del self.y
        self.selected_cols = []
        for col in s.columns:
            miss_rate = self.filter_value_dist.get(col)
            if miss_rate is None or col not in self.cols:
                self.selected_cols.append(col)
            else:
                if self.min_threshold <= abs(miss_rate) <= self.max_threshold:
                    self.selected_cols.append(col)
        return self

    def udf_get_params(self):
        return {}


class MutualInfoFilter(FilterBase):
    """
    互信息过滤
    """

    def __init__(self, y=None, max_threshold=None, min_threshold=None, **kwargs):
        super().__init__(max_threshold=max_threshold, min_threshold=min_threshold, **kwargs)
        self.y = y

    def udf_fit(self, s: dataframe_type, **kwargs):
        from sklearn.feature_selection import mutual_info_classif
        assert self.y is not None and len(self.y) == len(s)
        self.filter_value_dist = dict()
        for idx, filter_value in enumerate(mutual_info_classif(s[self.cols], self.y)):
            self.filter_value_dist[self.cols[idx]] = filter_value
        del self.y
        self.selected_cols = []
        for col in s.columns:
            miss_rate = self.filter_value_dist.get(col)
            if miss_rate is None or col not in self.cols:
                self.selected_cols.append(col)
            else:
                if self.min_threshold <= abs(miss_rate) <= self.max_threshold:
                    self.selected_cols.append(col)
        return self

    def udf_get_params(self):
        return {}


class IVFilter(FilterBase):
    """
    IV过滤
    """

    def __init__(self, y=None, max_threshold=None, min_threshold=0.02, **kwargs):
        super().__init__(max_threshold=max_threshold, min_threshold=min_threshold, **kwargs)
        self.y = y

    def udf_fit(self, s: dataframe_type, **kwargs):
        assert self.y is not None and len(self.y) == len(s)
        self.filter_value_dist = \
            WOEEncoding(y=self.y, cols=self.cols).fit(s, **kwargs).show_detail()[["col", "iv"]].groupby("col").agg(
                {"iv": "sum"}).to_dict()["iv"]
        self.selected_cols = []
        for col in s.columns:
            miss_rate = self.filter_value_dist.get(col)
            if miss_rate is None or col not in self.cols:
                self.selected_cols.append(col)
            else:
                if self.min_threshold <= abs(miss_rate) <= self.max_threshold:
                    self.selected_cols.append(col)
        return self

    def udf_get_params(self):
        return {}


class PSIFilter(FilterBase):
    """
    psi过滤
    """

    def __init__(self, oot_x=None, max_threshold=1, min_threshold=None, save_detail=True, fill_na=True, **kwargs):
        super().__init__(max_threshold=max_threshold, min_threshold=min_threshold, **kwargs)
        self.oot_x = oot_x
        self.save_detail = save_detail
        self.dist_detail = []
        self.fill_na = fill_na

    def show_detail(self):
        return pd.DataFrame(data=self.dist_detail,
                            columns=["col", "bin_value", "ins_num", "ins_rate", "oot_num", "oot_rate", "psi"])

    def udf_fit(self, s: dataframe_type, **kwargs):
        oot_x = self.transform_all_parent(self.oot_x)
        if self.fill_na:
            fill_na_model = FillNa(cols=self.cols)
            s = fill_na_model.fit(s, **kwargs).transform(s, **kwargs)
            oot_x = fill_na_model.transform(oot_x, **kwargs)
        total_ins_num = len(s)
        total_oot_num = len(oot_x)
        self.dist_detail = []
        for col in self.cols:
            for bin_value in s[col].unique().tolist():
                ins_num = np.sum(s[col] == bin_value)
                oot_num = np.sum(oot_x[col] == bin_value)
                ins_rate = ins_num / total_ins_num
                oot_rate = oot_num / total_oot_num
                if ins_num == 0 or oot_num == 0:
                    psi = 0
                else:
                    psi = (ins_rate - oot_rate) * np.log(ins_rate / oot_rate)
                self.dist_detail.append([col, bin_value, ins_num, ins_rate, oot_num, oot_rate, psi])
        self.filter_value_dist = self.show_detail()[["col", "psi"]].groupby("col").agg(
            {"psi": "sum"}).to_dict()["psi"]
        self.selected_cols = []
        for col in s.columns:
            miss_rate = self.filter_value_dist.get(col)
            if miss_rate is None or col not in self.cols:
                self.selected_cols.append(col)
            else:
                if self.min_threshold <= abs(miss_rate) <= self.max_threshold:
                    self.selected_cols.append(col)

        del self.oot_x
        return self

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        return s[self.selected_cols]

    def udf_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        return self.extract_dict(s, self.selected_cols)

    def udf_get_params(self):
        params = {"save_detail": self.save_detail}
        if self.save_detail:
            params["dist_detail"] = self.dist_detail
        return params

    def udf_set_params(self, params: dict_type):
        self.save_detail = params["save_detail"]
        if self.save_detail:
            self.dist_detail = params["dist_detail"]
        else:
            self.dist_detail = []


class CorrFilter(FilterBase):
    """
    高相关列去重过滤器（v0.2.5 新增）：对数值列计算相关矩阵，|corr| 超过阈值的
    列对中删除一列（提供 y 时保留与目标相关性更高的列，否则保留顺序靠前的列）。
    Kaggle 常用于树模型前的冗余特征清理。

    Args:
        threshold: 相关性阈值（|corr| > threshold 视为冗余）
        y: 目标变量（可选，用于决定保留哪一列）
        cols: 参与筛选的列；"all"/None 表示全部列（非数值列自动跳过）

    Example:
        >>> from easymlops.table.feature_selection import CorrFilter
        >>> pipe = CorrFilter(threshold=0.95, y=y_train)
        >>> pipe.fit(train_df)
        >>> pipe.dropped_cols_   # 被删除的列
    """

    def __init__(self, threshold=0.95, y=None, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.y = y
        self.dropped_cols_ = []

    def udf_fit(self, s: dataframe_type, **kwargs):
        numeric_cols = [c for c in self.cols if np.issubdtype(s[c].dtype, np.number)]
        if len(numeric_cols) < 2:
            self.selected_cols = list(self.cols)
            self.dropped_cols_ = []
            self.filter_value_dist = {}
            return self

        if self.y is not None:
            target_corr = {c: abs(float(pd.Series(s[c].values).corr(
                pd.Series(np.asarray(self.y, dtype=float))))) for c in numeric_cols}
        else:
            target_corr = {c: 0.0 for c in numeric_cols}

        corr_matrix = s[numeric_cols].corr().abs()
        dropped = set()
        # 按列对相关性从高到低贪心处理
        pairs = []
        for i, a in enumerate(numeric_cols):
            for b in numeric_cols[i + 1:]:
                pairs.append((corr_matrix.loc[a, b], a, b))
        pairs.sort(key=lambda x: -x[0])
        for corr_value, a, b in pairs:
            if corr_value <= self.threshold:
                break
            if a in dropped and b in dropped:
                continue
            # 保留与目标相关性更高(或顺序靠前)的列
            drop = b if target_corr.get(a, 0) >= target_corr.get(b, 0) else a
            if drop in dropped:
                drop = b if drop == a else a
            dropped.add(drop)

        self.dropped_cols_ = sorted(dropped)
        self.selected_cols = [c for c in self.cols if c not in dropped]
        self.filter_value_dist = {"dropped_cols": ",".join(self.dropped_cols_)}
        return self

    def udf_get_params(self):
        return {"threshold": self.threshold, "dropped_cols_": self.dropped_cols_,
                "y_len": len(self.y) if self.y is not None else 0}

    def udf_set_params(self, params: dict_type):
        self.threshold = params["threshold"]
        self.dropped_cols_ = params["dropped_cols_"]


def _lgb_importance(s: dataframe_type, y, feature_cols, task="classification",
                    native_init_params=None, seed=42):
    """训练 LightGBM 并返回 {col: importance}（惰性导入 lightgbm）。"""
    import lightgbm as lgb
    x = s[feature_cols].fillna(0).to_numpy(dtype=float)
    params = {"verbosity": -1, "seed": seed}
    if native_init_params:
        params.update(native_init_params)
    if task == "classification":
        num_class = len(np.unique(y))
        if num_class <= 2:
            params.update({"objective": "binary"})
        else:
            params.update({"objective": "multiclass", "num_class": num_class})
        model = lgb.LGBMClassifier(**params)
    else:
        params.update({"objective": "regression"})
        model = lgb.LGBMRegressor(**params)
    model.fit(x, y)
    return dict(zip(feature_cols, model.feature_importances_))


class NullImportanceFilter(FilterBase):
    """
    空重要性筛选（v0.2.5 新增，Kaggle 特征筛选利器）：将 y 随机打乱训练 LGBM
    得到"空重要性"分布，真实重要性显著高于空分布分位数（默认 >75 分位）的
    特征才保留——能有效剔除"碰巧重要"的噪声特征。

    Args:
        y: 目标变量（必填）
        task: "classification" / "regression"
        n_runs: y 打乱次数（越大分布越稳，耗时越长）
        percentile: 空重要性分位数阈值（默认 75）
        native_init_params: 透传 LGBM 的参数（如 {"n_estimators": 100}）
        random_state: 随机种子
        cols: 参与筛选的列；"all"/None 表示全部列（非数值列自动跳过）

    Example:
        >>> from easymlops.table.feature_selection import NullImportanceFilter
        >>> pipe = NullImportanceFilter(y=y_train, n_runs=10)
        >>> pipe.fit(train_df)
        >>> pipe.selected_cols
    """

    def __init__(self, y=None, task="classification", n_runs=20, percentile=75,
                 native_init_params=None, random_state=42, **kwargs):
        kwargs.setdefault("min_threshold", 0.0)   # score > 0 才保留
        super().__init__(**kwargs)
        self.y = y
        self.task = task
        self.n_runs = int(n_runs)
        self.percentile = percentile
        self.native_init_params = native_init_params
        self.random_state = random_state

    def udf_fit(self, s: dataframe_type, **kwargs):
        if self.y is None:
            raise Exception("NullImportanceFilter requires y at construction")
        numeric_cols = [c for c in self.cols if np.issubdtype(s[c].dtype, np.number)]
        y = np.asarray(self.y)
        real_imp = _lgb_importance(s, y, numeric_cols, self.task,
                                   self.native_init_params, self.random_state)
        null_imps = {c: [] for c in numeric_cols}
        for i in range(self.n_runs):
            y_shuffled = y.copy()
            np.random.RandomState(self.random_state + i + 1).shuffle(y_shuffled)
            imp = _lgb_importance(s, y_shuffled, numeric_cols, self.task,
                                  self.native_init_params, self.random_state + i + 1)
            for c in numeric_cols:
                null_imps[c].append(imp[c])
        self.filter_value_dist = {
            c: float(real_imp[c] - np.percentile(null_imps[c], self.percentile))
            for c in numeric_cols
        }
        # 非数值列不参与打分, 直接保留
        self.selected_cols = [c for c in self.cols
                              if c not in numeric_cols or
                              self.filter_value_dist[c] > self.min_threshold]
        return self

    def udf_get_params(self):
        return {"task": self.task, "n_runs": self.n_runs, "percentile": self.percentile,
                "native_init_params": self.native_init_params,
                "random_state": self.random_state,
                "y_len": len(self.y) if self.y is not None else 0}

    def udf_set_params(self, params: dict_type):
        self.task = params["task"]
        self.n_runs = params["n_runs"]
        self.percentile = params["percentile"]
        self.native_init_params = params["native_init_params"]
        self.random_state = params["random_state"]


class RFEFilter(FilterBase):
    """
    递归特征消除（v0.2.5 新增）：以 LightGBM 重要性做贪心后向消除，每轮删除
    最不重要的 step 个特征，直到剩余 n_features_to_select 个。

    Args:
        y: 目标变量（必填）
        n_features_to_select: 最终保留的特征数
        task: "classification" / "regression"
        step: 每轮删除的特征数（默认 1；设大可加速）
        native_init_params: 透传 LGBM 的参数
        random_state: 随机种子
        cols: 参与筛选的列；"all"/None 表示全部列（非数值列自动跳过）

    Example:
        >>> from easymlops.table.feature_selection import RFEFilter
        >>> pipe = RFEFilter(y=y_train, n_features_to_select=10)
        >>> pipe.fit(train_df)
        >>> pipe.selected_cols
    """

    def __init__(self, y=None, n_features_to_select=10, task="classification", step=1,
                 native_init_params=None, random_state=42, **kwargs):
        super().__init__(**kwargs)
        self.y = y
        self.n_features_to_select = int(n_features_to_select)
        self.task = task
        self.step = max(1, int(step))
        self.native_init_params = native_init_params
        self.random_state = random_state

    def udf_fit(self, s: dataframe_type, **kwargs):
        if self.y is None:
            raise Exception("RFEFilter requires y at construction")
        numeric_cols = [c for c in self.cols if np.issubdtype(s[c].dtype, np.number)]
        remaining = list(numeric_cols)
        y = np.asarray(self.y)
        final_imp = None
        while len(remaining) > self.n_features_to_select:
            imp = _lgb_importance(s, y, remaining, self.task,
                                  self.native_init_params, self.random_state)
            drop_count = min(self.step, len(remaining) - self.n_features_to_select)
            drop_cols = sorted(imp, key=lambda c: imp[c])[:drop_count]
            remaining = [c for c in remaining if c not in drop_cols]
            final_imp = imp
        if len(remaining) == len(numeric_cols):
            final_imp = _lgb_importance(s, y, remaining, self.task,
                                        self.native_init_params, self.random_state)
        self.selected_cols = [c for c in self.cols if c in remaining or
                              not np.issubdtype(s[c].dtype, np.number)]
        self.filter_value_dist = final_imp if final_imp else {}
        return self

    def udf_get_params(self):
        return {"n_features_to_select": self.n_features_to_select, "task": self.task,
                "step": self.step, "native_init_params": self.native_init_params,
                "random_state": self.random_state,
                "y_len": len(self.y) if self.y is not None else 0}

    def udf_set_params(self, params: dict_type):
        self.n_features_to_select = params["n_features_to_select"]
        self.task = params["task"]
        self.step = params["step"]
        self.native_init_params = params["native_init_params"]
        self.random_state = params["random_state"]
