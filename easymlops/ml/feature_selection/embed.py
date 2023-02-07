from ...base import *
from ..classification import LGBMClassification, LogisticRegressionClassification


class EmbedBase(PipeObject):
    def __init__(self, y: series_type = None, cols="all", min_threshold=None, max_threshold=None,
                 native_init_params=None,
                 native_fit_params=None, skip_check_transform_type=True, skip_check_transform_value=True,
                 **kwargs):
        super().__init__(skip_check_transform_value=skip_check_transform_value,
                         skip_check_transform_type=skip_check_transform_type, **kwargs)
        self.y = y
        self.cols = cols
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        if self.min_threshold is None:
            self.min_threshold = np.finfo(np.float64).min
        if self.max_threshold is None:
            self.max_threshold = np.finfo(np.float64).max
        # 底层模型自带参数
        self.native_init_params = native_init_params
        self.native_fit_params = native_fit_params
        if self.native_init_params is None:
            self.native_init_params = dict()
        if self.native_fit_params is None:
            self.native_fit_params = dict()
        # embed_value_dist
        self.embed_value_dist = dict()
        self.selected_cols = None

    def show_detail(self):
        return pd.DataFrame([self.embed_value_dist])

    def before_fit(self, s: dataframe_type) -> dataframe_type:
        s = super().before_fit(s)
        if str(self.cols).lower() in ["none", "all", "null", "nan"]:
            self.cols = self.input_col_names
        assert type(self.cols) == list and type(self.cols[0]) == str
        return s

    def _transform(self, s: dataframe_type) -> dataframe_type:
        return s[self.selected_cols]

    def _transform_single(self, s: dict_type) -> dict_type:
        return self.extract_dict(s, self.selected_cols)

    def _get_params(self):
        return {"cols": self.cols, "native_init_params": self.native_init_params,
                "native_fit_params": self.native_fit_params, "min_threshold": self.min_threshold,
                "max_threshold": self.max_threshold,
                "embed_value_dist": self.embed_value_dist, "selected_cols": self.selected_cols}

    def _set_params(self, params: dict_type):
        self.cols = params["cols"]
        self.native_init_params = params["native_init_params"]
        self.native_fit_params = params["native_fit_params"]
        self.min_threshold = params["min_threshold"]
        self.max_threshold = params["max_threshold"]
        self.embed_value_dist = params["embed_value_dist"]
        self.selected_cols = params["selected_cols"]


class LREmbed(EmbedBase):
    """
    logistic embed 特征选择
    """

    def __init__(self, y=None, min_threshold=1e-3, multi_class="auto", solver="newton-cg", max_iter=1000,
                 **kwargs):
        super().__init__(y=y, min_threshold=min_threshold, **kwargs)
        self.multi_class = multi_class
        self.solver = solver
        self.max_iter = max_iter

    def _fit(self, s: dataframe_type):
        s_ = s[self.cols]
        model = LogisticRegressionClassification(y=self.y, multi_class=self.multi_class, solver=self.solver,
                                                 max_iter=self.max_iter,
                                                 native_init_params=self.native_init_params,
                                                 native_fit_params=self.native_fit_params).fit(s_)
        self.embed_value_dist = dict()
        for idx, weight in enumerate(model.lr.coef_[0]):
            self.embed_value_dist[self.cols[idx]] = abs(weight)
        self.selected_cols = []
        for col in s.columns:
            weight = self.embed_value_dist.get(col)
            if weight is None or col not in self.cols:
                self.selected_cols.append(col)
            else:
                if self.min_threshold <= abs(weight) <= self.max_threshold:
                    self.selected_cols.append(col)
        return self

    def _get_params(self):
        return {}


class LGBMEmbed(EmbedBase):
    """
    lgb embed 特征选择
    """

    def __init__(self, y=None, objective="regression", importance_type="split", **kwargs):
        super().__init__(y=y, **kwargs)
        self.objective = objective
        self.importance_type = importance_type

    def _fit(self, s: dataframe_type):
        s_ = s[self.cols]
        model = LGBMClassification(y=self.y, use_faster_predictor=False, objective=self.objective,
                                   native_init_params=self.native_init_params,
                                   native_fit_params=self.native_fit_params)
        if self.objective != "multiclass":
            self.native_init_params["num_class"] = 1
        model.fit(s_)
        self.embed_value_dist = dict()
        for idx, weight in enumerate(model.lgb_model.feature_importance(self.importance_type)):
            self.embed_value_dist[self.cols[idx]] = abs(weight)
        self.selected_cols = []
        for col in s.columns:
            weight = self.embed_value_dist.get(col)
            if weight is None or col not in self.cols:
                self.selected_cols.append(col)
            else:
                if self.min_threshold <= abs(weight) <= self.max_threshold:
                    self.selected_cols.append(col)
        return self

    def _get_params(self):
        return {}
