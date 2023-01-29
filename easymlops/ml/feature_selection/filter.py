from ...base import *
from ..encoding import WOEEncoding
from ..preprocessing import FillNa


class FilterBase(PipeObject):
    def __init__(self, cols="all", as_null_values=None, min_threshold=None, max_threshold=None,
                 skip_check_transform_type=True, skip_check_transform_value=True,
                 **kwargs):
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
        # filter_value_dist分布
        self.filter_value_dist = dict()
        self.selected_cols = None

    def get_filter_value_dist(self):
        return self.filter_value_dist

    def show_detail(self):
        return pd.DataFrame([self.filter_value_dist])

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
        return {"cols": self.cols, "min_threshold": self.min_threshold,
                "max_threshold": self.max_threshold, "as_null_values": self.as_null_values,
                "filter_value_dist": self.filter_value_dist, "selected_cols": self.selected_cols}

    def _set_params(self, params: dict_type):
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

    def _fit(self, s: dataframe_type):
        self.filter_value_dist = dict()
        for col in self.cols:
            if self.as_null_values is not None and ("str" in type(s[col]) or "object" in type(s[col])):
                for null_value in self.as_null_values:
                    s[col] = np.where(s[col] == null_value, None, s[col])
            miss_rate = s[col].isnull().sum() / len(s)
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

    def _get_params(self):
        return {}


class VarianceFilter(FilterBase):
    """
    方差值过滤
    """

    def __init__(self, max_threshold=None, min_threshold=1e-3, **kwargs):
        super().__init__(max_threshold=max_threshold, min_threshold=min_threshold, **kwargs)

    def _fit(self, s: dataframe_type):
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

    def _get_params(self):
        return {}


class PersonCorrFilter(FilterBase):
    """
    person相关性绝对值过滤
    """

    def __init__(self, y=None, max_threshold=None, min_threshold=0.01, **kwargs):
        super().__init__(max_threshold=max_threshold, min_threshold=min_threshold, **kwargs)
        self.y = y

    def _fit(self, s: dataframe_type):
        assert self.y is not None and len(self.y) == len(s)
        s["target_value__"] = self.y
        self.filter_value_dist = s[self.cols + ["target_value__"]].corr()["target_value__"].to_dict()
        del s["target_value__"]
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

    def _get_params(self):
        return {}


class Chi2Filter(FilterBase):
    """
    卡方过滤
    """

    def __init__(self, y=None, max_threshold=None, min_threshold=0.01, **kwargs):
        super().__init__(max_threshold=max_threshold, min_threshold=min_threshold, **kwargs)
        self.y = y

    def _fit(self, s: dataframe_type):
        from sklearn.feature_selection import chi2
        assert self.y is not None and len(self.y) == len(s)
        self.filter_value_dist = dict()
        for idx, filter_value in enumerate(chi2(s[self.cols], self.y)[0]):
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

    def _get_params(self):
        return {}


class PValueFilter(FilterBase):
    """
    P值过滤
    """

    def __init__(self, y=None, max_threshold=None, min_threshold=None, **kwargs):
        super().__init__(max_threshold=max_threshold, min_threshold=min_threshold, **kwargs)
        self.y = y

    def _fit(self, s: dataframe_type):
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

    def _get_params(self):
        return {}


class MutualInfoFilter(FilterBase):
    """
    互信息过滤
    """

    def __init__(self, y=None, max_threshold=None, min_threshold=None, **kwargs):
        super().__init__(max_threshold=max_threshold, min_threshold=min_threshold, **kwargs)
        self.y = y

    def _fit(self, s: dataframe_type):
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

    def _get_params(self):
        return {}


class IVFilter(FilterBase):
    """
    IV过滤
    """

    def __init__(self, y=None, max_threshold=None, min_threshold=0.02, **kwargs):
        super().__init__(max_threshold=max_threshold, min_threshold=min_threshold, **kwargs)
        self.y = y

    def _fit(self, s: dataframe_type):
        assert self.y is not None and len(self.y) == len(s)
        self.filter_value_dist = \
            WOEEncoding(y=self.y, cols=self.cols).fit(s).show_detail()[["col", "iv"]].groupby("col").agg(
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

    def _get_params(self):
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

    def _fit(self, s: dataframe_type):
        oot_x = self.oot_x
        if self.fill_na:
            fill_na_model = FillNa(cols=self.cols)
            s = fill_na_model.fit(s).transform(s)
            oot_x = fill_na_model.transform(oot_x)
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

    def _transform(self, s: dataframe_type) -> dataframe_type:
        return s[self.selected_cols]

    def _transform_single(self, s: dict_type) -> dict_type:
        return self.extract_dict(s, self.selected_cols)

    def _get_params(self):
        params = {"save_detail": self.save_detail}
        if self.save_detail:
            params["dist_detail"] = self.dist_detail
        return params

    def _set_params(self, params: dict_type):
        self.save_detail = params["save_detail"]
        if self.save_detail:
            self.dist_detail = params["dist_detail"]
        else:
            self.dist_detail = []
