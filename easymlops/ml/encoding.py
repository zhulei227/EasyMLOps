from ..base import *
from .preprocessing import FillNa


class EncodingBase(PipeObject):
    """
    只需要对单个key-value处理的类型
    """

    def __init__(self, cols="all", y=None, fill_na=True, **kwargs):
        super().__init__(**kwargs)
        self.cols = cols
        self.y = y
        self.fill_na = fill_na
        self.fill_na_model = None

    def apply_function_series(self, col: str, x: series_type):
        """
        col:当前col名称
        x:当前col对应的值
        """
        raise Exception("need to implement")

    def apply_function_single(self, col: str, x):
        """
        col:当前col名称
        x:当前col对应的值
        """
        raise Exception("need to implement")

    @staticmethod
    def extract_dict(s: dict_type, keys: list):
        new_s = dict()
        for key in keys:
            new_s[key] = s[key]
        return new_s

    def before_fit(self, s: dataframe_type) -> dataframe_type:
        s = super().before_fit(s)
        if str(self.cols) == "all" or self.cols is None or (type(self.cols) == list and len(self.cols) == 0):
            self.cols = []
            for col in s.columns.tolist():
                self.cols.append((col, col))
        else:
            if type(self.cols) == list:
                if type(self.cols[0]) == tuple or type(self.cols[0]) == list:
                    pass
                else:
                    new_cols = []
                    for col in self.cols:
                        new_cols.append((col, col))
                    self.cols = new_cols
            else:
                raise Exception("cols should be None,'all' or a list")
        # 是否fill na
        if self.fill_na:
            self.fill_na_model = FillNa(cols=[col for col, _ in self.cols], fill_category_value="nan",
                                        fill_number_value=0)
            s = self.fill_na_model.fit(s).transform(s)
        return s

    def before_transform(self, s: dataframe_type) -> dataframe_type:
        s = super().before_transform(s)
        if self.fill_na:
            s = self.fill_na_model.transform(s)
        return s

    def _transform(self, s: dataframe_type) -> dataframe_type:
        for col, new_col in self.cols:
            if col in s.columns:
                s[new_col] = self.apply_function_series(col, s[col])
        return s

    def before_transform_single(self, s: dict_type) -> dict_type:
        s = super().before_transform_single(s)
        if self.fill_na:
            s = self.fill_na_model.transform_single(s)
        return s

    def _transform_single(self, s: dict_type) -> dict_type:
        for col, new_col in self.cols:
            if col in s.keys():
                s[new_col] = self.apply_function_single(col, s[col])
        return s

    def _get_params(self) -> dict_type:
        params = {"cols": self.cols, "fill_na": self.fill_na}
        if self.fill_na:
            params["fill_na_model"] = self.fill_na_model.get_params()
        return params

    def _set_params(self, params: dict):
        self.cols = params["cols"]
        self.fill_na = params["fill_na"]
        if self.fill_na:
            self.fill_na_model = FillNa()
            self.fill_na_model.set_params(params["fill_na_model"])


class TargetEncoding(EncodingBase):
    """
      input type:pandas.dataframe
      input like:
      | 0 | 1 |
      |good|A|
      |bad|B|
      |good|A|
      |bad|B|

      y like:
      |y|
      |1|
      |0|
      |0|
      |1|
      ----------------------------
      output type:pandas.dataframe
      output like:
      | 0 | 1 |
      |0.5|0.5|
      |0.5|0.5|
      |0.5|0.5|
      |0.5|0.5|
      """

    def __init__(self, cols="all", error_value=0, fill_na=True, **kwargs):
        super().__init__(cols=cols, fill_na=fill_na, **kwargs)
        self.error_value = error_value
        self.target_map_detail = dict()

    def show_detail(self):
        data = []
        for col, map_detail in self.target_map_detail.items():
            for bin_value, target_value in map_detail.items():
                data.append([col, bin_value, target_value])
        return pd.DataFrame(data=data, columns=["col", "bin_value", "target_value"])

    def _fit(self, s: dataframe_type):
        assert self.y is not None and len(self.y) == len(s)
        s["y_"] = self.y
        for col, _ in self.cols:
            tmp_ = s[[col, "y_"]]
            col_map = list(tmp_.groupby([col]).agg({"y_": ["mean"]}).to_dict().values())[0]
            self.target_map_detail[col] = col_map
        del s["y_"]
        return self

    def apply_function_single(self, col: str, x):
        map_detail_ = self.target_map_detail.get(col, dict())
        return float(map_detail_.get(x, self.error_value))

    def apply_function_series(self, col: str, x: series_type):
        map_detail_ = self.target_map_detail.get(col, dict())
        x_ = copy.deepcopy(x)
        match_ = copy.deepcopy(x)
        for key, value in map_detail_.items():
            x[x_ == key] = value
            match_[x_ == key] = True
        x[match_ != True] = self.error_value
        return x.astype(float)

    def _get_params(self) -> dict_type:
        return {"target_map_detail": self.target_map_detail, "error_value": self.error_value}

    def _set_params(self, params: dict_type):
        self.target_map_detail = params["target_map_detail"]
        self.error_value = params["error_value"]


class LabelEncoding(EncodingBase):
    """
      input type:pandas.dataframe
      input like:
      | 0 | 1 |
      |good|A|
      |bad|B|
      |good|A|
      |bad|B|
      ----------------------------
      output type:pandas.dataframe
      output like:
      | 0 | 1 |
      |1|0|
      |0|1|
      |1|0|
      |0|1|
      """

    def __init__(self, cols="all", error_value=0, fill_na=True, **kwargs):
        super().__init__(cols=cols, fill_na=fill_na, **kwargs)
        self.error_value = error_value
        self.label_map_detail = dict()

    def show_detail(self):
        return pd.DataFrame([self.label_map_detail])

    def _fit(self, s: dataframe_type):
        for col, _ in self.cols:
            col_map = s[col].value_counts().to_dict()
            c = 1
            for key, value in col_map.items():
                col_map[key] = c
                c += 1
            self.label_map_detail[col] = col_map
        return self

    def apply_function_single(self, col: str, x):
        map_detail_ = self.label_map_detail.get(col, dict())
        return int(map_detail_.get(x, self.error_value))

    def apply_function_series(self, col: str, x: series_type):
        map_detail_ = self.label_map_detail.get(col, dict())
        x_ = copy.deepcopy(x)
        match_ = copy.deepcopy(x)
        for key, value in map_detail_.items():
            x[x_ == key] = value
            match_[x_ == key] = True
        x[match_ != True] = self.error_value
        return x.astype(int)

    def _get_params(self) -> dict_type:
        return {"label_map_detail": self.label_map_detail, "error_value": self.error_value}

    def _set_params(self, params: dict):
        self.label_map_detail = params["label_map_detail"]
        self.error_value = params["error_value"]


class OneHotEncoding(EncodingBase):
    """
         input type:pandas.dataframe
         input like:
         | x | y |
         |good|A|
         |bad|B|
         |good|A|
         |bad|B|
         ----------------------------
         output type:pandas.dataframe
         output like:
         | x_good | x_bad | y_A | y_B |
         |   1    |   0   |  1  |  0  |
         |   0    |   1   |  0  |  1  |
         |   1    |   0   |  1  |  0  |
         |   0    |   1   |  0  |  1  |
         """

    def __init__(self, cols="all", drop_col=True, fill_na=True, **kwargs):
        super().__init__(cols=cols, fill_na=fill_na, **kwargs)
        self.drop_col = drop_col
        self.one_hot_detail = dict()

    def show_detail(self):
        return pd.DataFrame([self.one_hot_detail])

    def _fit(self, s: dataframe_type):
        for col, _ in self.cols:
            self.one_hot_detail[col] = s[col].unique().tolist()
        return self

    def _transform(self, s: dataframe_type) -> dataframe_type:
        for col, new_col in self.cols:
            if col not in self.one_hot_detail.keys():
                raise Exception("{} not in {}".format(col, self.one_hot_detail.keys()))
            values = self.one_hot_detail.get(col)
            for value in values:
                s["{}_{}".format(new_col, value)] = (s[col] == value).astype(np.uint8)
        if self.drop_col:
            for col in self.one_hot_detail.keys():
                del s[col]
        return s

    def _transform_single(self, s: dict_type) -> dict_type:
        for col, new_col in self.cols:
            if col not in self.one_hot_detail.keys():
                raise Exception("{} not in {}".format(col, self.one_hot_detail.keys()))
            values = self.one_hot_detail.get(col)
            for value in values:
                if s[col] == value:
                    s["{}_{}".format(new_col, value)] = 1
                else:
                    s["{}_{}".format(new_col, value)] = 0
        if self.drop_col:
            for col in self.one_hot_detail.keys():
                del s[col]
        return s

    def _get_params(self) -> dict_type:
        return {"one_hot_detail": self.one_hot_detail, "drop_col": self.drop_col}

    def _set_params(self, params: dict):
        self.one_hot_detail = params["one_hot_detail"]
        self.drop_col = params["drop_col"]


class WOEEncoding(EncodingBase):
    """
         input type:pandas.dataframe
         input like:
         | x | y |
         |good|1|
         |bad|0|
         |good|1|
         |bad|0|
         ----------------------------
         output type:pandas.dataframe
         output like:
         | x |
         | 1 |
         | 0 |
         | 1 |
         | 0 |
         """

    def __init__(self, y=None, cols="all", drop_col=False, save_detail=True, fill_na=True, error_value=0, **kwargs):
        super().__init__(cols=cols, fill_na=fill_na, **kwargs)
        self.drop_col = drop_col
        self.error_value = error_value
        self.woe_map_detail = dict()
        self.save_detail = save_detail
        self.dist_detail = []
        self.y = y

    def _fit(self, s: dataframe_type):
        # 检测y的长度与训练数据是否一致
        assert self.y is not None and len(self.y) == len(s)
        total_bad_num = np.sum(self.y == 1)
        total_good_num = len(self.y) - total_bad_num
        if total_bad_num == 0 or total_good_num == 0:
            raise Exception("should total_bad_num > 0 and total_good_num > 0")
        s["target_value__"] = self.y
        self.woe_map_detail = dict()
        self.dist_detail = []
        for col, _ in self.cols:
            for bin_value in s[col].unique().tolist():
                tmp = s[s[col] == bin_value]
                bad_num = np.sum(tmp["target_value__"] == 1)
                good_num = len(tmp) - bad_num
                bad_rate = bad_num / total_bad_num
                good_rate = good_num / total_good_num
                if good_num == 0 or bad_num == 0:
                    woe = 0
                    iv = 0
                else:
                    woe = np.log(good_rate / bad_rate)
                    iv = (good_rate - bad_rate) * woe
                if self.woe_map_detail.get(col) is None:
                    self.woe_map_detail[col] = dict()
                self.woe_map_detail[col][bin_value] = woe
                self.dist_detail.append([col, bin_value, bad_num, bad_rate, good_num, good_rate, woe, iv])
        del s["target_value__"]
        return self

    def show_detail(self):
        return pd.DataFrame(data=self.dist_detail,
                            columns=["col", "bin_value", "bad_num", "bad_rate", "good_num", "good_rate", "woe", "iv"])

    def _transform(self, s: dataframe_type) -> dataframe_type:
        for col, new_col in self.cols:
            if col not in self.woe_map_detail.keys():
                raise Exception("{} not in {}".format(col, self.woe_map_detail.keys()))
            s[new_col] = self.apply_function_series(col, s[col])
        if self.drop_col:
            for col in self.woe_map_detail.keys():
                del s[col]
        return s

    def _transform_single(self, s: dict_type) -> dict_type:
        for col, new_col in self.cols:
            if col not in self.woe_map_detail.keys():
                raise Exception("{} not in {}".format(col, self.woe_map_detail.keys()))
            s[new_col] = self.apply_function_single(col, s[col])
        if self.drop_col:
            for col in self.woe_map_detail.keys():
                del s[col]
        return s

    def apply_function_single(self, col: str, x):
        map_detail_ = self.woe_map_detail.get(col, dict())
        return float(map_detail_.get(x, self.error_value))

    def apply_function_series(self, col: str, x: series_type):
        map_detail_ = self.woe_map_detail.get(col, dict())
        x_ = copy.deepcopy(x)
        match_ = copy.deepcopy(x)
        for key, value in map_detail_.items():
            x[x_ == key] = value
            match_[x_ == key] = True
        x[match_ != True] = self.error_value
        return x.astype(float)

    def _get_params(self) -> dict_type:
        params = {"woe_map_detail": self.woe_map_detail, "error_value": self.error_value,
                  "drop_col": self.drop_col, "save_detail": self.save_detail}
        if self.save_detail:
            params["dist_detail"] = self.dist_detail
        return params

    def _set_params(self, params: dict):
        self.save_detail = params["save_detail"]
        self.error_value = params["error_value"]
        if self.save_detail:
            self.dist_detail = params["dist_detail"]
        else:
            self.dist_detail = []
        self.woe_map_detail = params["woe_map_detail"]
        self.drop_col = params["drop_col"]
