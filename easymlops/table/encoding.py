from easymlops.table.core import *


class EncodingBase(TablePipeObjectBase):
    """
    Encoding的基础类
    """

    def __init__(self, cols="all", y=None, **kwargs):
        super().__init__(**kwargs)
        self.cols = cols
        self.y = y

    def apply_function_series(self, col: str, x: series_type):
        raise Exception("need to implement")

    def apply_function_single(self, col: str, x):
        raise Exception("need to implement")

    @staticmethod
    def extract_dict(s: dict_type, keys: list):
        new_s = dict()
        for key in keys:
            new_s[key] = s[key]
        return new_s

    def before_fit(self, s: dataframe_type, **kwargs) -> dataframe_type:
        s = super().before_fit(s, **kwargs)
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
        return s

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        for col, new_col in self.cols:
            if col in s.columns:
                s[new_col] = self.apply_function_series(col, s[col])
        return s

    def udf_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        for col, new_col in self.cols:
            if col in s.keys():
                s[new_col] = self.apply_function_single(col, s[col])
        return s

    def udf_get_params(self) -> dict_type:
        params = {"cols": self.cols}
        return params

    def udf_set_params(self, params: dict):
        self.cols = params["cols"]


class TargetEncoding(EncodingBase):

    def __init__(self, cols="all", error_value=0, smoothing=False, **kwargs):
        """

        :param cols:
        :param error_value: 找不到col取值时返回error_value，也可以设置为np.nan待后续处理
        :param smoothing: 平滑
        :param kwargs:
        """
        super().__init__(cols=cols, **kwargs)
        self.error_value = error_value
        self.target_map_detail = dict()
        self.smoothing = smoothing

    def show_detail(self):
        data = []
        for col, map_detail in self.target_map_detail.items():
            for bin_value, target_value in map_detail.items():
                data.append([col, bin_value, target_value])
        return pd.DataFrame(data=data, columns=["col", "bin_value", "target_value"])

    def udf_fit(self, s: dataframe_type, **kwargs):
        assert self.y is not None and len(self.y) == len(s)
        s["y_"] = self.y
        if self.smoothing:
            total_target_value = np.mean(self.y)
            total_count = len(self.y)
        for col, _ in self.cols:
            tmp_ = s[[col, "y_"]]
            col_map = list(tmp_.groupby([col]).agg({"y_": ["mean"]}).to_dict().values())[0]
            if self.smoothing:
                col_map_count = list(tmp_.groupby([col]).agg({"y_": ["count"]}).to_dict().values())[0]
                for col_name, target_value in col_map.items():
                    col_count = col_map_count.get(col_name)
                    col_rate = col_count / total_count
                    new_target_value = col_rate * target_value + (1 - col_rate) * total_target_value
                    col_map[col_name] = new_target_value
            self.target_map_detail[col] = col_map
        del s["y_"]
        return self

    def apply_function_single(self, col: str, x):
        map_detail_ = self.target_map_detail.get(col, dict())
        return np.float64(map_detail_.get(x, self.error_value))

    def apply_function_series(self, col: str, x: series_type):
        map_detail_ = self.target_map_detail.get(col, dict())
        return x.map(map_detail_).fillna(self.error_value).astype(np.float64)

    def udf_get_params(self) -> dict_type:
        return {"target_map_detail": self.target_map_detail, "error_value": self.error_value,
                "smoothing": self.smoothing}

    def udf_set_params(self, params: dict_type):
        self.target_map_detail = params["target_map_detail"]
        self.error_value = params["error_value"]
        self.smoothing = params["smoothing"]


class LabelEncoding(EncodingBase):

    def __init__(self, cols="all", error_value=0, **kwargs):
        """

        :param cols:
        :param error_value:
        :param kwargs:
        """
        super().__init__(cols=cols, **kwargs)
        self.error_value = error_value
        self.label_map_detail = dict()

    def show_detail(self):
        return pd.DataFrame([self.label_map_detail])

    def udf_fit(self, s: dataframe_type, **kwargs):
        for col, _ in self.cols:
            col_value_list = s[col].unique().tolist()
            c = 1
            col_map = dict()
            for key in col_value_list:
                col_map[key] = c
                c += 1
            self.label_map_detail[col] = col_map
        return self

    def apply_function_single(self, col: str, x):
        map_detail_ = self.label_map_detail.get(col, dict())
        return np.int64(map_detail_.get(x, self.error_value))

    def apply_function_series(self, col: str, x: series_type):
        map_detail_ = self.label_map_detail.get(col, dict())
        return x.map(map_detail_).fillna(self.error_value).astype(np.int64)

    def udf_get_params(self) -> dict_type:
        return {"label_map_detail": self.label_map_detail, "error_value": self.error_value}

    def udf_set_params(self, params: dict):
        self.label_map_detail = params["label_map_detail"]
        self.error_value = params["error_value"]


class OneHotEncoding(EncodingBase):

    def __init__(self, cols="all", drop_col=True, **kwargs):
        """

        :param cols:
        :param drop_col: one-hot展开后，默认删除原col
        :param kwargs:
        """
        super().__init__(cols=cols, **kwargs)
        self.drop_col = drop_col
        self.one_hot_detail = dict()

    def show_detail(self):
        return pd.DataFrame([self.one_hot_detail])

    def udf_fit(self, s: dataframe_type, **kwargs):
        for col, _ in self.cols:
            self.one_hot_detail[col] = s[col].unique().tolist()
        return self

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
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

    def udf_transform_single(self, s: dict_type, **kwargs) -> dict_type:
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

    def udf_get_params(self) -> dict_type:
        return {"one_hot_detail": self.one_hot_detail, "drop_col": self.drop_col}

    def udf_set_params(self, params: dict):
        self.one_hot_detail = params["one_hot_detail"]
        self.drop_col = params["drop_col"]


class WOEEncoding(EncodingBase):

    def __init__(self, y=None, cols="all", drop_col=False, save_detail=True, error_value=0, **kwargs):
        """

        :param y:
        :param cols:
        :param drop_col: encoding之后默认不删除原来的col
        :param save_detail: 是否保存map细节
        :param error_value: 出错时返回默认值
        :param kwargs:
        """
        super().__init__(cols=cols, **kwargs)
        self.drop_col = drop_col
        self.error_value = error_value
        self.woe_map_detail = dict()
        self.save_detail = save_detail
        self.dist_detail = []
        self.y = y

    def udf_fit(self, s: dataframe_type, **kwargs):
        # 检测y的长度与训练数据是否一致
        assert self.y is not None and len(self.y) == len(s)
        total_bad_num = np.sum(self.y == 1)
        total_good_num = len(self.y) - total_bad_num
        if total_bad_num == 0 or total_good_num == 0:
            raise Exception("should total_bad_num > 0 and total_good_num > 0")
        self.woe_map_detail = dict()
        self.dist_detail = []
        for col, _ in self.cols:
            for bin_value in s[col].unique().tolist():
                hint_index = s[col] == bin_value
                bad_num = np.sum(hint_index & self.y == 1)
                good_num = np.sum(hint_index) - bad_num
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
        return self

    def show_detail(self):
        return pd.DataFrame(data=self.dist_detail,
                            columns=["col", "bin_value", "bad_num", "bad_rate", "good_num", "good_rate", "woe", "iv"])

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        for col, new_col in self.cols:
            if col not in self.woe_map_detail.keys():
                raise Exception("{} not in {}".format(col, self.woe_map_detail.keys()))
            s[new_col] = self.apply_function_series(col, s[col])
        if self.drop_col:
            for col in self.woe_map_detail.keys():
                del s[col]
        return s

    def udf_transform_single(self, s: dict_type, **kwargs) -> dict_type:
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
        return np.float64(map_detail_.get(x, self.error_value))

    def apply_function_series(self, col: str, x: series_type):
        map_detail_ = self.woe_map_detail.get(col, dict())
        return x.map(map_detail_).fillna(self.error_value).astype(np.float64)

    def udf_get_params(self) -> dict_type:
        params = {"woe_map_detail": self.woe_map_detail, "error_value": self.error_value,
                  "drop_col": self.drop_col, "save_detail": self.save_detail}
        if self.save_detail:
            params["dist_detail"] = self.dist_detail
        return params

    def udf_set_params(self, params: dict):
        self.save_detail = params["save_detail"]
        self.error_value = params["error_value"]
        if self.save_detail:
            self.dist_detail = params["dist_detail"]
        else:
            self.dist_detail = []
        self.woe_map_detail = params["woe_map_detail"]
        self.drop_col = params["drop_col"]
