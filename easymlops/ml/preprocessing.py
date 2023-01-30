import numpy as np

from ..base import *


class PreprocessBase(PipeObject):
    """
    只需要对单个key-value处理的类型
    """

    def __init__(self, cols="all", **kwargs):
        super().__init__(**kwargs)
        self.cols = cols

    def _user_defined_function(self, col, x):
        """
        对col的对应值x执行某种自定义操作
        """
        raise Exception("need to implement")

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
        return s

    def _transform(self, s: dataframe_type) -> dataframe_type:
        for col, new_col in self.cols:
            if col in s.columns:
                s[new_col] = s[col].apply(lambda x: self._user_defined_function(col, x))
        return s

    def _transform_single(self, s: dict_type) -> dict_type:
        for col, new_col in self.cols:
            if col in s.keys():
                s[new_col] = self._user_defined_function(col, s[col])
        return s

    def _get_params(self):
        return {"cols": self.cols}

    def _set_params(self, params: dict):
        self.cols = params["cols"]


class FixInput(PreprocessBase):
    """
    固定输入，对不存在的col用None填充
    """

    def __init__(self, cols="all", fill_number_value=np.nan, fill_category_value=None,
                 skip_check_transform_type=True,
                 skip_check_transform_value=True, **kwargs):
        super().__init__(cols=cols, skip_check_transform_type=skip_check_transform_type,
                         skip_check_transform_value=skip_check_transform_value, **kwargs)
        self.fill_number_value = fill_number_value
        self.fill_category_value = fill_category_value
        self.column_dtypes = dict()

    def as_number(self, x):
        try:
            return float(str(x))
        except:
            return self.fill_number_value

    def as_category(self, x):
        try:
            return str(x)
        except:
            return self.fill_category_value

    def _fit(self, s: dataframe_type):
        self.output_col_names = self.input_col_names
        # 记录数据类型
        self.column_dtypes = dict()
        for col in self.output_col_names:
            col_type = str(s[col].dtype).lower()
            if "int" in col_type or "float" in col_type:
                self.column_dtypes[col] = "number"
            else:
                self.column_dtypes[col] = "category"
        return self

    def _check_miss_addition_columns(self, input_transform_columns):
        # 检查缺失字段
        miss_columns = list(set(self.output_col_names) - set(input_transform_columns))
        if len(miss_columns) > 0:
            print(
                "({}) module, please check these missing columns:\033[1;43m{}\033[0m, "
                "they will by filled by {}(number),{}(category)".format(
                    self.name, miss_columns, self.fill_number_value, self.fill_category_value))
        # 检查多余字段
        addition_columns = list(set(input_transform_columns) - set(self.output_col_names))
        if len(addition_columns) > 0:
            print("({}) module, please check these additional columns:\033[1;43m{}\033[0m".format(self.name,
                                                                                                  addition_columns))

    def transform(self, s: dataframe_type) -> dataframe_type:
        self._check_miss_addition_columns(s.columns)
        if self.copy_transform_data:
            s_ = copy.copy(s)
        else:
            s_ = s
        for col in self.output_col_names:
            col_type = self.column_dtypes.get(col)
            if col not in s_.columns:
                s_[col] = None
            # 调整数据类型
            if col_type == "number":
                s_[col] = s_[col].apply(lambda x: self.as_number(x))
            else:
                s_[col] = s_[col].apply(lambda x: self.as_category(x))
        return s_[self.output_col_names]

    def transform_single(self, s: dict_type) -> dict_type:
        self._check_miss_addition_columns(s.keys())
        s_ = copy.copy(s)
        for col in self.output_col_names:
            col_type = self.column_dtypes.get(col)
            if col not in s_.keys():
                s_[col] = None
            # 调整数据类型
            if col_type == "number":
                s_[col] = self.as_number(s_[col])
            else:
                s_[col] = self.as_category(s_[col])
        return self.extract_dict(s_, self.output_col_names)

    def _get_params(self) -> dict_type:
        return {"fill_number_value": self.fill_number_value,
                "fill_category_value": self.fill_category_value, "column_dtypes": self.column_dtypes}

    def _set_params(self, params: dict_type):
        self.fill_number_value = params["fill_number_value"]
        self.fill_category_value = params["fill_category_value"]
        self.column_dtypes = params["column_dtypes"]


class Replace(PreprocessBase):
    """
    source_values:["*"," "]
    target_value:""
    ----------------------------
    input type:pandas.dataframe
    input like:
    |input|
    |hi*|
    |hi |
    -------------------------
    output type:pandas.serdataframeies
    output like:
    |output|
    |hi|
    |hi|
    """

    def __init__(self, cols="all", source_values=None, target_value="", **kwargs):
        super().__init__(cols=cols, **kwargs)
        self.source_values = source_values
        if source_values is None:
            self.source_values = []
        assert type(self.source_values) == list
        self.target_value = target_value

    def _transform(self, s: dataframe_type) -> dataframe_type:
        for col, new_col in self.cols:
            for source_value in self.source_values:
                s[new_col] = s[col].astype(str).str.replace(source_value, self.target_value)
        return s

    def _transform_single(self, s: dict_type) -> dict_type:
        for col, new_col in self.cols:
            for source_value in self.source_values:
                s[new_col] = str(s[col]).replace(source_value, self.target_value)
        return s

    def _get_params(self):
        return {"source_values": self.source_values, "target_value": self.target_value}

    def _set_params(self, params: dict_type):
        self.source_values = params["source_values"]
        self.target_value = params["target_value"]


class DropCols(PreprocessBase):
    def _transform(self, s: dataframe_type) -> dataframe_type:
        for col, _ in self.cols:
            if col in s.columns.tolist():
                del s[col]
        return s

    def _transform_single(self, s: dict_type) -> dict_type:
        for col, _ in self.cols:
            if col in s.keys():
                s.pop(col)
        return s

    def _get_params(self):
        return {}


class SelectCols(PreprocessBase):
    def _transform(self, s: dataframe_type) -> dataframe_type:
        selected_cols = [col for col, _ in self.cols]
        for col in selected_cols:
            if col not in s.columns:
                raise Exception("{} not in {}".format(col, s.columns.tolist))
        s = s[selected_cols]
        return s

    def _transform_single(self, s: dict_type) -> dict_type:
        s_ = dict()
        selected_cols = [col for col, _ in self.cols]
        for col in selected_cols:
            if col not in s.keys():
                raise Exception("{} not in {}".format(col, s.keys()))
        for col in selected_cols:
            s_[col] = s[col]
        return s_

    def _get_params(self):
        return {}


class FillNa(PreprocessBase):
    """
    优先级fill_detail>fill_mode>fill_value
    fill_mode可选:mean,median,mode
    """

    def __init__(self, cols="all", fill_mode=None, fill_number_value=0, fill_category_value="nan", fill_detail=None,
                 **kwargs):
        super().__init__(cols=cols, **kwargs)
        self.fill_number_value = fill_number_value
        self.fill_category_value = fill_category_value
        self.fill_mode = fill_mode
        self.fill_detail = fill_detail

    def _fit(self, s: dataframe_type):
        if self.fill_detail is None:
            self.fill_detail = dict()
            if self.fill_mode is not None:
                for col, _ in self.cols:
                    if self.fill_mode == "mean":
                        self.fill_detail[col] = s[col].mean()
                    elif self.fill_mode == "median":
                        self.fill_detail[col] = s[col].median()
                    elif self.fill_mode == "mode":
                        self.fill_detail[col] = s[col].mode()[0]
                    else:
                        raise Exception("fill_model should be [mean,median,mode]")
            else:
                for col, _ in self.cols:
                    if "int" in str(s[col].dtype).lower() or "float" in str(s[col].dtype).lower():
                        self.fill_detail[col] = self.fill_number_value
                    else:
                        self.fill_detail[col] = self.fill_category_value
        else:
            new_cols = []
            for col, new_col in self.cols:
                if col in self.fill_detail.keys():
                    new_cols.append((col, new_col))
            self.cols = new_cols
        return self

    def _user_defined_function(self, col, x):
        if str(x).lower() in ["none", "nan", "np.nan", "null"]:
            x = self.fill_detail.get(col)
        return x

    def _get_params(self) -> dict_type:
        return {"fill_detail": self.fill_detail}

    def _set_params(self, params: dict):
        self.fill_detail = params["fill_detail"]


class TransToCategory(PreprocessBase):
    def __init__(self, cols="all", map_detail=(["None", "nan", "", " ", "*", "inf"], "nan"), **kwargs):
        super().__init__(cols=cols, **kwargs)
        self.map_detail = map_detail

    def _user_defined_function(self, col, x):
        x = str(x)
        try:
            x = str(int(float(x)))
        except:
            pass
        if x in self.map_detail[0]:
            x = self.map_detail[1]
        return x

    def _get_params(self) -> dict_type:
        return {"map_detail": self.map_detail}

    def _set_params(self, params: dict_type):
        self.map_detail = params["map_detail"]


class TransToFloat(PreprocessBase):
    def __init__(self, cols="all", nan_fill_value=0, **kwargs):
        super().__init__(cols=cols, **kwargs)
        self.nan_fill_value = nan_fill_value

    def _user_defined_function(self, col, x):
        try:
            return float(x)
        except:
            x = self.nan_fill_value
        return x

    def _get_params(self) -> dict_type:
        return {"nan_fill_value": self.nan_fill_value}

    def _set_params(self, params: dict):
        self.nan_fill_value = params["nan_fill_value"]


class TransToInt(PreprocessBase):
    def __init__(self, cols="all", nan_fill_value=0, **kwargs):
        super().__init__(cols=cols, **kwargs)
        self.nan_fill_value = nan_fill_value

    def _user_defined_function(self, col, x):
        try:
            return int(float(x))
        except:
            x = self.nan_fill_value
        return x

    def _get_params(self) -> dict_type:
        return {"nan_fill_value": self.nan_fill_value}

    def _set_params(self, params: dict):
        self.nan_fill_value = params["nan_fill_value"]


class TransToLower(PreprocessBase):
    def _user_defined_function(self, col, x):
        try:
            return str(x).lower()
        except:
            return ""

    def _get_params(self):
        return {}


class TransToUpper(PreprocessBase):
    def _user_defined_function(self, col, x):
        try:
            return str(x).upper()
        except:
            return ""

    def _get_params(self):
        return {}


class CategoryMapValues(PreprocessBase):
    def __init__(self, cols="all", default_map=([""], ""), map_detail=None, **kwargs):
        super().__init__(cols=cols, **kwargs)
        self.default_map = default_map
        self.map_detail = map_detail

    def _user_defined_function(self, col, x):
        if x in self.map_detail.get(col, ([""], ""))[0]:
            return self.map_detail.get(col, ([""], ""))[1]
        else:
            return x

    def _fit(self, s: dataframe_type):
        if self.map_detail is None:
            self.map_detail = dict()
            for col, _ in self.cols:
                self.map_detail[col] = self.default_map
        else:
            new_cols = []
            for col, new_col in self.cols:
                if col in self.map_detail.keys():
                    new_cols.append((col, new_col))
            self.cols = new_cols
        return self

    def _get_params(self) -> dict_type:
        return {"map_detail": self.map_detail}

    def _set_params(self, params: dict):
        self.map_detail = params["map_detail"]


class Clip(PreprocessBase):
    def __init__(self, cols="all", default_clip=None, clip_detail=None, percent_range=None, **kwargs):
        """
       优先级clip_detail>percent_range>default_clip
        """
        super().__init__(cols=cols, **kwargs)
        self.default_clip = default_clip
        self.clip_detail = clip_detail
        self.percent_range = percent_range

    def _user_defined_function(self, col, x):
        clip_detail_ = self.clip_detail.get(col, (None, None))
        if clip_detail_[0] is not None and str(x).lower() not in ["none", "nan", "null", "", "inf", "np.nan", "np.inf"]:
            x = clip_detail_[0] if x <= clip_detail_[0] else x
        if clip_detail_[1] is not None and str(x).lower() not in ["none", "nan", "null", "", "inf", "np.nan", "np.inf"]:
            x = clip_detail_[1] if x >= clip_detail_[1] else x
        return x

    def _fit(self, s: dataframe_type):
        if self.clip_detail is None:
            self.clip_detail = dict()
            if self.percent_range is None:
                for col, _ in self.cols:
                    self.clip_detail[col] = self.default_clip
            else:
                for col, _ in self.cols:
                    if self.percent_range[0] is not None:
                        low = np.percentile(s[col], self.percent_range[0])
                    else:
                        low = None
                    if self.percent_range[1] is not None:
                        top = np.percentile(s[col], self.percent_range[1])
                    else:
                        top = None
                    self.clip_detail[col] = (low, top)
        else:
            new_cols = []
            for col, new_col in self.cols:
                if col in self.clip_detail.keys():
                    new_cols.append((col, new_col))
            self.cols = new_cols
        return self

    def _get_params(self) -> dict_type:
        return {"clip_detail": self.clip_detail}

    def _set_params(self, params: dict):
        self.clip_detail = params["clip_detail"]


class MinMaxScaler(PreprocessBase):
    """
    最大最小归一化
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.min_max_detail = dict()

    def _check_min_max_equal(self, col, min_value, max_value):
        if min_value == max_value:
            print("({}), in  column \033[1;43m{}\033[0m ,min value and max value has the same value:{},"
                  "the finally result will be set 1".format(self.name, col, min_value))

    def show_detail(self):
        data = []
        for col, value in self.min_max_detail.items():
            min_value, max_value = value
            data.append([col, min_value, max_value])
        return pd.DataFrame(data=data, columns=["col", "min_value", "max_value"])

    def _fit(self, s: dataframe_type):
        for col, _ in self.cols:
            col_value = s[col]
            min_value = np.min(col_value)
            max_value = np.max(col_value)
            self.min_max_detail[col] = (min_value, max_value)
        return self

    def _transform(self, s: dataframe_type) -> dataframe_type:
        for col, new_col in self.cols:
            min_value, max_value = self.min_max_detail.get(col)
            self._check_min_max_equal(col, min_value, max_value)
            if min_value == max_value:
                s[new_col] = 1
            else:
                s[new_col] = s[col].apply(lambda x: (x - min_value) / (max_value - min_value))
        return s

    def _transform_single(self, s: dict_type) -> dict_type:
        for col, new_col in self.cols:
            min_value, max_value = self.min_max_detail.get(col)
            self._check_min_max_equal(col, min_value, max_value)
            if min_value == max_value:
                s[new_col] = 1
            else:
                s[new_col] = (s[col] - min_value) / (max_value - min_value)
        return s

    def _get_params(self):
        return {"min_max_detail": self.min_max_detail}

    def _set_params(self, params: dict):
        self.min_max_detail = params["min_max_detail"]


class Normalizer(PreprocessBase):
    """
    标准化
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mean_std_detail = dict()

    def _check_std(self, col, std):
        if std == 0:
            print("({}), in  column \033[1;43m{}\033[0m ,the std is 0,"
                  "the finally result will be set 1".format(self.name, col))

    def show_detail(self):
        data = []
        for col, value in self.mean_std_detail.items():
            mean_value, std_value = value
            data.append([col, mean_value, std_value])
        return pd.DataFrame(data=data, columns=["col", "mean_value", "std_value"])

    def _fit(self, s: dataframe_type):
        for col, _ in self.cols:
            col_value = s[col]
            mean_value = np.mean(col_value)
            std_value = np.std(col_value)
            self.mean_std_detail[col] = (mean_value, std_value)
        return self

    def _transform(self, s: dataframe_type) -> dataframe_type:
        for col, new_col in self.cols:
            mean_value, std_value = self.mean_std_detail.get(col)
            self._check_std(col, std_value)
            if std_value == 0:
                s[new_col] = 1
            else:
                s[new_col] = s[col].apply(lambda x: (x - mean_value) / std_value)
        return s

    def _transform_single(self, s: dict_type) -> dict_type:
        for col, new_col in self.cols:
            mean_value, std_value = self.mean_std_detail.get(col)
            self._check_std(col, std_value)
            if std_value == 0:
                s[new_col] = 1
            else:
                s[new_col] = (s[col] - mean_value) / std_value
        return s

    def _get_params(self):
        return {"mean_std_detail": self.mean_std_detail}

    def _set_params(self, params: dict):
        self.mean_std_detail = params["mean_std_detail"]


class Bins(PreprocessBase):
    """
    分箱
    strategy:uniform/等距；quantile/等位；kmeans/聚类
    """

    def __init__(self, n_bins=10, strategy="quantile", **kwargs):
        super().__init__(**kwargs)
        self.n_bins = n_bins
        self.strategy = strategy
        self.bin_detail = dict()

    def _fit(self, s: dataframe_type):
        from sklearn.preprocessing import KBinsDiscretizer
        for col, _ in self.cols:
            bin_model = KBinsDiscretizer(n_bins=self.n_bins, encode="ordinal", strategy=self.strategy)
            bin_model.fit(s[[col]])
            self.bin_detail[col] = bin_model
        return self

    def _transform(self, s: dataframe_type) -> dataframe_type:
        for col, new_col in self.cols:
            bin_model = self.bin_detail[col]
            s[new_col] = bin_model.transform(s[[col]])
        return s

    def _transform_single(self, s: dict_type) -> dict_type:
        input_dataframe = pd.DataFrame([s])
        return self._transform(input_dataframe).to_dict("record")[0]

    def _get_params(self):
        return {"n_bins": self.n_bins, "strategy": self.strategy, "bin_detail": self.bin_detail}

    def _set_params(self, params: dict_type):
        self.bin_detail = params["bin_detail"]
        self.n_bins = params["n_bins"]
        self.strategy = params["strategy"]
