from ..base import *
from .perfopt import ReduceMemUsage


class PreprocessBase(PipeObject):
    """
    所有下面Preprocess类的父类
    """

    def __init__(self, cols="all", **kwargs):
        """
        cols输入格式:
        1.cols="all",cols=None表示对所有columns进行操作
        2.cols=["col1","col2"]表示对col1和col2操作
        3.cols=[("col1","new_col1"),("col2","new_col2")]表示对col1和col2操作，并将结果赋值给new_col1和new_col2，并不修改原始col1和col2
        """
        super().__init__(**kwargs)
        self.cols = cols

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
                s[new_col] = self.apply_function_series(col, s[col])
        return s

    def _transform_single(self, s: dict_type) -> dict_type:
        for col, new_col in self.cols:
            if col in s.keys():
                s[new_col] = self.apply_function_single(col, s[col])
        return s

    def _get_params(self):
        return {"cols": self.cols}

    def _set_params(self, params: dict):
        self.cols = params["cols"]


class FixInput(PreprocessBase):
    """
    固定cols中:
    1.column的名称以及顺序
    2.column的数据类型，将所有数据分为number和category两种类型，对于空或者异常值用fill_number_value或fill_category_value填充
    """

    def __init__(self, cols="all", reduce_mem_usage=True, skip_check_transform_type=True, show_check_detail=True,
                 skip_check_transform_value=True, **kwargs):
        super().__init__(cols=cols, skip_check_transform_type=skip_check_transform_type,
                         skip_check_transform_value=skip_check_transform_value, **kwargs)
        self.column_dtypes = dict()
        self.reduce_mem_usage = reduce_mem_usage
        self.reduce_mem_usage_mode = None
        self.show_check_detail = show_check_detail

    def _fit(self, s: dataframe_type):
        self.output_col_names = self.input_col_names
        # 记录数据类型
        self.column_dtypes = dict()
        for col in self.output_col_names:
            col_type = str(s[col].dtype).lower()
            if "int" in col_type or "float" in col_type:
                self.column_dtypes[col] = float
            else:
                self.column_dtypes[col] = str
        # reduce mem usage
        if self.reduce_mem_usage:
            self.reduce_mem_usage_mode = ReduceMemUsage()
            self.reduce_mem_usage_mode.fit(s)
        return self

    def _check_miss_addition_columns(self, input_transform_columns):
        # 检查缺失字段
        miss_columns = list(set(self.output_col_names) - set(input_transform_columns))
        if len(miss_columns) > 0 and self.show_check_detail:
            print(
                "({}) module, please check these missing columns:\033[1;43m{}\033[0m, "
                "they will by filled by 0(int),None(float),np.nan(category)".format(
                    self.name, miss_columns))
        # 检查多余字段
        addition_columns = list(set(input_transform_columns) - set(self.output_col_names))
        if len(addition_columns) > 0 and self.show_check_detail:
            print("({}) module, please check these additional columns:\033[1;43m{}\033[0m".format(self.name,
                                                                                                  addition_columns))

    def transform(self, s: dataframe_type) -> dataframe_type:
        # 检查缺失
        self._check_miss_addition_columns(s.columns)
        # copy数据
        if self.copy_transform_data:
            s_ = copy.copy(s)
        else:
            s_ = s
        for col in self.output_col_names:
            col_type = self.column_dtypes.get(col)
            # 空值填充
            if col not in s_.columns:
                if col_type == str:
                    s_[col] = None
                else:
                    s_[col] = np.nan
            # 调整数据类型
            try:
                s_[col] = s_[col].astype(col_type)
            except:
                # 只有float会报错
                s_[col] = s_[col].apply(lambda x: self.as_float(x))
        # reduce mem usage
        if self.reduce_mem_usage:
            s_ = self.reduce_mem_usage_mode.transform(s_)
        return s_[self.output_col_names]

    @staticmethod
    def as_float(x):
        try:
            return float(x)
        except:
            return np.nan

    def transform_single(self, s: dict_type) -> dict_type:
        # 检验冲突
        self._check_miss_addition_columns(s.keys())
        # copy数据
        s_ = copy.copy(s)
        for col in self.output_col_names:
            col_type = self.column_dtypes.get(col)
            # 空值填充
            if col not in s_.keys():
                if col_type == str:
                    s_[col] = None
                else:
                    s_[col] = np.nan
            # 调整数据类型
            try:
                s_[col] = col_type(s_[col])
            except:
                # 只有float会出错
                s_[col] = np.nan
        # reduce mem usage
        if self.reduce_mem_usage:
            s_ = self.reduce_mem_usage_mode.transform_single(s_)
        return self.extract_dict(s_, self.output_col_names)

    def _get_params(self) -> dict_type:
        params = {"column_dtypes": self.column_dtypes, "reduce_mem_usage": self.reduce_mem_usage}
        if self.reduce_mem_usage:
            params["reduce_mem_usage_mode"] = self.reduce_mem_usage_mode.get_params()
        return params

    def _set_params(self, params: dict_type):
        self.column_dtypes = params["column_dtypes"]
        self.reduce_mem_usage = params["reduce_mem_usage"]
        if self.reduce_mem_usage:
            self.reduce_mem_usage_mode = ReduceMemUsage()
            self.reduce_mem_usage_mode.set_params(params["reduce_mem_usage_mode"])


class Replace(PreprocessBase):
    """
    1.将cols中特定字符替换为目标字符
    cols=["input"]
    source_values:["*"," "]
    target_value:""
    ----------------------------
    |input|
    |hi*|
    |hi |
    -------------------------
    |input|
    |hi|
    |hi|
    ----------------------------------------------------
    2.cols输入格式:
    2.1.cols="all",cols=None表示对所有columns进行操作
    2.2.cols=["col1","col2"]表示对col1和col2操作
    2.3.cols=[("col1","new_col1"),("col2","new_col2")]表示对col1和col2操作，并将结果赋值给new_col1和new_col2，并不修改原始col1和col2
    """

    def __init__(self, cols="all", source_values=None, target_value="", **kwargs):
        super().__init__(cols=cols, **kwargs)
        self.source_values = source_values
        if source_values is None:
            self.source_values = []
        assert type(self.source_values) == list
        self.target_value = target_value

    def apply_function_single(self, col: str, x):
        if x in self.source_values:
            return self.target_value
        else:
            return x

    def apply_function_series(self, col: str, x: series_type):
        return x.replace(self.source_values, self.target_value)

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
    """
    删掉特定的列:cols
    cols=["col1"]
    ----------------
    | col1 | col2 |
    |  1   |  2   |
    |  3   |  4   |
    ---------------
     | col2 |
     |  2   |
     |  4   |
    """

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
    """
    选择特定的cols
    cols=["col1","col2"]
    ---------------------------
    | col1 | col2 |  col3  |
    |  1   |  2   |   5    |
    |  3   |  4   |   6    |
    ---------------------------
    | col1 | col2 |
    |  1   |  2   |
    |  3   |  4   |
    """

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
    对cols中空进行填充，有三个优先级
    1.fill_detail不为空,则处理fill_detail所指定的填充值，格式为fill_detail={"col1":1,"col2":"miss"}表示将col1中的空填充为1，col2中的填充为"miss"
    2.fill_detail为空时，接着看fill_mode，可选项有mean:表示均值填充，median:中位数填充，mode:众数填充(对number和category类型均可)
    3.当fill_detail和fill_mode均为空时，将number数据类型用fill_number_value填充，category数据类型用fill_category_value填充
    """

    def __init__(self, cols="all", fill_mode=None, fill_number_value=0, fill_category_value=None, fill_detail=None,
                 default_null_values=None,
                 **kwargs):
        super().__init__(cols=cols, **kwargs)
        if default_null_values is None:
            default_null_values = ["none", "nan", "np.nan", "null", "", " ", "inf", "np.inf"]
        self.fill_number_value = fill_number_value
        self.fill_category_value = fill_category_value
        self.fill_mode = fill_mode
        self.fill_detail = fill_detail
        self.default_null_values = default_null_values

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

    def apply_function_single(self, col, x):
        if str(x).lower() in self.default_null_values:
            x = self.fill_detail.get(col)
        return x

    def apply_function_series(self, col: str, x: series_type):
        fill_value = self.fill_detail.get(col)
        x_ = x.astype(str).str.lower()
        for null_value in self.default_null_values:
            x[x_ == null_value] = fill_value
        return x

    def _get_params(self) -> dict_type:
        return {"fill_detail": self.fill_detail, "default_null_values": self.default_null_values}

    def _set_params(self, params: dict):
        self.fill_detail = params["fill_detail"]
        self.default_null_values = params["default_null_values"]


class TransToCategory(PreprocessBase):
    """
    1.将cols中的数据转换为category字符类型
    2.通过map_detail=(["none", "nan", "", " ", "*", "inf", "null"], "nan")可以补充一个replace的操作，如果字符在map_detail[0]中，则替换为map_detail[1]
    """

    def __init__(self, cols="all", map_detail=(["none", "nan", "", " ", "*", "inf", "null"], "nan"), **kwargs):
        super().__init__(cols=cols, **kwargs)
        self.map_detail = map_detail

    def apply_function_single(self, col: str, x):
        x = str(x)
        if x.lower() in self.map_detail[0]:
            x = self.map_detail[1]
        return x

    def apply_function_series(self, col: str, x: series_type):
        x = x.astype(str)
        try:
            x_ = x.str.lower()
            for null_value in self.map_detail[0]:
                x[x_ == null_value] = self.map_detail[1]
            return x
        except:
            return x.apply(lambda x__: self.apply_function_single(col, x__))

    def _get_params(self) -> dict_type:
        return {"map_detail": self.map_detail}

    def _set_params(self, params: dict_type):
        self.map_detail = params["map_detail"]


class TransToFloat(PreprocessBase):
    """
    将cols中的数据转换为float类型，对于处理异常的情况用nan_fill_value填充
    """

    def __init__(self, cols="all", nan_fill_value=0, **kwargs):
        super().__init__(cols=cols, **kwargs)
        self.nan_fill_value = nan_fill_value

    def apply_function_single(self, col: str, x):
        try:
            return float(x)
        except:
            x = self.nan_fill_value
        return x

    def apply_function_series(self, col: str, x: series_type):
        try:
            x = x.astype(float)
        except:
            x = x.apply(lambda x_: self.apply_function_single(col, x_))
        return x

    def _get_params(self) -> dict_type:
        return {"nan_fill_value": self.nan_fill_value}

    def _set_params(self, params: dict):
        self.nan_fill_value = params["nan_fill_value"]


class TransToInt(PreprocessBase):
    """
    将cols中的数据转换为int类型，对于处理异常的情况用nan_fill_value填充
    """

    def __init__(self, cols="all", nan_fill_value=0, **kwargs):
        super().__init__(cols=cols, **kwargs)
        self.nan_fill_value = nan_fill_value

    def apply_function_single(self, col: str, x):
        try:
            return int(float(x))
        except:
            x = self.nan_fill_value
        return x

    def apply_function_series(self, col: str, x: series_type):
        try:
            return x.astype(float).astype(int)
        except:
            return x.apply(lambda x_: self.apply_function_single(col, x_))

    def _get_params(self) -> dict_type:
        return {"nan_fill_value": self.nan_fill_value}

    def _set_params(self, params: dict):
        self.nan_fill_value = params["nan_fill_value"]


class TransToLower(PreprocessBase):
    """
    将字符中的所有英文字符转小写
    """

    def apply_function_single(self, col: str, x):
        return str(x).lower()

    def apply_function_series(self, col: str, x: series_type):
        return x.astype(str).str.lower()

    def _get_params(self):
        return {}


class TransToUpper(PreprocessBase):
    """
    将字符中的所有英文字符转大写
    """

    def apply_function_single(self, col: str, x):
        return str(x).upper()

    def apply_function_series(self, col: str, x: series_type):
        return x.astype(str).str.lower()

    def _get_params(self):
        return {}


class CategoryMapValues(PreprocessBase):
    """
    本质也是replace操作,有两个优先级别
    1.map_detail不为空，格式为map_detail={"col1":(["a","b"],"c"),"col2":([0,-1],1)}表示将col1中的"a","b"替换为"c",将col2中的0,-1替换为1
    2.map_detail为空时，default_map生效，比如default_map=(["a","b"],"c")表示将所有cols中出现的"a","b"替换为"c"
    """

    def __init__(self, cols="all", default_map=([""], ""), map_detail=None, **kwargs):
        super().__init__(cols=cols, **kwargs)
        self.default_map = default_map
        self.map_detail = map_detail

    def apply_function_single(self, col: str, x):
        if x in self.map_detail.get(col, ([""], ""))[0]:
            return self.map_detail.get(col, ([""], ""))[1]
        else:
            return x

    def apply_function_series(self, col: str, x: series_type):
        map_values = self.map_detail.get(col, ([""], ""))[0]
        target_value = self.map_detail.get(col, ([""], ""))[1]
        return x.replace(map_values, target_value)

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
    """
    对数值数据进行盖帽操作，有三个优先级
    1.clip_detail不为空，clip_detail={"col1":(-1,1),"col2":(0,1)}表示将col1中<=-1的值设置为-1，>=1的值设置为1，将col2中<=0的值设置为0，>=1的值设置为1
    2.clip_detail为空，percent_range不为空，percent_range=(1,99)表示对所有cols，对最小的1%和最高的99%数据进行clip
    3.clip_detail和percent_range均为空时，default_clip=(0,1)表示对所有cols，<=0的设置为0，>=1的设置为1
    """

    def __init__(self, cols="all", default_clip=None, clip_detail=None, percent_range=None, **kwargs):
        """
       优先级clip_detail>percent_range>default_clip
        """
        super().__init__(cols=cols, **kwargs)
        self.default_clip = default_clip
        self.clip_detail = clip_detail
        self.percent_range = percent_range

    def apply_function_single(self, col: str, x):
        clip_detail_ = self.clip_detail.get(col, (None, None))
        if clip_detail_[0] is not None and str(x).lower() not in ["none", "nan", "null", "", "inf", "np.nan", "np.inf"]:
            x = clip_detail_[0] if x <= clip_detail_[0] else x
        if clip_detail_[1] is not None and str(x).lower() not in ["none", "nan", "null", "", "inf", "np.nan", "np.inf"]:
            x = clip_detail_[1] if x >= clip_detail_[1] else x
        return x

    def apply_function_series(self, col: str, x: series_type):
        clip_detail_ = self.clip_detail.get(col, (None, None))
        if clip_detail_[0] is not None and str(x).lower() not in ["none", "nan", "null", "", "inf", "np.nan", "np.inf"]:
            x = np.clip(x, clip_detail_[0], a_max=None)
        if clip_detail_[1] is not None and str(x).lower() not in ["none", "nan", "null", "", "inf", "np.nan", "np.inf"]:
            x = np.clip(x, a_min=None, a_max=clip_detail_[1])
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
    对cols进行最大最小归一化:(x-min)/(max-min)
    但如果max和min相等，则设置为1
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
                s[new_col] = (s[col] - min_value) / (max_value - min_value)
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
    对cols进行标准化:(x-mean)/std
    但如果std=0，则直接设置为1
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
                s[new_col] = (s[col] - mean_value) / std_value
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
    对cols进行分箱
    n_bins:分箱数
    strategy:uniform/quantile/kmeans分别表示等距/等位/kmeans聚类分箱
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
