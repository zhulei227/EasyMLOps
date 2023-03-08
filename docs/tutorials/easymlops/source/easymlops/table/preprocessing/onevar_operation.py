"""
对单个col进行的操作
"""
from .core import *


class Replace(PreprocessBase):
    """
    cols输入格式:
    1.cols="all",cols=None表示对所有columns进行操作
    2.cols=["col1","col2"]表示对col1和col2操作
    3.cols=[("col1","new_col1"),("col2","new_col2")]表示对col1和col2操作，并将结果赋值给new_col1和new_col2，并不修改原始col1和col2
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
        index = self.get_match_values_index(x, self.source_values)
        x.loc[index] = self.target_value
        return x

    def udf_get_params(self):
        return {"source_values": self.source_values, "target_value": self.target_value}

    def udf_set_params(self, params: dict_type):
        self.source_values = params["source_values"]
        self.target_value = params["target_value"]


class ClipString(PreprocessBase):
    """
    截图字符串的指定位置
    """

    def __init__(self, cols="all", default_clip_index: tuple = (0, None), clip_detail: dict = None,
                 **kwargs):
        super().__init__(cols, **kwargs)
        self.default_clip_index = default_clip_index

        self.clip_detail = clip_detail

    def udf_fit(self, s: dataframe_type, **kwargs):
        if self.clip_detail is None:
            self.clip_detail = dict()
            for col, _ in self.cols:
                self.clip_detail[col] = self.default_clip_index
        else:
            new_cols = []
            for col, new_col in self.cols:
                if col in self.clip_detail.keys():
                    new_cols.append((col, new_col))
            self.cols = new_cols
        return self

    def apply_function_series(self, col: str, x: series_type):
        clip_index = slice(*self.clip_detail[col])
        try:
            return x.astype(str).str[clip_index]
        except Exception as e:
            print("module:{},in function:{},raise the exception:{}".format(self.name, "apply_function_series", e))
            return x.astype(str).apply(lambda x_: self.apply_function_single(col, x_))

    def apply_function_single(self, col: str, x):
        clip_index = slice(*self.clip_detail[col])
        try:
            return str(x)[clip_index]
        except Exception as e:
            print("module:{},in function:{},raise the exception:{}".format(self.name, "apply_function_single", e))
            return ""

    def udf_get_params(self):
        return {"clip_detail": self.clip_detail, "default_clip_index": self.default_clip_index}

    def udf_set_params(self, params: dict):
        self.default_clip_index = params["default_clip_index"]
        self.clip_detail = params["clip_detail"]


class FillNa(PreprocessBase):
    """
    对cols中空进行填充，有三个优先级
    1.fill_detail不为空,则处理fill_detail所指定的填充值，格式为fill_detail={"col1":1,"col2":"miss"}表示将col1中的空填充为1，col2中的填充为"miss"
    2.fill_detail为空时，接着看fill_mode，可选项有mean:表示均值填充，median:中位数填充，mode:众数填充(对number和category类型均可)
    3.当fill_detail和fill_mode均为空时，将number数据类型用fill_number_value填充，category数据类型用fill_category_value填充
    """

    def __init__(self, cols="all", fill_mode=None, fill_number_value=0, fill_category_value="nan", fill_detail=None,
                 default_null_values=None,
                 **kwargs):
        super().__init__(cols=cols, **kwargs)
        if default_null_values is None:
            default_null_values = GlobalNullValues
        self.fill_number_value = fill_number_value
        self.fill_category_value = fill_category_value
        self.fill_mode = fill_mode
        self.fill_detail = fill_detail
        self.default_null_values = default_null_values

    def udf_fit(self, s: dataframe_type, **kwargs):
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
        fill_value = self.fill_detail.get(col)
        if str(x).lower() in self.default_null_values:
            x = fill_value
        return x

    def apply_function_series(self, col: str, x: series_type):
        fill_value = self.fill_detail.get(col)
        index = self.get_match_values_index(x, self.default_null_values)
        x.loc[index] = fill_value
        return x

    def udf_get_params(self) -> dict_type:
        return {"fill_detail": self.fill_detail, "default_null_values": self.default_null_values}

    def udf_set_params(self, params: dict):
        self.fill_detail = params["fill_detail"]
        self.default_null_values = params["default_null_values"]


class IsNull(PreprocessBase):
    """
    判断是否为空
    """

    def __init__(self, cols="all", default_null_values=None,
                 **kwargs):
        super().__init__(cols=cols, **kwargs)
        if default_null_values is None:
            default_null_values = GlobalNullValues
        self.default_null_values = default_null_values
        self.null_value_map = dict()
        for null_value in self.default_null_values:
            self.null_value_map[null_value] = 1

    def udf_fit(self, s: dataframe_type, **kwargs):
        new_cols = []
        for col, new_col in self.cols:
            if col == new_col:
                new_cols.append((col, col + "_is_null"))
        self.cols = new_cols
        return self

    def apply_function_single(self, col, x):
        if str(x).lower() in self.default_null_values:
            return 1
        else:
            return 0

    def apply_function_series(self, col: str, x: series_type):
        return x.map(self.null_value_map).fillna(0).astype(np.uint8)

    def udf_get_params(self) -> dict_type:
        return {"default_null_values": self.default_null_values, "null_value_map": self.null_value_map}

    def udf_set_params(self, params: dict):
        self.default_null_values = params["default_null_values"]
        self.null_value_map = params["null_value_map"]


class IsNotNull(PreprocessBase):
    """
    判断是否不为空
    """

    def __init__(self, cols="all", default_null_values=None,
                 **kwargs):
        super().__init__(cols=cols, **kwargs)
        if default_null_values is None:
            default_null_values = GlobalNullValues
        self.default_null_values = default_null_values
        self.null_value_map = dict()
        for null_value in self.default_null_values:
            self.null_value_map[null_value] = 0

    def udf_fit(self, s: dataframe_type, **kwargs):
        new_cols = []
        for col, new_col in self.cols:
            if col == new_col:
                new_cols.append((col, col + "_is_not_null"))
        self.cols = new_cols
        return self

    def apply_function_single(self, col, x):
        if str(x).lower() in self.default_null_values:
            return 0
        else:
            return 1

    def apply_function_series(self, col: str, x: series_type):
        return x.map(self.null_value_map).fillna(1).astype(np.uint8)

    def udf_get_params(self) -> dict_type:
        return {"default_null_values": self.default_null_values, "null_value_map": self.null_value_map}

    def udf_set_params(self, params: dict):
        self.default_null_values = params["default_null_values"]
        self.null_value_map = params["null_value_map"]


class TransToCategory(PreprocessBase):
    """
    将cols中的数据转换为category字符类型
    """

    def __init__(self, cols="all", **kwargs):
        super().__init__(cols=cols, **kwargs)

    def apply_function_single(self, col: str, x):
        return str(x)

    def apply_function_series(self, col: str, x: series_type):
        return x.astype(str)

    def udf_get_params(self) -> dict_type:
        return {}


class TransToFloat(PreprocessBase):

    def __init__(self, cols="all", nan_fill_value=0, **kwargs):
        super().__init__(cols=cols, **kwargs)
        self.nan_fill_value = nan_fill_value

    def apply_function_single(self, col: str, x):
        x = pd.to_numeric(x, errors="coerce")
        if "nan" in str(x).lower():
            x = self.nan_fill_value
        return x

    def apply_function_series(self, col: str, x: series_type):
        return pd.to_numeric(x, errors="coerce").fillna(self.nan_fill_value)

    def udf_get_params(self) -> dict_type:
        return {"nan_fill_value": self.nan_fill_value}

    def udf_set_params(self, params: dict):
        self.nan_fill_value = params["nan_fill_value"]


class TransToInt(PreprocessBase):
    """
    将cols中的数据转换为int类型，对于处理异常的情况用nan_fill_value填充
    """

    def __init__(self, cols="all", nan_fill_value=0, **kwargs):
        super().__init__(cols=cols, **kwargs)
        self.nan_fill_value = nan_fill_value

    def apply_function_single(self, col: str, x):
        x = pd.to_numeric(x, errors="coerce")
        if "nan" in str(x).lower():
            x = self.nan_fill_value
        return np.int64(x)

    def apply_function_series(self, col: str, x: series_type):
        return pd.to_numeric(x, errors="coerce").fillna(self.nan_fill_value).astype(np.int64)

    def udf_get_params(self) -> dict_type:
        return {"nan_fill_value": self.nan_fill_value}

    def udf_set_params(self, params: dict):
        self.nan_fill_value = params["nan_fill_value"]


class TransToLower(PreprocessBase):
    """
    将字符中的所有英文字符转小写
    """

    def apply_function_single(self, col: str, x):
        return str(x).lower()

    def apply_function_series(self, col: str, x: series_type):
        return x.astype(str).str.lower()

    def udf_get_params(self):
        return {}


class TransToUpper(PreprocessBase):
    """
    将字符中的所有英文字符转大写
    """

    def apply_function_single(self, col: str, x):
        return str(x).upper()

    def apply_function_series(self, col: str, x: series_type):
        return x.astype(str).str.lower()

    def udf_get_params(self):
        return {}


class Abs(PreprocessBase):
    """
    取绝对值
    """

    def apply_function_single(self, col: str, x):
        return abs(x)

    def apply_function_series(self, col: str, x: series_type):
        return abs(x)

    def udf_get_params(self):
        return {}


class MapValues(PreprocessBase):
    """
    基本结构表示:[(["a","b",1],"c"),
               ("[4,5)",1),
               0]
    []中:
    1.第一个表示离散型映射，把"a","b",1映射为"c"
    2.第二个表示范围内映射，把4<=x<5的值，映射为1，其中[,]表示闭区间,(,)表示开区间
    3.列表的最后一个是没有匹配上的设置的默认值
    """

    def __init__(self, cols="all", map_detail=None, default_map=None, **kwargs):
        super().__init__(cols=cols, **kwargs)
        self.default_map = default_map
        self.map_detail = map_detail

    def udf_fit(self, s: dataframe_type, **kwargs):
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

    @staticmethod
    def get_match_range_index(x, range_str):
        x_ = pd.to_numeric(x, errors="coerce")
        min_value, max_value = range_str.split(',')
        if '[' in min_value:
            min_value = min_value.strip().replace('[', '')
            min_value = float(min_value.strip())
            match_index1 = min_value <= x_
        else:
            min_value = min_value.strip().replace('(', '')
            min_value = float(min_value.strip())
            match_index1 = min_value < x_

        if ']' in max_value:
            max_value = max_value.strip().replace(']', '')
            max_value = float(max_value.strip())
            match_index2 = x_ <= max_value
        else:
            max_value = max_value.strip().replace(')', '')
            max_value = float(max_value.strip())
            match_index2 = x_ < max_value
        return match_index1 & match_index2

    def apply_function_series(self, col: str, x: series_type):
        map_detail = self.map_detail[col]
        default_value = map_detail[-1]
        if type(default_value) != tuple:
            return_default_value = True
            map_detail_ = map_detail[:-1]
        else:
            return_default_value = False
            map_detail_ = map_detail
        match_index_with_target_value = []
        # 分别获取每种映射的match_index和target_value
        # 这里注意不要逐步替换，避免同一个值被转换2次
        for match_pattern, target_value in map_detail_:
            if type(match_pattern) == list:
                match_index = self.get_match_values_index(x, match_pattern)
            else:
                match_index = self.get_match_range_index(x, match_pattern)
            match_index_with_target_value.append((match_index, target_value))

        # 定一个全False的index
        total_match_index = match_index_with_target_value[0][0] != match_index_with_target_value[0][0]
        for match_index, target_value in match_index_with_target_value:
            x.loc[match_index] = target_value
            total_match_index = total_match_index | match_index
        if return_default_value:
            x.loc[~total_match_index] = default_value  # 都未被匹配的设置默认值
        return x

    def apply_function_single(self, col: str, x):
        map_detail = self.map_detail[col]
        default_value = map_detail[-1]
        if type(default_value) != tuple:
            return_default_value = True
            map_detail_ = map_detail[:-1]
        else:
            return_default_value = False
            map_detail_ = map_detail
        for match_pattern, target_value in map_detail_:
            if type(match_pattern) == list:
                if x in match_pattern:
                    return target_value
            elif self.get_match_range_index(x, match_pattern):
                return target_value
        if return_default_value:
            return default_value
        else:
            return x

    def udf_get_params(self) -> dict_type:
        return {"map_detail": self.map_detail}

    def udf_set_params(self, params: dict):
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
        x = np.clip(pd.to_numeric(x, errors="coerce"), a_min=clip_detail_[0], a_max=clip_detail_[1])
        return x

    def apply_function_series(self, col: str, x: series_type):
        clip_detail_ = self.clip_detail.get(col, (None, None))
        x = np.clip(pd.to_numeric(x, errors="coerce"), a_min=clip_detail_[0], a_max=clip_detail_[1])
        return x

    def udf_fit(self, s: dataframe_type, **kwargs):
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

    def udf_get_params(self) -> dict_type:
        return {"clip_detail": self.clip_detail}

    def udf_set_params(self, params: dict):
        self.clip_detail = params["clip_detail"]


class MinMaxScaler(PreprocessBase):
    """
    对cols进行最大最小归一化:(x-min)/(max-min)
    但如果max和min相等，则设置为1
    """

    def __init__(self, decimals=2, **kwargs):
        super().__init__(**kwargs)
        self.min_max_detail = dict()
        self.col_types = dict()
        self.decimals = decimals

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

    def udf_fit(self, s: dataframe_type, **kwargs):
        for col, _ in self.cols:
            col_value = s[col].astype(np.float64)
            min_value = np.min(col_value)
            max_value = np.max(col_value)
            self.min_max_detail[col] = (min_value, max_value)
            self.col_types[col] = self.get_col_type(s[col])
        return self

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        for col, new_col in self.cols:
            min_value, max_value = self.min_max_detail.get(col)
            col_type = np.float64
            self._check_min_max_equal(col, min_value, max_value)
            if min_value == max_value:
                s[new_col] = 1
            else:
                s[new_col] = (s[col].astype(col_type) - col_type(min_value)) / (
                        col_type(max_value) - col_type(min_value))
                s[new_col] = np.round(s[new_col], self.decimals)
        return s

    def udf_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        for col, new_col in self.cols:
            min_value, max_value = self.min_max_detail.get(col)
            col_type = np.float64
            self._check_min_max_equal(col, min_value, max_value)
            if min_value == max_value:
                s[new_col] = 1
            else:
                s[new_col] = (col_type(s[col]) - col_type(min_value)) / (col_type(max_value) - col_type(min_value))
                s[new_col] = np.round(s[new_col], self.decimals)
        return s

    def udf_get_params(self):
        return {"min_max_detail": self.min_max_detail, "col_types": self.col_types, "decimals": self.decimals}

    def udf_set_params(self, params: dict):
        self.min_max_detail = params["min_max_detail"]
        self.col_types = params["col_types"]
        self.decimals = params["decimals"]


class Normalizer(PreprocessBase):
    """
    对cols进行标准化:(x-mean)/std
    但如果std=0，则直接设置为1
    """

    def __init__(self, decimals=2, **kwargs):
        super().__init__(**kwargs)
        self.mean_std_detail = dict()
        self.col_types = dict()
        self.decimals = decimals

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

    def udf_fit(self, s: dataframe_type, **kwargs):
        for col, _ in self.cols:
            col_value = s[col].astype(np.float64)
            mean_value = np.mean(col_value)
            std_value = np.std(col_value)
            self.mean_std_detail[col] = (mean_value, std_value)
            self.col_types[col] = self.get_col_type(s[col])
        return self

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        for col, new_col in self.cols:
            mean_value, std_value = self.mean_std_detail.get(col)
            col_type = np.float64
            self._check_std(col, std_value)
            if std_value == 0:
                s[new_col] = 1
            else:
                s[new_col] = (s[col].astype(col_type) - col_type(mean_value)) / col_type(std_value)
                s[new_col] = np.round(s[new_col], self.decimals)
        return s

    def udf_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        for col, new_col in self.cols:
            mean_value, std_value = self.mean_std_detail.get(col)
            col_type = np.float64
            self._check_std(col, std_value)
            if std_value == 0:
                s[new_col] = 1
            else:
                s[new_col] = (col_type(s[col]) - col_type(mean_value)) / col_type(std_value)
                s[new_col] = np.round(s[new_col], self.decimals)
        return s

    def udf_get_params(self):
        return {"mean_std_detail": self.mean_std_detail, "col_types": self.col_types, "decimals": self.decimals}

    def udf_set_params(self, params: dict):
        self.mean_std_detail = params["mean_std_detail"]
        self.col_types = params["col_types"]
        self.decimals = params["decimals"]


class Bins(PreprocessBase):
    """
    对cols进行分箱
    n_bins:分箱数
    strategy:uniform/quantile/kmeans分别表示等距/等位/kmeans聚类分箱
    """

    def __init__(self, n_bins=10, strategy="quantile", **kwargs):
        super().__init__(**kwargs)
        self.n_bins = n_bins
        if self.n_bins < 2 ** 8:
            self.col_type = np.uint8
        elif self.n_bins < 2 ** 16:
            self.col_type = np.uint16
        elif self.n_bins < 2 ** 32:
            self.col_type = np.uint32
        else:
            self.col_type = np.uint64
        self.strategy = strategy
        self.bin_detail = dict()

    def udf_fit(self, s: dataframe_type, **kwargs):
        from sklearn.preprocessing import KBinsDiscretizer
        for col, _ in self.cols:
            bin_model = KBinsDiscretizer(n_bins=self.n_bins, encode="ordinal", strategy=self.strategy)
            bin_model.fit(s[[col]])
            self.bin_detail[col] = bin_model
        return self

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        for col, new_col in self.cols:
            bin_model = self.bin_detail[col]
            s[new_col] = bin_model.transform(s[[col]]).astype(self.col_type)
        return s

    def transform_single(self, s: dict_type, **kwargs) -> dict_type:
        input_dataframe = pd.DataFrame([s])
        return self.transform(input_dataframe, **kwargs).to_dict("record")[0]

    def udf_get_params(self):
        return {"n_bins": self.n_bins, "strategy": self.strategy, "bin_detail": self.bin_detail,
                "col_type": self.col_type}

    def udf_set_params(self, params: dict_type):
        self.bin_detail = params["bin_detail"]
        self.n_bins = params["n_bins"]
        self.strategy = params["strategy"]
        self.col_type = params["col_type"]
