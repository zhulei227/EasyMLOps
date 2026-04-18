from .core import *

pd.options.mode.copy_on_write = True


class Replace(PreprocessBase):
    """
    对单个列进行值替换操作。
    
    将指定列中的某些值替换为目标值，常用于数据清洗和标准化。
    
    Example:
        >>> pipe = Replace(cols=["status"], source_values=["N/A", "null"], target_value="unknown")
    """
    
    def __init__(self, cols="all", source_values=None, target_value="", **kwargs):
        """
        初始化替换操作。
        
        Args:
            cols: 要操作的列，支持 "all"、列名列表或 (原列名, 新列名) 元组列表
            source_values: 要替换的值列表
            target_value: 替换后的目标值
            **kwargs: 其他父类参数
        """
        super().__init__(cols=cols, **kwargs)
        self.source_values = source_values
        if source_values is None:
            self.source_values = []
        assert type(self.source_values) == list
        self.target_value = target_value

    def apply_function_single(self, col: str, x):
        """对单个值进行替换。"""
        if x in self.source_values:
            return self.target_value
        else:
            return x

    def apply_function_series(self, col: str, x: series_type):
        """对 Series 进行批量替换。"""
        index = self.get_match_values_index(x, self.source_values)
        x.loc[index] = self.target_value
        return x

    def udf_get_params(self):
        """获取参数。"""
        return {"source_values": self.source_values, "target_value": self.target_value}

    def udf_set_params(self, params: dict_type):
        """设置参数。"""
        self.source_values = params["source_values"]
        self.target_value = params["target_value"]


class ClipString(PreprocessBase):
    """
    截取字符串的指定位置。
    
    根据指定的起始和结束索引截取字符串内容。
    
    Example:
        >>> pipe = ClipString(cols=["name"], default_clip_index=(0, 10))
    """
    
    def __init__(self, cols="all", default_clip_index: tuple = (0, None), clip_detail: dict = None,
                 **kwargs):
        """
        初始化截取字符串操作。
        
        Args:
            cols: 要操作的列
            default_clip_index: 默认截取索引，格式为 (起始, 结束)
            clip_detail: 各列的详细截取配置
            **kwargs: 其他父类参数
        """
        super().__init__(cols, **kwargs)
        self.default_clip_index = default_clip_index
        self.clip_detail = clip_detail

    def udf_fit(self, s: dataframe_type, **kwargs):
        """
        训练阶段，确定需要截取的列和索引。
        
        Args:
            s: 输入数据
            **kwargs: 其他参数
            
        Returns:
            self
        """
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
        """对 Series 进行字符串截取。"""
        clip_index = slice(*self.clip_detail[col])
        try:
            return x.astype(str).str[clip_index]
        except Exception as e:
            print("module:{},in function:{},raise the exception:{}".format(self.name, "apply_function_series", e))
            return x.astype(str).apply(lambda x_: self.apply_function_single(col, x_))

    def apply_function_single(self, col: str, x):
        """对单个值进行字符串截取。"""
        clip_index = slice(*self.clip_detail[col])
        try:
            return str(x)[clip_index]
        except Exception as e:
            print("module:{},in function:{},raise the exception:{}".format(self.name, "apply_function_single", e))
            return ""

    def udf_get_params(self):
        """获取参数。"""
        return {"clip_detail": self.clip_detail, "default_clip_index": self.default_clip_index}

    def udf_set_params(self, params: dict):
        """设置参数。"""
        self.default_clip_index = params["default_clip_index"]
        self.clip_detail = params["clip_detail"]


class FillNa(PreprocessBase):
    """
    对指定列中的空值进行填充。
    
    支持三种填充优先级：
    1. fill_detail: 指定各列的填充值，格式为 {"col1": 1, "col2": "miss"}
    2. fill_mode: 自动计算填充值，支持 mean(均值)、median(中位数)、mode(众数)
    3. 默认填充: 数值型用 fill_number_value，类别型用 fill_category_value
    
    Example:
        >>> pipe = FillNa(cols=["age", "city"], fill_mode="mean")
        >>> pipe = FillNa(cols=["age"], fill_detail={"age": 0})
    """
    
    def __init__(self, cols="all", fill_mode=None, fill_number_value=0, fill_category_value="nan", fill_detail=None,
                 default_null_values=None,
                 **kwargs):
        """
        初始化空值填充操作。
        
        Args:
            cols: 要操作的列
            fill_mode: 填充模式，可选 "mean"、"median"、"mode"
            fill_number_value: 数值型默认填充值
            fill_category_value: 类别型默认填充值
            fill_detail: 各列详细填充配置
            default_null_values: 视为空值的列表
            **kwargs: 其他父类参数
        """
        super().__init__(cols=cols, **kwargs)
        if default_null_values is None:
            default_null_values = GlobalNullValues
        self.fill_number_value = fill_number_value
        self.fill_category_value = fill_category_value
        self.fill_mode = fill_mode
        self.fill_detail = fill_detail
        self.default_null_values = default_null_values

    def udf_fit(self, s: dataframe_type, **kwargs):
        """
        训练阶段，计算各列的填充值。
        
        Args:
            s: 输入数据
            **kwargs: 其他参数
            
        Returns:
            self
        """
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
        """对单个值进行空值填充。"""
        fill_value = self.fill_detail.get(col)
        if str(x).lower() in self.default_null_values:
            x = fill_value
        return x

    def apply_function_series(self, col: str, x: series_type):
        """对 Series 进行空值填充。"""
        fill_value = self.fill_detail.get(col)
        index = self.get_match_values_index(x, self.default_null_values)
        x.loc[index] = fill_value
        return x

    def udf_get_params(self) -> dict_type:
        """获取参数。"""
        return {"fill_detail": self.fill_detail, "default_null_values": self.default_null_values}

    def udf_set_params(self, params: dict):
        """设置参数。"""
        self.fill_detail = params["fill_detail"]
        self.default_null_values = params["default_null_values"]


class IsNull(PreprocessBase):
    """
    判断指定列是否为空值。
    
    生成新的布尔列，表示原始列中的值是否为空。
    
    Example:
        >>> pipe = IsNull(cols=["age", "name"])
    """
    
    def __init__(self, cols="all", default_null_values=None,
                 **kwargs):
        """
        初始化空值判断操作。
        
        Args:
            cols: 要操作的列
            default_null_values: 视为空值的列表
            **kwargs: 其他父类参数
        """
        super().__init__(cols=cols, **kwargs)
        if default_null_values is None:
            default_null_values = GlobalNullValues
        self.default_null_values = default_null_values
        self.null_value_map = dict()
        for null_value in self.default_null_values:
            self.null_value_map[null_value] = 1

    def udf_fit(self, s: dataframe_type, **kwargs):
        """
        训练阶段，确定输出列名。
        
        Args:
            s: 输入数据
            **kwargs: 其他参数
            
        Returns:
            self
        """
        new_cols = []
        for col, new_col in self.cols:
            if col == new_col:
                new_cols.append((col, col + "_is_null"))
        self.cols = new_cols
        return self

    def apply_function_single(self, col, x):
        """判断单个值是否为空。"""
        if str(x).lower() in self.default_null_values:
            return 1
        else:
            return 0

    def apply_function_series(self, col: str, x: series_type):
        """批量判断 Series 是否为空。"""
        return x.map(self.null_value_map).fillna(0).astype(np.uint8)

    def udf_get_params(self) -> dict_type:
        """获取参数。"""
        return {"default_null_values": self.default_null_values, "null_value_map": self.null_value_map}

    def udf_set_params(self, params: dict):
        """设置参数。"""
        self.default_null_values = params["default_null_values"]
        self.null_value_map = params["null_value_map"]


class IsNotNull(PreprocessBase):
    """
    判断指定列是否不为空值。
    
    生成新的布尔列，表示原始列中的值是否非空。
    
    Example:
        >>> pipe = IsNotNull(cols=["age", "name"])
    """
    
    def __init__(self, cols="all", default_null_values=None,
                 **kwargs):
        """
        初始化非空判断操作。
        
        Args:
            cols: 要操作的列
            default_null_values: 视为空值的列表
            **kwargs: 其他父类参数
        """
        super().__init__(cols=cols, **kwargs)
        if default_null_values is None:
            default_null_values = GlobalNullValues
        self.default_null_values = default_null_values
        self.null_value_map = dict()
        for null_value in self.default_null_values:
            self.null_value_map[null_value] = 0

    def udf_fit(self, s: dataframe_type, **kwargs):
        """
        训练阶段，确定输出列名。
        
        Args:
            s: 输入数据
            **kwargs: 其他参数
            
        Returns:
            self
        """
        new_cols = []
        for col, new_col in self.cols:
            if col == new_col:
                new_cols.append((col, col + "_is_not_null"))
        self.cols = new_cols
        return self

    def apply_function_single(self, col, x):
        """判断单个值是否非空。"""
        if str(x).lower() in self.default_null_values:
            return 0
        else:
            return 1

    def apply_function_series(self, col: str, x: series_type):
        """批量判断 Series 是否非空。"""
        return x.map(self.null_value_map).fillna(1).astype(np.uint8)

    def udf_get_params(self) -> dict_type:
        return {"default_null_values": self.default_null_values, "null_value_map": self.null_value_map}

    def udf_set_params(self, params: dict):
        self.default_null_values = params["default_null_values"]
        self.null_value_map = params["null_value_map"]


class TransToCategory(PreprocessBase):
    """
    将指定列的数据转换为类别（category）类型。
    
    将数值或其他类型的列转换为字符串类别类型，常用于分类特征。
    
    Example:
        >>> pipe = TransToCategory(cols=["status", "type"])
    """
    
    def __init__(self, cols="all", **kwargs):
        """
        初始化类型转换操作。
        
        Args:
            cols: 要操作的列
            **kwargs: 其他父类参数
        """
        super().__init__(cols=cols, **kwargs)

    def apply_function_single(self, col: str, x):
        """将单个值转换为字符串。"""
        return str(x)

    def apply_function_series(self, col: str, x: series_type):
        """将 Series 转换为字符串类型。"""
        return x.astype(str)

    def udf_get_params(self) -> dict_type:
        """获取参数。"""
        return {}


class TransToFloat(PreprocessBase):
    """
    将指定列的数据转换为浮点数（float）类型。
    
    将字符串或其他数值类型转换为浮点数，支持异常值处理。
    
    Example:
        >>> pipe = TransToFloat(cols=["price", "score"])
    """
    
    def __init__(self, cols="all", nan_fill_value=0, **kwargs):
        """
        初始化浮点数转换操作。
        
        Args:
            cols: 要操作的列
            nan_fill_value: 转换失败时的填充值
            **kwargs: 其他父类参数
        """
        super().__init__(cols=cols, **kwargs)
        self.nan_fill_value = nan_fill_value

    def apply_function_single(self, col: str, x):
        """将单个值转换为浮点数。"""
        x = pd.to_numeric(x, errors="coerce")
        if "nan" in str(x).lower():
            x = self.nan_fill_value
        return x

    def apply_function_series(self, col: str, x: series_type):
        """将 Series 转换为浮点数类型。"""
        return pd.to_numeric(x, errors="coerce").fillna(self.nan_fill_value)

    def udf_get_params(self) -> dict_type:
        """获取参数。"""
        return {"nan_fill_value": self.nan_fill_value}

    def udf_set_params(self, params: dict):
        """设置参数。"""
        self.nan_fill_value = params["nan_fill_value"]


class TransToInt(PreprocessBase):
    """
    将指定列的数据转换为整数（int）类型。
    
    将字符串或其他数值类型转换为整数，支持异常值处理。
    
    Example:
        >>> pipe = TransToInt(cols=["count", "age"])
    """
    
    def __init__(self, cols="all", nan_fill_value=0, **kwargs):
        """
        初始化整数转换操作。
        
        Args:
            cols: 要操作的列
            nan_fill_value: 转换失败时的填充值
            **kwargs: 其他父类参数
        """
        super().__init__(cols=cols, **kwargs)
        self.nan_fill_value = nan_fill_value

    def apply_function_single(self, col: str, x):
        """将单个值转换为整数。"""
        x = pd.to_numeric(x, errors="coerce")
        if "nan" in str(x).lower():
            x = self.nan_fill_value
        return np.int64(x)

    def apply_function_series(self, col: str, x: series_type):
        """将 Series 转换为整数类型。"""
        return pd.to_numeric(x, errors="coerce").fillna(self.nan_fill_value).astype(np.int64)

    def udf_get_params(self) -> dict_type:
        """获取参数。"""
        return {"nan_fill_value": self.nan_fill_value}

    def udf_set_params(self, params: dict):
        """设置参数。"""
        self.nan_fill_value = params["nan_fill_value"]


class TransToLower(PreprocessBase):
    """
    将指定列的英文字符转换为小写。
    
    将字符串列中的所有英文字母转换为小写形式。
    
    Example:
        >>> pipe = TransToLower(cols=["name", "city"])
    """
    
    def apply_function_single(self, col: str, x):
        """将单个值转换为小写。"""
        return str(x).lower()

    def apply_function_series(self, col: str, x: series_type):
        """将 Series 转换为小写。"""
        return x.astype(str).str.lower()

    def udf_get_params(self):
        """获取参数。"""
        return {}


class TransToUpper(PreprocessBase):
    """
    将指定列的英文字符转换为大写。
    
    将字符串列中的所有英文字母转换为大写形式。
    
    Example:
        >>> pipe = TransToUpper(cols=["name", "city"])
    """
    
    def apply_function_single(self, col: str, x):
        """将单个值转换为大写。"""
        return str(x).upper()

    def apply_function_series(self, col: str, x: series_type):
        """将 Series 转换为大写。"""
        return x.astype(str).str.lower()

    def udf_get_params(self):
        """获取参数。"""
        return {}


class Abs(PreprocessBase):
    """
    计算指定列数值的绝对值。
    
    对数值列进行绝对值运算，常用于处理可能有负数的特征。
    
    Example:
        >>> pipe = Abs(cols=["temperature", "distance"])
    """
    
    def apply_function_single(self, col: str, x):
        """计算单个值的绝对值。"""
        return abs(x)

    def apply_function_series(self, col: str, x: series_type):
        """计算 Series 的绝对值。"""
        return abs(x)

    def udf_get_params(self):
        """获取参数。"""
        return {}


class MapValues(PreprocessBase):
    """
    根据映射规则转换列的值。
    
    支持两种映射方式：
    1. 离散值映射: {"a": 1, "b": 2} 或 [("a", "b", 1), ...]
    2. 范围映射: "[4,5)" 表示 4 <= x < 5 的值映射为指定值
    
    Example:
        >>> pipe = MapValues(cols=["score"], map_detail={"score": [("[60,70)", 1], ("[70,80)", 2), 0]})
    """
    
    def __init__(self, cols="all", map_detail=None, default_map=None, **kwargs):
        """
        初始化值映射操作。
        
        Args:
            cols: 要操作的列
            map_detail: 各列的映射规则
            default_map: 默认映射规则
            **kwargs: 其他父类参数
        """
        super().__init__(cols=cols, **kwargs)
        self.default_map = default_map
        self.map_detail = map_detail

    def udf_fit(self, s: dataframe_type, **kwargs):
        """
        训练阶段，处理映射配置。
        
        Args:
            s: 输入数据
            **kwargs: 其他参数
            
        Returns:
            self
        """
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
        """
        根据范围字符串获取匹配索引。
        
        Args:
            x: 输入数据
            range_str: 范围字符串，如 "[4,5)"
            
        Returns:
            布尔索引
        """
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
        """根据映射规则批量转换 Series。"""
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
        """根据映射规则转换单个值。"""
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
        """获取参数。"""
        return {"map_detail": self.map_detail}

    def udf_set_params(self, params: dict):
        """设置参数。"""
        self.map_detail = params["map_detail"]


class Clip(PreprocessBase):
    """
    对数值数据进行盖帽（clip）操作。
    
    将数值限制在指定范围内，支持三种优先级：
    1. clip_detail: 指定各列的clip范围，如 {"col1": (-1, 1), "col2": (0, 1)}
    2. percent_range: 根据百分位计算clip范围，如 (1, 99)
    3. default_clip: 默认clip范围，如 (0, 1)
    
    Example:
        >>> pipe = Clip(cols=["price"], default_clip=(0, 1000))
    """
    
    def __init__(self, cols="all", default_clip=None, clip_detail=None, percent_range=None, **kwargs):
        """
        初始化盖帽操作。
        
        Args:
            cols: 要操作的列
            default_clip: 默认clip范围 (min, max)
            clip_detail: 各列详细clip配置
            percent_range: 百分位范围
            **kwargs: 其他父类参数
        """
        super().__init__(cols=cols, **kwargs)
        self.default_clip = default_clip
        self.clip_detail = clip_detail
        self.percent_range = percent_range

    def apply_function_single(self, col: str, x):
        """对单个值进行盖帽。"""
        clip_detail_ = self.clip_detail.get(col, (None, None))
        x = np.clip(pd.to_numeric(x, errors="coerce"), a_min=clip_detail_[0], a_max=clip_detail_[1])
        return x

    def apply_function_series(self, col: str, x: series_type):
        """对 Series 进行盖帽。"""
        clip_detail_ = self.clip_detail.get(col, (None, None))
        x = np.clip(pd.to_numeric(x, errors="coerce"), a_min=clip_detail_[0], a_max=clip_detail_[1])
        return x

    def udf_fit(self, s: dataframe_type, **kwargs):
        """
        训练阶段，计算各列的clip范围。
        
        Args:
            s: 输入数据
            **kwargs: 其他参数
            
        Returns:
            self
        """
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
        """获取参数。"""
        return {"clip_detail": self.clip_detail}

    def udf_set_params(self, params: dict):
        """设置参数。"""
        self.clip_detail = params["clip_detail"]


class MinMaxScaler(PreprocessBase):
    """
    对指定列进行最大最小归一化。
    
    计算公式: (x - min) / (max - min)
    如果 max == min，则结果设为 1。
    
    Example:
        >>> pipe = MinMaxScaler(cols=["price", "score"])
    """
    
    def __init__(self, decimals=2, **kwargs):
        """
        初始化归一化操作。
        
        Args:
            decimals: 保留小数位数
            **kwargs: 其他父类参数
        """
        super().__init__(**kwargs)
        self.min_max_detail = dict()
        self.col_types = dict()
        self.decimals = decimals

    def _check_min_max_equal(self, col, min_value, max_value):
        """检查最小值和最大值是否相等。"""
        if min_value == max_value:
            print("({}), in  column \033[1;43m{}\033[0m ,min value and max value has the same value:{},"
                  "the finally result will be set 1".format(self.name, col, min_value))

    def show_detail(self):
        """显示归一化的详细信息。"""
        data = []
        for col, value in self.min_max_detail.items():
            min_value, max_value = value
            data.append([col, min_value, max_value])
        return pd.DataFrame(data=data, columns=["col", "min_value", "max_value"])

    def udf_fit(self, s: dataframe_type, **kwargs):
        """
        训练阶段，计算各列的最小值和最大值。
        
        Args:
            s: 输入数据
            **kwargs: 其他参数
            
        Returns:
            self
        """
        for col, _ in self.cols:
            col_value = s[col].astype(np.float64)
            min_value = np.min(col_value)
            max_value = np.max(col_value)
            self.min_max_detail[col] = (min_value, max_value)
            self.col_types[col] = self.get_col_type(s[col])
        return self

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        """对数据进行归一化转换。"""
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
        """对单条数据进行归一化转换。"""
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
        """获取参数。"""
        return {"min_max_detail": self.min_max_detail, "col_types": self.col_types, "decimals": self.decimals}

    def udf_set_params(self, params: dict):
        """设置参数。"""
        self.min_max_detail = params["min_max_detail"]
        self.col_types = params["col_types"]
        self.decimals = params["decimals"]


class Normalizer(PreprocessBase):
    """
    对指定列进行标准化（Z-score标准化）。
    
    计算公式: (x - mean) / std
    如果 std == 0，则结果设为 1。
    
    Example:
        >>> pipe = Normalizer(cols=["price", "score"])
    """
    
    def __init__(self, decimals=2, **kwargs):
        """
        初始化标准化操作。
        
        Args:
            decimals: 保留小数位数
            **kwargs: 其他父类参数
        """
        super().__init__(**kwargs)
        self.mean_std_detail = dict()
        self.col_types = dict()
        self.decimals = decimals

    def _check_std(self, col, std):
        """检查标准差是否为0。"""
        if std == 0:
            print("({}), in  column \033[1;43m{}\033[0m ,the std is 0,"
                  "the finally result will be set 1".format(self.name, col))

    def show_detail(self):
        """显示标准化的详细信息。"""
        data = []
        for col, value in self.mean_std_detail.items():
            mean_value, std_value = value
            data.append([col, mean_value, std_value])
        return pd.DataFrame(data=data, columns=["col", "mean_value", "std_value"])

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        """对数据进行标准化转换。"""
        for col, new_col in self.cols:
            mean_value, std_value = self.mean_std_detail.get(col)
            col_type = np.float64
            if kwargs.get("show_process") is True:
                self._check_std(col, std_value)
            if std_value == 0:
                s[new_col] = 1
            else:
                s[new_col] = (s[col].astype(col_type) - col_type(mean_value)) / col_type(std_value)
                s[new_col] = np.round(s[new_col], self.decimals)
        return s

    def udf_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        """对单条数据进行标准化转换。"""
        for col, new_col in self.cols:
            mean_value, std_value = self.mean_std_detail.get(col)
            col_type = np.float64
            if kwargs.get("show_process") is True:
                self._check_std(col, std_value)
            if std_value == 0:
                s[new_col] = 1
            else:
                s[new_col] = (col_type(s[col]) - col_type(mean_value)) / col_type(std_value)
                s[new_col] = np.round(s[new_col], self.decimals)
        return s

    def udf_get_params(self):
        """获取参数。"""
        return {"mean_std_detail": self.mean_std_detail, "col_types": self.col_types, "decimals": self.decimals}

    def udf_set_params(self, params: dict):
        """设置参数。"""
        self.mean_std_detail = params["mean_std_detail"]
        self.col_types = params["col_types"]
        self.decimals = params["decimals"]


class Bins(PreprocessBase):
    """
    对指定列进行分箱操作。
    
    将连续变量离散化，支持三种分箱策略：
    - uniform: 等距分箱
    - quantile: 等位分箱（每箱样本数相等）
    - kmeans: K-Means聚类分箱
    
    Example:
        >>> pipe = Bins(cols=["age"], n_bins=5, strategy="quantile")
    """
    
    def __init__(self, n_bins=10, strategy="quantile", **kwargs):
        """
        初始化分箱操作。
        
        Args:
            n_bins: 分箱数量
            strategy: 分箱策略，可选 "uniform"、"quantile"、"kmeans"
            **kwargs: 其他父类参数
        """
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
        """
        训练阶段，构建分箱模型。
        
        Args:
            s: 输入数据
            **kwargs: 其他参数
            
        Returns:
            self
        """
        from sklearn.preprocessing import KBinsDiscretizer
        for col, _ in self.cols:
            bin_model = KBinsDiscretizer(n_bins=self.n_bins, encode="ordinal", strategy=self.strategy)
            bin_model.fit(s[[col]])
            self.bin_detail[col] = bin_model
        return self

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        """对数据进行分箱转换。"""
        for col, new_col in self.cols:
            bin_model = self.bin_detail[col]
            s[new_col] = bin_model.transform(s[[col]]).astype(self.col_type)
        return s

    def transform_single(self, s: dict_type, **kwargs) -> dict_type:
        """对单条数据进行分箱转换。"""
        input_dataframe = pd.DataFrame([s])
        return self.transform(input_dataframe, **kwargs).to_dict("records")[0]

    def udf_get_params(self):
        """获取参数。"""
        return {"n_bins": self.n_bins, "strategy": self.strategy, "bin_detail": self.bin_detail,
                "col_type": self.col_type}

    def udf_set_params(self, params: dict_type):
        """设置参数。"""
        self.bin_detail = params["bin_detail"]
        self.n_bins = params["n_bins"]
        self.strategy = params["strategy"]
        self.col_type = params["col_type"]


class Tanh(PreprocessBase):
    """
    对指定列应用 Tanh 激活函数。
    
    计算公式: tanh(x) = (e^x - e^-x) / (e^x + e^-x)
    输出范围: (-1, 1)
    
    Example:
        >>> pipe = Tanh(cols=["score"])
    """
    
    def __init__(self, cols="all", prefix="tanh", **kwargs):
        """
        初始化 Tanh 激活操作。
        
        Args:
            cols: 要操作的列
            prefix: 输出列前缀
            **kwargs: 其他父类参数
        """
        super().__init__(cols=cols, prefix=prefix, **kwargs)

    def apply_function_single(self, col: str, x):
        """对单个值应用 Tanh 函数。"""
        return np.tanh(x)

    def apply_function_series(self, col: str, x: series_type):
        """对 Series 应用 Tanh 函数。"""
        return np.tanh(x)

    def udf_fit(self, s: dataframe_type, **kwargs):
        """训练阶段（无需训练，直接返回self）。"""
        return self

    def udf_get_params(self):
        """获取参数。"""
        return {}


class Relu(PreprocessBase):
    """
    对指定列应用 ReLU 激活函数。
    
    计算公式: max(0, x)
    输出范围: [0, +∞)
    
    Example:
        >>> pipe = Relu(cols=["score"])
    """
    
    def __init__(self, cols="all", prefix="relu", **kwargs):
        """
        初始化 ReLU 激活操作。
        
        Args:
            cols: 要操作的列
            prefix: 输出列前缀
            **kwargs: 其他父类参数
        """
        super().__init__(cols=cols, prefix=prefix, **kwargs)

    def apply_function_single(self, col: str, x):
        """对单个值应用 ReLU 函数。"""
        return np.where(x < 0, 0, x)

    def apply_function_series(self, col: str, x: series_type):
        """对 Series 应用 ReLU 函数。"""
        return np.where(x < 0, 0, x)

    def udf_fit(self, s: dataframe_type, **kwargs):
        """训练阶段（无需训练，直接返回self）。"""
        return self

    def udf_get_params(self):
        return {}


class Sigmoid(PreprocessBase):
    """
    对指定列应用 Sigmoid 激活函数。
    
    计算公式: sigmoid(x) = 1 / (1 + e^-x)
    输出范围: (0, 1)
    
    Example:
        >>> pipe = Sigmoid(cols=["score"])
    """
    
    def __init__(self, cols="all", prefix="sigmoid", **kwargs):
        """
        初始化 Sigmoid 激活操作。
        
        Args:
            cols: 要操作的列
            prefix: 输出列前缀
            **kwargs: 其他父类参数
        """
        super().__init__(cols=cols, prefix=prefix, **kwargs)

    def apply_function_single(self, col: str, x):
        """对单个值应用 Sigmoid 函数。"""
        return 1 / (1 + np.exp(-x))

    def apply_function_series(self, col: str, x: series_type):
        """对 Series 应用 Sigmoid 函数。"""
        return 1 / (1 + np.exp(-x))

    def udf_fit(self, s: dataframe_type, **kwargs):
        """训练阶段（无需训练，直接返回self）。"""
        return self

    def udf_get_params(self):
        """获取参数。"""
        return {}


class Swish(PreprocessBase):
    """
    对指定列应用 Swish 激活函数。
    
    计算公式: swish(x) = x * sigmoid(x) = x / (1 + e^-x)
    Google Brain 提出的自门控激活函数。
    
    Example:
        >>> pipe = Swish(cols=["score"])
    """
    
    def __init__(self, cols="all", prefix="swish", **kwargs):
        """
        初始化 Swish 激活操作。
        
        Args:
            cols: 要操作的列
            prefix: 输出列前缀
            **kwargs: 其他父类参数
        """
        super().__init__(cols=cols, prefix=prefix, **kwargs)

    def apply_function_single(self, col: str, x):
        """对单个值应用 Swish 函数。"""
        return x / (1 + np.exp(-x))

    def apply_function_series(self, col: str, x: series_type):
        """对 Series 应用 Swish 函数。"""
        return x / (1 + np.exp(-x))

    def udf_fit(self, s: dataframe_type, **kwargs):
        """训练阶段（无需训练，直接返回self）。"""
        return self

    def udf_get_params(self):
        """获取参数。"""
        return {}


class DateMonthInfo(PreprocessBase):
    """
    从日期列中提取月份信息。
    
    解析日期字符串，提取月份部分（1-12）。
    
    Example:
        >>> pipe = DateMonthInfo(cols=["date"])
    """
    
    def __init__(self, cols="all", prefix="date_month", **kwargs):
        """
        初始化月份提取操作。
        
        Args:
            cols: 要操作的列
            prefix: 输出列前缀
            **kwargs: 其他父类参数
        """
        super().__init__(cols=cols, prefix=prefix, **kwargs)

    def apply_function_single(self, col: str, x):
        """从单个日期值中提取月份。"""
        # 默认格式为"xxxx-xx-xx xx:xx:xx"
        x = str(x)[:19]
        if len(x) < 19:
            return 0
        # 切割日期和时间
        arr = x.split(" ")
        if len(arr) != 2:
            return 0
        day = arr[0]
        # 切割年月日
        day_arr = day.split("-")
        if len(day_arr) != 3:
            date_month = 0
        else:
            try:
                date_month = int(float(day_arr[1]))
            except:
                date_month = 0
        return date_month

    def apply_function_series(self, col: str, x: series_type):
        """批量从日期 Series 中提取月份。"""
        return x.apply(lambda x_: self.apply_function_single(col, x_))

    def udf_fit(self, s: dataframe_type, **kwargs):
        """训练阶段（无需训练，直接返回self）。"""
        return self

    def udf_get_params(self):
        """获取参数。"""
        return {}


class DateHourInfo(PreprocessBase):
    """
    从日期列中提取小时信息。
    
    解析日期字符串，提取小时部分（0-23）。
    
    Example:
        >>> pipe = DateHourInfo(cols=["date"])
    """
    
    def __init__(self, cols="all", prefix="date_hour", **kwargs):
        """
        初始化小时提取操作。
        
        Args:
            cols: 要操作的列
            prefix: 输出列前缀
            **kwargs: 其他父类参数
        """
        super().__init__(cols=cols, prefix=prefix, **kwargs)

    def apply_function_single(self, col: str, x):
        """从单个日期值中提取小时。"""
        # 默认格式为"xxxx-xx-xx xx:xx:xx"
        x = str(x)[:19]
        if len(x) < 19:
            return 0
        # 切割日期和时间
        arr = x.split(" ")
        if len(arr) != 2:
            return 0
        time = arr[1]
        # 切割时间
        time_arr = time.split(":")
        if len(time_arr) != 3:
            date_hour = 0
        else:
            try:
                date_hour = int(float(time_arr[0]))
            except:
                date_hour = 0
        return date_hour

    def apply_function_series(self, col: str, x: series_type):
        """批量从日期 Series 中提取小时。"""
        return x.apply(lambda x_: self.apply_function_single(col, x_))

    def udf_fit(self, s: dataframe_type, **kwargs):
        """训练阶段（无需训练，直接返回self）。"""
        return self

    def udf_get_params(self):
        """获取参数。"""
        return {}


class DateMinuteInfo(PreprocessBase):
    """
    从日期列中提取分钟信息。
    
    解析日期字符串，提取分钟部分（0-59）。
    
    Example:
        >>> pipe = DateMinuteInfo(cols=["date"])
    """
    
    def __init__(self, cols="all", prefix="date_minute", **kwargs):
        """
        初始化分钟提取操作。
        
        Args:
            cols: 要操作的列
            prefix: 输出列前缀
            **kwargs: 其他父类参数
        """
        super().__init__(cols=cols, prefix=prefix, **kwargs)

    def apply_function_single(self, col: str, x):
        """从单个日期值中提取分钟。"""
        # 默认格式为"xxxx-xx-xx xx:xx:xx"
        x = str(x)[:19]
        if len(x) < 19:
            return 0
        # 切割日期和时间
        arr = x.split(" ")
        if len(arr) != 2:
            return 0
        time = arr[1]
        # 切割时间
        time_arr = time.split(":")
        if len(time_arr) != 3:
            date_minute = 0
        else:
            try:
                date_minute = int(float(time_arr[1]))
            except:
                date_minute = 0
        return date_minute

    def apply_function_series(self, col: str, x: series_type):
        """批量从日期 Series 中提取分钟。"""
        return x.apply(lambda x_: self.apply_function_single(col, x_))

    def udf_fit(self, s: dataframe_type, **kwargs):
        """训练阶段（无需训练，直接返回self）。"""
        return self

    def udf_get_params(self):
        """获取参数。"""
        return {}


class DateTotalMinuteInfo(PreprocessBase):
    """
    从日期列中提取累积分钟数。
    
    解析日期字符串，计算从00:00开始的累积分钟数。
    计算公式: hour * 60 + minute
    
    Example:
        >>> pipe = DateTotalMinuteInfo(cols=["date"])
    """
    
    def __init__(self, cols="all", prefix="date_total_minute", **kwargs):
        """
        初始化累积分钟提取操作。
        
        Args:
            cols: 要操作的列
            prefix: 输出列前缀
            **kwargs: 其他父类参数
        """
        super().__init__(cols=cols, prefix=prefix, **kwargs)

    def apply_function_single(self, col: str, x):
        """从单个日期值中提取累积分钟。"""
        # 默认格式为"xxxx-xx-xx xx:xx:xx"
        x = str(x)[:19]
        if len(x) < 19:
            return 0
        # 切割日期和时间
        arr = x.split(" ")
        if len(arr) != 2:
            return 0
        time = arr[1]
        # 切割时间
        time_arr = time.split(":")
        if len(time_arr) != 3:
            date_hour = 0
            date_minute = 0
        else:
            try:
                date_hour = int(float(time_arr[0]))
                date_minute = int(float(time_arr[1]))
            except:
                date_hour = 0
                date_minute = 0
        return 60 * date_hour + date_minute

    def apply_function_series(self, col: str, x: series_type):
        """批量从日期 Series 中提取累积分钟。"""
        return x.apply(lambda x_: self.apply_function_single(col, x_))

    def udf_fit(self, s: dataframe_type, **kwargs):
        """训练阶段（无需训练，直接返回self）。"""
        return self

    def udf_get_params(self):
        """获取参数。"""
        return {}
