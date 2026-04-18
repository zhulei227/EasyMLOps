"""
对两个变量的操作,输出值为1个
"""
from .core import *


class BiOperationBase(PreprocessBase):
    """
    二元变量操作基类。
    
    支持对两个变量进行运算操作，可指定左右操作数来自列或固定值。
    
    Example:
        >>> pipe = Add(left_col_name="price", right_col_name="tax", output_col_name="total")
    """
    
    def __init__(self, left_col_name=None, right_col_name=None, left_value=None, right_value=None, output_col_name=None,
                 operate_name="", **kwargs):
        """
        初始化二元操作。
        
        Args:
            left_col_name: 左侧操作变量的列名
            right_col_name: 右侧操作变量的列名
            left_value: 左侧操作值，当 left_col_name=None 时生效
            right_value: 右侧操作值，当 right_col_name=None 时生效
            output_col_name: 操作后输出的列名
            operate_name: 操作名称
            **kwargs: 其他父类参数
        """
        super().__init__(**kwargs)
        self.left_col_name = left_col_name
        self.right_col_name = right_col_name
        self.left_value = left_value
        self.right_value = right_value
        self.left_col_type = None
        self.right_col_type = None
        self.output_col_name = output_col_name
        self.operate_name = operate_name
        if self.output_col_name is None:
            if left_col_name is not None and right_col_name is None:
                self.output_col_name = left_col_name
            elif left_col_name is None and right_col_name is not None:
                self.output_col_name = right_col_name
            elif left_col_name is not None and right_col_name is not None:
                self.output_col_name = left_col_name + "_{}_".format(self.operate_name) + right_col_name

    def udf_fit(self, s: dataframe_type, **kwargs):
        """训练阶段，记录数据类型。"""
        if self.left_col_name is not None:
            self.left_col_type = self.get_col_type(s[self.left_col_name].dtype)
        if self.right_col_name is not None:
            self.right_col_type = self.get_col_type(s[self.right_col_name].dtype)
        return self

    @staticmethod
    def apply_function_single_bi(left_x, right_x):
        """对单个值进行二元运算。"""
        raise Exception("need to implement!")

    @staticmethod
    def apply_function_series_bi(left_x, right_x):
        """对 Series 进行二元运算。"""
        raise Exception("need to implement!")

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        """对数据进行二元运算转换。"""
        # 获取左侧值
        if self.left_col_type is None:
            left_value_s = self.right_col_type(self.left_value)
        else:
            left_value_s = s[self.left_col_name].astype(self.left_col_type)
        # 获取右侧值
        if self.right_col_type is None:
            right_value_s = self.left_col_type(self.right_value)
        else:
            right_value_s = s[self.right_col_name].astype(self.right_col_type)

        s[self.output_col_name] = self.apply_function_series_bi(left_value_s, right_value_s)
        return s

    def udf_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        """对单条数据进行二元运算转换。"""
        # 获取左侧值
        if self.left_col_type is None:
            left_value_s = self.right_col_type(self.left_value)
        else:
            left_value_s = self.left_col_type(s[self.left_col_name])
        # 获取右侧值
        if self.right_col_type is None:
            right_value_s = self.left_col_type(self.right_value)
        else:
            right_value_s = self.right_col_type(s[self.right_col_name])
        s[self.output_col_name] = self.apply_function_single_bi(left_value_s, right_value_s)
        return s

    def udf_get_params(self):
        """获取参数。"""
        return {"output_col_name": self.output_col_name, "right_col_type": self.right_col_type,
                "left_col_type": self.left_col_type, "right_value": self.right_value, "left_value": self.left_value,
                "right_col_name": self.right_col_name, "left_col_name": self.left_col_name,
                "operate_name": self.operate_name}

    def udf_set_params(self, params: dict):
        """设置参数。"""
        self.left_col_name = params["left_col_name"]
        self.right_col_name = params["right_col_name"]
        self.left_value = params["left_value"]
        self.right_value = params["right_value"]
        self.left_col_type = params["left_col_type"]
        self.right_col_type = params["right_col_type"]
        self.output_col_name = params["output_col_name"]
        self.operate_name = params["operate_name"]


class Add(BiOperationBase):
    """
    加法操作。
    
    支持数值的加法和字符串的拼接。
    
    Example:
        >>> pipe = Add(left_col_name="price", right_col_name="tax", output_col_name="total")
    """
    
    def __init__(self, **kwargs):
        """初始化加法操作。"""
        super().__init__(operate_name="add", **kwargs)

    @staticmethod
    def apply_function_series_bi(left_x, right_x):
        """对 Series 进行加法运算。"""
        return left_x + right_x

    @staticmethod
    def apply_function_single_bi(left_x, right_x):
        """对单个值进行加法运算。"""
        return left_x + right_x

    def udf_get_params(self):
        """获取参数。"""
        return {}


class Subtract(BiOperationBase):
    """
    减法操作。
    
    支持数值之间的减法运算。
    
    Example:
        >>> pipe = Subtract(left_col_name="total", right_col_name="cost", output_col_name="profit")
    """
    
    def __init__(self, **kwargs):
        """初始化减法操作。"""
        super().__init__(operate_name="subtract", **kwargs)

    @staticmethod
    def apply_function_series_bi(left_x, right_x):
        """对 Series 进行减法运算。"""
        return left_x - right_x

    @staticmethod
    def apply_function_single_bi(left_x, right_x):
        """对单个值进行减法运算。"""
        return left_x - right_x

    def udf_get_params(self):
        """获取参数。"""
        return {}


class Multiply(BiOperationBase):
    """
    乘法操作。
    
    支持数值之间的乘法运算。
    
    Example:
        >>> pipe = Multiply(left_col_name="price", right_col_name="quantity", output_col_name="total")
    """
    
    def __init__(self, **kwargs):
        """初始化乘法操作。"""
        super().__init__(operate_name="multiply", **kwargs)

    @staticmethod
    def apply_function_series_bi(left_x, right_x):
        """对 Series 进行乘法运算。"""
        return left_x * right_x

    @staticmethod
    def apply_function_single_bi(left_x, right_x):
        """对单个值进行乘法运算。"""
        return left_x * right_x

    def udf_get_params(self):
        """获取参数。"""
        return {}


class Divide(BiOperationBase):
    """
    除法操作。
    
    支持数值之间的除法运算。
    
    Example:
        >>> pipe = Divide(left_col_name="total", right_col_name="count", output_col_name="average")
    """
    
    def __init__(self, **kwargs):
        """初始化除法操作。"""
        super().__init__(operate_name="divide", **kwargs)

    @staticmethod
    def apply_function_series_bi(left_x, right_x):
        """对 Series 进行除法运算。"""
        return left_x / right_x

    @staticmethod
    def apply_function_single_bi(left_x, right_x):
        """对单个值进行除法运算。"""
        return left_x / right_x

    def udf_get_params(self):
        """获取参数。"""
        return {}


class DivideExact(BiOperationBase):
    """
    整除操作。
    
    支持数值之间的整除运算。
    
    Example:
        >>> pipe = DivideExact(left_col_name="total", right_col_name="count", output_col_name="whole")
    """
    
    def __init__(self, **kwargs):
        """初始化整除操作。"""
        super().__init__(operate_name="divide_exact", **kwargs)

    @staticmethod
    def apply_function_series_bi(left_x, right_x):
        """对 Series 进行整除运算。"""
        return left_x // right_x

    @staticmethod
    def apply_function_single_bi(left_x, right_x):
        """对单个值进行整除运算。"""
        return left_x // right_x

    def udf_get_params(self):
        """获取参数。"""
        return {}


class Mod(BiOperationBase):
    """
    取模操作。
    
    支持数值之间的取模运算。
    
    Example:
        >>> pipe = Mod(left_col_name="total", right_col_name="divisor", output_col_name="remainder")
    """
    
    def __init__(self, **kwargs):
        """初始化取模操作。"""
        super().__init__(operate_name="mod", **kwargs)

    @staticmethod
    def apply_function_series_bi(left_x, right_x):
        """对 Series 进行取模运算。"""
        return left_x // 1 % right_x

    @staticmethod
    def apply_function_single_bi(left_x, right_x):
        """对单个值进行取模运算。"""
        return left_x // 1 % right_x

    def udf_get_params(self):
        """获取参数。"""
        return {}


class Equal(BiOperationBase):
    """
    相等判断操作。
    
    判断两个值是否相等，相等返回1，否则返回0。
    
    Example:
        >>> pipe = Equal(left_col_name="status", right_col_name="expected", output_col_name="is_equal")
    """
    
    def __init__(self, **kwargs):
        """初始化相等判断操作。"""
        super().__init__(operate_name="equal", **kwargs)

    @staticmethod
    def apply_function_series_bi(left_x, right_x):
        """判断 Series 是否相等。"""
        return np.where(left_x == right_x, 1, 0)

    @staticmethod
    def apply_function_single_bi(left_x, right_x):
        """判断单个值是否相等。"""
        return 1 if left_x == right_x else 0

    def udf_get_params(self):
        """获取参数。"""
        return {}


class GreaterThan(BiOperationBase):
    """
    大于判断操作。
    
    判断左侧值是否大于右侧值，大于返回1，否则返回0。
    
    Example:
        >>> pipe = GreaterThan(left_col_name="score", right_col_name="threshold", output_col_name="pass")
    """
    
    def __init__(self, **kwargs):
        """初始化大于判断操作。"""
        super().__init__(operate_name="greater_than", **kwargs)

    @staticmethod
    def apply_function_series_bi(left_x, right_x):
        """判断 Series 是否大于。"""
        return np.where(left_x > right_x, 1, 0)

    @staticmethod
    def apply_function_single_bi(left_x, right_x):
        """判断单个值是否大于。"""
        return 1 if left_x > right_x else 0

    def udf_get_params(self):
        """获取参数。"""
        return {}


class GreaterEqualThan(BiOperationBase):
    """
    大于等于判断操作。
    
    判断左侧值是否大于等于右侧值，大于等于返回1，否则返回0。
    
    Example:
        >>> pipe = GreaterEqualThan(left_col_name="score", right_col_name="threshold", output_col_name="pass")
    """
    
    def __init__(self, **kwargs):
        """初始化大于等于判断操作。"""
        super().__init__(operate_name="greater_equal_than", **kwargs)

    @staticmethod
    def apply_function_series_bi(left_x, right_x):
        """判断 Series 是否大于等于。"""
        return np.where(left_x >= right_x, 1, 0)

    @staticmethod
    def apply_function_single_bi(left_x, right_x):
        """判断单个值是否大于等于。"""
        return 1 if left_x >= right_x else 0

    def udf_get_params(self):
        """获取参数。"""
        return {}


class LessThan(BiOperationBase):
    """
    小于判断操作。
    
    判断左侧值是否小于右侧值，小于返回1，否则返回0。
    
    Example:
        >>> pipe = LessThan(left_col_name="score", right_col_name="threshold", output_col_name="fail")
    """
    
    def __init__(self, **kwargs):
        """初始化小于判断操作。"""
        super().__init__(operate_name="less_than", **kwargs)

    @staticmethod
    def apply_function_series_bi(left_x, right_x):
        """判断 Series 是否小于。"""
        return np.where(left_x < right_x, 1, 0)

    @staticmethod
    def apply_function_single_bi(left_x, right_x):
        """判断单个值是否小于。"""
        return 1 if left_x < right_x else 0

    def udf_get_params(self):
        """获取参数。"""
        return {}


class LessEqualThan(BiOperationBase):
    """
    小于等于判断操作。
    
    判断左侧值是否小于等于右侧值，小于等于返回1，否则返回0。
    
    Example:
        >>> pipe = LessEqualThan(left_col_name="score", right_col_name="threshold", output_col_name="fail")
    """
    
    def __init__(self, **kwargs):
        """初始化小于等于判断操作。"""
        super().__init__(operate_name="less_equal_than", **kwargs)

    @staticmethod
    def apply_function_series_bi(left_x, right_x):
        """判断 Series 是否小于等于。"""
        return np.where(left_x <= right_x, 1, 0)

    @staticmethod
    def apply_function_single_bi(left_x, right_x):
        """判断单个值是否小于等于。"""
        return 1 if left_x <= right_x else 0

    def udf_get_params(self):
        """获取参数。"""
        return {}


class And(BiOperationBase):
    """
    逻辑与操作。
    
    判断两个条件是否同时满足，同时满足返回1，否则返回0。
    
    Example:
        >>> pipe = And(left_col_name="condition1", right_col_name="condition2", output_col_name="both")
    """
    
    def __init__(self, **kwargs):
        """初始化逻辑与操作。"""
        super().__init__(operate_name="and", **kwargs)

    @staticmethod
    def apply_function_series_bi(left_x, right_x):
        """判断 Series 逻辑与。"""
        return np.where(left_x * right_x > 0, 1, 0)

    @staticmethod
    def apply_function_single_bi(left_x, right_x):
        """判断单个值逻辑与。"""
        return 1 if left_x * right_x > 0 else 0

    def udf_get_params(self):
        """获取参数。"""
        return {}


class Or(BiOperationBase):
    """
    逻辑或操作。
    
    判断两个条件是否至少满足一个，至少一个满足返回1，否则返回0。
    
    Example:
        >>> pipe = Or(left_col_name="condition1", right_col_name="condition2", output_col_name="either")
    """
    
    def __init__(self, **kwargs):
        """初始化逻辑或操作。"""
        super().__init__(operate_name="or", **kwargs)

    @staticmethod
    def apply_function_series_bi(left_x, right_x):
        """判断 Series 逻辑或。"""
        return np.where(left_x + right_x > 0, 1, 0)

    @staticmethod
    def apply_function_single_bi(left_x, right_x):
        """判断单个值逻辑或。"""
        return 1 if left_x + right_x > 0 else 0

    def udf_get_params(self):
        """获取参数。"""
        return {}


class DateDayDiff(BiOperationBase):
    """
    日期天数差计算。
    
    计算两个日期之间的天数差（左值 - 右值）。
    
    Args:
        error_value: 异常值，当日期解析失败时返回的值
    
    Example:
        >>> pipe = DateDayDiff(left_col_name="end_date", right_col_name="start_date", output_col_name="days")
    """
    
    def __init__(self, error_value=-365, **kwargs):
        """
        初始化日期天数差操作。
        
        Args:
            error_value: 异常值
            **kwargs: 其他父类参数
        """
        super().__init__(operate_name="day_diff", **kwargs)
        self.error_value = error_value
        null_extreme_values = GlobalNullValues + GlobalExtremeValues
        self.null_extreme_values = list(set([str(item) for item in null_extreme_values]))

    def apply_function_series_bi(self, left_x: series_type, right_x: series_type):
        """计算 Series 日期天数差。"""
        left_date = pd.to_datetime(left_x.astype(str), errors="coerce")
        right_date = pd.to_datetime(right_x.astype(str), errors="coerce")
        rst = (left_date - right_date).astype('timedelta64[D]').fillna(self.error_value).astype(np.int64)
        return np.where(np.abs(1.0 * rst) > 365000, self.error_value, rst)

    def apply_function_single_bi(self, left_x, right_x):
        """计算单个日期天数差。"""
        try:
            if str(left_x) in self.null_extreme_values or str(right_x) in self.null_extreme_values:
                return self.error_value
            else:
                rst = np.int64(np.timedelta64(np.datetime64(left_x) - np.datetime64(right_x), "D"))
                return self.error_value if np.abs(1.0 * rst) > 365000 else rst
        except Exception as e:
            print("module:{},in function:{},raise exception:{}".format(self.name, "apply_function_single_bi", e))
            return self.error_value

    def udf_get_params(self):
        """获取参数。"""
        return {"error_value": self.error_value, "null_extreme_values": self.null_extreme_values}

    def udf_set_params(self, params: dict):
        """设置参数。"""
        self.error_value = params["error_value"]
        self.null_extreme_values = params["null_extreme_values"]
