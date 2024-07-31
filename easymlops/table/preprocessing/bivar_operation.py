"""
对两个变量的操作
"""
from .core import *


class BiOperationBase(PreprocessBase):
    """
    二元变量操作
    """

    def __init__(self, left_col_name=None, right_col_name=None, left_value=None, right_value=None, output_col_name=None,
                 operate_name="", **kwargs):
        """
        :param left_col_name: 左侧操作变量的col
        :param right_col_name: 右侧操作变量的col
        :param left_value: 左侧操作值，left_col_name=None时生效
        :param right_value: 右侧操作值，right_col_name=None时生效
        :param output_col_name: 操作后输出的col
        :param operate_name: 操作名称
        :param kwargs:
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
        # 记录数据类型
        if self.left_col_name is not None:
            self.left_col_type = self.get_col_type(s[self.left_col_name].dtype)
        if self.right_col_name is not None:
            self.right_col_type = self.get_col_type(s[self.right_col_name].dtype)
        return self

    @staticmethod
    def apply_function_single_bi(left_x, right_x):
        raise Exception("need to implement!")

    @staticmethod
    def apply_function_series_bi(left_x, right_x):
        raise Exception("need to implement!")

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
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
        return {"output_col_name": self.output_col_name, "right_col_type": self.right_col_type,
                "left_col_type": self.left_col_type, "right_value": self.right_value, "left_value": self.left_value,
                "right_col_name": self.right_col_name, "left_col_name": self.left_col_name,
                "operate_name": self.operate_name}

    def udf_set_params(self, params: dict):
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
    支持数值的加和字符串拼接
    """

    def __init__(self, **kwargs):
        super().__init__(operate_name="add", **kwargs)

    @staticmethod
    def apply_function_series_bi(left_x, right_x):
        return left_x + right_x

    @staticmethod
    def apply_function_single_bi(left_x, right_x):
        return left_x + right_x

    def udf_get_params(self):
        return {}


class Subtract(BiOperationBase):
    """
    数值减操作
    """

    def __init__(self, **kwargs):
        super().__init__(operate_name="subtract", **kwargs)

    @staticmethod
    def apply_function_series_bi(left_x, right_x):
        return left_x - right_x

    @staticmethod
    def apply_function_single_bi(left_x, right_x):
        return left_x - right_x

    def udf_get_params(self):
        return {}


class Multiply(BiOperationBase):
    """
    数值乘法
    """

    def __init__(self, **kwargs):
        super().__init__(operate_name="multiply", **kwargs)

    @staticmethod
    def apply_function_series_bi(left_x, right_x):
        return left_x * right_x

    @staticmethod
    def apply_function_single_bi(left_x, right_x):
        return left_x * right_x

    def udf_get_params(self):
        return {}


class Divide(BiOperationBase):
    """
    数值除法
    """

    def __init__(self, **kwargs):
        super().__init__(operate_name="divide", **kwargs)

    @staticmethod
    def apply_function_series_bi(left_x, right_x):
        return left_x / right_x

    @staticmethod
    def apply_function_single_bi(left_x, right_x):
        return left_x / right_x

    def udf_get_params(self):
        return {}


class DivideExact(BiOperationBase):
    """
    数值整除
    """

    def __init__(self, **kwargs):
        super().__init__(operate_name="divide_exact", **kwargs)

    @staticmethod
    def apply_function_series_bi(left_x, right_x):
        return left_x // right_x

    @staticmethod
    def apply_function_single_bi(left_x, right_x):
        return left_x // right_x

    def udf_get_params(self):
        return {}


class Mod(BiOperationBase):
    """
    数值求余
    """

    def __init__(self, **kwargs):
        super().__init__(operate_name="mod", **kwargs)

    @staticmethod
    def apply_function_series_bi(left_x, right_x):
        return left_x // 1 % right_x

    @staticmethod
    def apply_function_single_bi(left_x, right_x):
        return left_x // 1 % right_x

    def udf_get_params(self):
        return {}


class Equal(BiOperationBase):
    """
    判断是否相等
    """

    def __init__(self, **kwargs):
        super().__init__(operate_name="equal", **kwargs)

    @staticmethod
    def apply_function_series_bi(left_x, right_x):
        return np.where(left_x == right_x, 1, 0)

    @staticmethod
    def apply_function_single_bi(left_x, right_x):
        return 1 if left_x == right_x else 0

    def udf_get_params(self):
        return {}


class GreaterThan(BiOperationBase):
    """
    >
    """

    def __init__(self, **kwargs):
        super().__init__(operate_name="greater_than", **kwargs)

    @staticmethod
    def apply_function_series_bi(left_x, right_x):
        return np.where(left_x > right_x, 1, 0)

    @staticmethod
    def apply_function_single_bi(left_x, right_x):
        return 1 if left_x > right_x else 0

    def udf_get_params(self):
        return {}


class GreaterEqualThan(BiOperationBase):
    """
    >=
    """

    def __init__(self, **kwargs):
        super().__init__(operate_name="greater_equal_than", **kwargs)

    @staticmethod
    def apply_function_series_bi(left_x, right_x):
        return np.where(left_x >= right_x, 1, 0)

    @staticmethod
    def apply_function_single_bi(left_x, right_x):
        return 1 if left_x >= right_x else 0

    def udf_get_params(self):
        return {}


class LessThan(BiOperationBase):
    """
    <
    """

    def __init__(self, **kwargs):
        super().__init__(operate_name="less_than", **kwargs)

    @staticmethod
    def apply_function_series_bi(left_x, right_x):
        return np.where(left_x < right_x, 1, 0)

    @staticmethod
    def apply_function_single_bi(left_x, right_x):
        return 1 if left_x < right_x else 0

    def udf_get_params(self):
        return {}


class LessEqualThan(BiOperationBase):
    """
    <=
    """

    def __init__(self, **kwargs):
        super().__init__(operate_name="less_equal_than", **kwargs)

    @staticmethod
    def apply_function_series_bi(left_x, right_x):
        return np.where(left_x <= right_x, 1, 0)

    @staticmethod
    def apply_function_single_bi(left_x, right_x):
        return 1 if left_x <= right_x else 0

    def udf_get_params(self):
        return {}


class And(BiOperationBase):
    """
    逻辑且
    """

    def __init__(self, **kwargs):
        super().__init__(operate_name="and", **kwargs)

    @staticmethod
    def apply_function_series_bi(left_x, right_x):
        return np.where(left_x * right_x > 0, 1, 0)

    @staticmethod
    def apply_function_single_bi(left_x, right_x):
        return 1 if left_x * right_x > 0 else 0

    def udf_get_params(self):
        return {}


class Or(BiOperationBase):
    """
    逻辑或
    """

    def __init__(self, **kwargs):
        super().__init__(operate_name="or", **kwargs)

    @staticmethod
    def apply_function_series_bi(left_x, right_x):
        return np.where(left_x + right_x > 0, 1, 0)

    @staticmethod
    def apply_function_single_bi(left_x, right_x):
        return 1 if left_x + right_x > 0 else 0

    def udf_get_params(self):
        return {}


class DateDayDiff(BiOperationBase):
    """
    日期天数差
    """

    def __init__(self, error_value=-365, **kwargs):
        super().__init__(operate_name="day_diff", **kwargs)
        self.error_value = error_value
        null_extreme_values = GlobalNullValues + GlobalExtremeValues
        self.null_extreme_values = list(set([str(item) for item in null_extreme_values]))

    def apply_function_series_bi(self, left_x: series_type, right_x: series_type):
        left_date = pd.to_datetime(left_x.astype(str), errors="coerce")
        right_date = pd.to_datetime(right_x.astype(str), errors="coerce")
        rst = (left_date - right_date).astype('timedelta64[D]').fillna(self.error_value).astype(np.int64)
        return np.where(np.abs(1.0 * rst) > 365000, self.error_value, rst)

    def apply_function_single_bi(self, left_x, right_x):
        try:

            if str(left_x) in self.null_extreme_values or str(right_x) in self.null_extreme_values:
                return self.error_value
            else:
                rst = np.int64(np.timedelta64(np.datetime64(left_x) - np.datetime64(right_x), "D"))
                # 异常字符串通常会返回一个很大的值，如:np.iinfo(np.int64).min
                return self.error_value if np.abs(1.0 * rst) > 365000 else rst
        except Exception as e:
            print("module:{},in function:{},raise exception:{}".format(self.name, "apply_function_single_bi", e))
            return self.error_value

    def udf_get_params(self):
        return {"error_value": self.error_value, "null_extreme_values": self.null_extreme_values}

    def udf_set_params(self, params: dict):
        self.error_value = params["error_value"]
        self.null_extreme_values = params["null_extreme_values"]
