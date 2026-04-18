"""
对两个变量的操作,输出值有多个
"""
from .core import *
from easymlops.table.utils import PandasUtils
from easymlops.table.preprocessing import FillNa


class BiInputMultiOutputOperationBase(PreprocessBase):
    """
    二元变量操作
    """

    def __init__(self, left_col_name=None, right_col_name=None, operate_name="", **kwargs):
        """
        :param left_col_name: 左侧操作变量的col
        :param right_col_name: 右侧操作变量的col
        :param operate_name: 操作名称
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.left_col_name = left_col_name
        self.right_col_name = right_col_name
        self.left_col_type = None
        self.right_col_type = None
        self.operate_name = operate_name

    def udf_fit(self, s: dataframe_type, **kwargs):
        # 记录数据类型
        self.left_col_type = self.get_col_type(s[self.left_col_name].dtype)
        self.right_col_type = self.get_col_type(s[self.right_col_name].dtype)
        return self

    def apply_function_single_bi(self, left_with_right):
        raise Exception("need to implement!")

    def apply_function_series_bi(self, left_with_right):
        raise Exception("need to implement!")

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        left_with_right_data = copy.deepcopy(s[[self.left_col_name, self.right_col_name]])
        s = PandasUtils.concat_duplicate_columns([s, self.apply_function_series_bi(left_with_right_data)])
        return s

    def udf_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        left_with_right_data = copy.deepcopy({self.left_col_name: s[self.left_col_name],
                                              self.right_col_name: s[self.right_col_name]})
        s.update(self.apply_function_single_bi(left_with_right_data))
        return s

    def udf_get_params(self):
        return {"right_col_type": self.right_col_type, "left_col_type": self.left_col_type,
                "right_col_name": self.right_col_name, "left_col_name": self.left_col_name,
                "operate_name": self.operate_name}

    def udf_set_params(self, params: dict):
        self.left_col_name = params["left_col_name"]
        self.right_col_name = params["right_col_name"]
        self.left_col_type = params["left_col_type"]
        self.right_col_type = params["right_col_type"]
        self.operate_name = params["operate_name"]


class CrossCategoryWithNumber(BiInputMultiOutputOperationBase):
    """
    交叉离散变量和数值变量,左侧为离散变量,右侧为数值变量
    """

    def __init__(self, left_col_name=None, right_col_name=None, operate_name="cross_cate_with_num__", missing_value=0,
                 **kwargs):
        super().__init__(left_col_name=left_col_name, right_col_name=right_col_name, operate_name=operate_name,
                         **kwargs)
        self.agg_map_info = {}
        self.missing_value = missing_value
        self.cate_fill_na = None
        self.agg_fill_na = None

    def udf_fit(self, s: dataframe_type, **kwargs):
        super().udf_fit(s, **kwargs)
        s_ = copy.deepcopy(s[[self.left_col_name, self.right_col_name]])
        self.cate_fill_na = FillNa([self.left_col_name])
        s_ = self.cate_fill_na.fit(s_).transform(s_)

        group_df = s_.groupby(self.left_col_name).agg(
            {self.right_col_name: ["mean", "median", "max", "min", "sum", "nunique", "std", "skew"]})
        group_df = group_df.reset_index()
        group_df = group_df.reset_index()
        del group_df["index"]
        group_df.columns = ["gid", "mean", "median", "max", "min", "sum", "nunique", "std", "skew"]

        self.agg_fill_na = FillNa(cols=["mean", "median", "max", "min", "sum", "nunique", "std", "skew"],
                                  fill_number_value=self.missing_value)
        group_df = self.agg_fill_na.fit(group_df).transform(group_df)

        for record in group_df.to_dict("records"):
            self.agg_map_info[record["gid"]] = record
        return self

    def apply_function_single_bi(self, left_with_right):
        for col in ["mean", "median", "max", "min", "sum", "nunique", "std", "skew"]:
            left_with_right[self.operate_name + f"{self.left_col_name}__{self.right_col_name}__{col}"] \
                = self.agg_map_info.get(left_with_right[self.left_col_name], {}).get(col, self.missing_value)

        # 补充
        left_with_right[self.operate_name + f"{self.left_col_name}__{self.right_col_name}__max_min_diff"] \
            = left_with_right[self.operate_name + f"{self.left_col_name}__{self.right_col_name}__max"] \
              - left_with_right[self.operate_name + f"{self.left_col_name}__{self.right_col_name}__min"]
        left_with_right[self.operate_name + f"{self.left_col_name}__{self.right_col_name}__mean_diff"] \
            = left_with_right[self.right_col_name] \
              - left_with_right[self.operate_name + f"{self.left_col_name}__{self.right_col_name}__mean"]
        left_with_right[self.operate_name + f"{self.left_col_name}__{self.right_col_name}__median_diff"] \
            = left_with_right[self.right_col_name] \
              - left_with_right[self.operate_name + f"{self.left_col_name}__{self.right_col_name}__median"]
        del left_with_right[self.left_col_name]
        del left_with_right[self.right_col_name]
        return left_with_right

    def apply_function_series_bi(self, left_with_right):
        for col in ["mean", "median", "max", "min", "sum", "nunique", "std", "skew"]:
            left_with_right[self.operate_name + f"{self.left_col_name}__{self.right_col_name}__{col}"] \
                = left_with_right[self.left_col_name].apply(
                lambda x: self.agg_map_info.get(x, {}).get(col, self.missing_value))
        # 补充
        left_with_right[self.operate_name + f"{self.left_col_name}__{self.right_col_name}__max_min_diff"] \
            = left_with_right[self.operate_name + f"{self.left_col_name}__{self.right_col_name}__max"] \
              - left_with_right[self.operate_name + f"{self.left_col_name}__{self.right_col_name}__min"]
        left_with_right[self.operate_name + f"{self.left_col_name}__{self.right_col_name}__mean_diff"] \
            = left_with_right[self.right_col_name] \
              - left_with_right[self.operate_name + f"{self.left_col_name}__{self.right_col_name}__mean"]
        left_with_right[self.operate_name + f"{self.left_col_name}__{self.right_col_name}__median_diff"] \
            = left_with_right[self.right_col_name] \
              - left_with_right[self.operate_name + f"{self.left_col_name}__{self.right_col_name}__median"]
        del left_with_right[self.left_col_name]
        del left_with_right[self.right_col_name]
        return left_with_right

    def udf_get_params(self):
        return {"agg_map_info": self.agg_map_info, "missing_value": self.missing_value,
                "cate_fill_na_params": self.cate_fill_na.get_params(),
                "agg_fill_na_params": self.agg_fill_na.get_params()}

    def udf_set_params(self, params: dict):
        self.agg_map_info = params["agg_map_info"]
        self.missing_value = params["missing_value"]
        self.cate_fill_na = FillNa()
        self.agg_fill_na = FillNa()
        self.cate_fill_na.set_params(params["cate_fill_na_params"])
        self.agg_fill_na.set_params(params["agg_fill_na_params"])


class CrossNumberWithNumber(BiInputMultiOutputOperationBase):
    """
    交叉数值变量和数值变量,左侧为数值变量,右侧为数值变量
    """

    def __init__(self, left_col_name=None, right_col_name=None, operate_name="cross_num_with_num__", missing_value=0,
                 **kwargs):
        super().__init__(left_col_name=left_col_name, right_col_name=right_col_name, operate_name=operate_name,
                         **kwargs)
        self.agg_map_info = {}
        self.fill_na_before = FillNa([self.left_col_name, self.right_col_name], fill_number_value=missing_value)
        self.fill_na_after = FillNa(fill_number_value=missing_value)

    def udf_fit(self, s: dataframe_type, **kwargs):
        super().udf_fit(s, **kwargs)
        s_ = copy.deepcopy(s[[self.left_col_name, self.right_col_name]])
        self.fill_na_before.fit(s_)
        self.fill_na_after.fit(self.apply_function_series_bi_(s_))
        return self

    def apply_function_single_bi(self, left_with_right):
        left_with_right = self.fill_na_before.transform_single(left_with_right)
        left_with_right[self.left_col_name]=np.float64(left_with_right[self.left_col_name])
        left_with_right[self.right_col_name] = np.float64(left_with_right[self.right_col_name])
        left_with_right[self.operate_name + f"{self.left_col_name}__{self.right_col_name}__add"] \
            = 1.0*left_with_right[self.left_col_name] + 1.0*left_with_right[self.right_col_name]
        left_with_right[self.operate_name + f"{self.left_col_name}__{self.right_col_name}__sub"] \
            = 1.0*left_with_right[self.left_col_name] - 1.0*left_with_right[self.right_col_name]
        left_with_right[self.operate_name + f"{self.left_col_name}__{self.right_col_name}__multiply"] \
            = 1.0*left_with_right[self.left_col_name] * 1.0*left_with_right[self.right_col_name]
        left_with_right[self.operate_name + f"{self.left_col_name}__{self.right_col_name}__div"] \
            = 1.0*left_with_right[self.left_col_name] / (1.0*left_with_right[self.right_col_name] + 1e-3)
        left_with_right[self.operate_name + f"{self.left_col_name}__{self.right_col_name}__p1dp2"] \
            = 1.0*left_with_right[self.left_col_name] / (
                1.0*left_with_right[self.right_col_name] * left_with_right[self.right_col_name] + 1e-3)
        left_with_right[self.operate_name + f"{self.left_col_name}__{self.right_col_name}__p2dp1"] \
            = 1.0*left_with_right[self.left_col_name] * left_with_right[self.left_col_name] / \
              (1.0*left_with_right[self.right_col_name] + 1e-3)
        left_with_right[self.operate_name + f"{self.left_col_name}__{self.right_col_name}__p1dp3_log"] \
            = np.log1p(1.0*left_with_right[self.left_col_name] / (
                1.0*left_with_right[self.right_col_name] *
                left_with_right[self.right_col_name] * left_with_right[self.right_col_name] + 1e-3))
        left_with_right[self.operate_name + f"{self.left_col_name}__{self.right_col_name}__p3dp1_log"] \
            = np.log1p(1.0*left_with_right[self.left_col_name] * left_with_right[self.left_col_name]
                       * left_with_right[self.left_col_name] / (1.0*left_with_right[self.right_col_name] + 1e-3))

        del left_with_right[self.left_col_name]
        del left_with_right[self.right_col_name]
        return self.fill_na_after.transform_single(left_with_right)

    def apply_function_series_bi_(self, left_with_right):
        left_with_right = self.fill_na_before.transform(left_with_right)
        left_with_right[self.left_col_name] = np.float64(left_with_right[self.left_col_name])
        left_with_right[self.right_col_name] = np.float64(left_with_right[self.right_col_name])
        left_with_right[self.operate_name + f"{self.left_col_name}__{self.right_col_name}__add"] \
            = 1.0 * left_with_right[self.left_col_name] + 1.0 * left_with_right[self.right_col_name]
        left_with_right[self.operate_name + f"{self.left_col_name}__{self.right_col_name}__sub"] \
            = 1.0 * left_with_right[self.left_col_name] - 1.0 * left_with_right[self.right_col_name]
        left_with_right[self.operate_name + f"{self.left_col_name}__{self.right_col_name}__multiply"] \
            = 1.0 * left_with_right[self.left_col_name] * 1.0 * left_with_right[self.right_col_name]
        left_with_right[self.operate_name + f"{self.left_col_name}__{self.right_col_name}__div"] \
            = 1.0 * left_with_right[self.left_col_name] / (1.0 * left_with_right[self.right_col_name] + 1e-3)
        left_with_right[self.operate_name + f"{self.left_col_name}__{self.right_col_name}__p1dp2"] \
            = 1.0 * left_with_right[self.left_col_name] / (
                1.0 * left_with_right[self.right_col_name] * left_with_right[self.right_col_name] + 1e-3)
        left_with_right[self.operate_name + f"{self.left_col_name}__{self.right_col_name}__p2dp1"] \
            = 1.0 * left_with_right[self.left_col_name] * left_with_right[self.left_col_name] / \
              (1.0 * left_with_right[self.right_col_name] + 1e-3)
        left_with_right[self.operate_name + f"{self.left_col_name}__{self.right_col_name}__p1dp3_log"] \
            = np.log1p(1.0 * left_with_right[self.left_col_name] / (
                1.0 * left_with_right[self.right_col_name] *
                left_with_right[self.right_col_name] * left_with_right[self.right_col_name] + 1e-3))
        left_with_right[self.operate_name + f"{self.left_col_name}__{self.right_col_name}__p3dp1_log"] \
            = np.log1p(1.0 * left_with_right[self.left_col_name] * left_with_right[self.left_col_name]
                       * left_with_right[self.left_col_name] / (1.0 * left_with_right[self.right_col_name] + 1e-3))

        del left_with_right[self.left_col_name]
        del left_with_right[self.right_col_name]
        return left_with_right

    def apply_function_series_bi(self, left_with_right):
        return self.fill_na_after.transform(self.apply_function_series_bi_(left_with_right))

    def udf_get_params(self):
        return {"fill_na_before_params": self.fill_na_before.get_params(),
                "fill_na_after_params": self.fill_na_after.get_params()}

    def udf_set_params(self, params: dict):
        self.fill_na_before = FillNa()
        self.fill_na_after = FillNa()
        self.fill_na_before.set_params(params["fill_na_before_params"])
        self.fill_na_after.set_params(params["fill_na_after_params"])
