from ...base import *
from ..perfopt import ReduceMemUsage


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

    @staticmethod
    def get_col_type(pandas_col_type):
        pandas_col_type = str(pandas_col_type).lower()
        if "int" in pandas_col_type:
            if "int8" in pandas_col_type:
                col_type = np.int8
            elif "int16" in pandas_col_type:
                col_type = np.int16
            elif "int32" in pandas_col_type:
                col_type = np.int32
            else:
                col_type = np.int64
        elif "float" in pandas_col_type:
            if "float16" in pandas_col_type:
                col_type = np.float16
            elif "float32" in pandas_col_type:
                col_type = np.float32
            else:
                col_type = np.float64
        else:
            col_type = str
        return col_type

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
        for col, pandas_col_type in s.dtypes.to_dict().items():
            col_type = self.get_col_type(pandas_col_type)
            self.column_dtypes[col] = col_type
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

    def apply_function_series(self, col: str, x: series_type):
        col_type = self.column_dtypes[col]
        if col_type == str:
            return x.astype(str)
        else:
            col_type_str = str(col_type).lower()
            min_value = np.iinfo(col_type).min if "int" in col_type_str else np.finfo(col_type).min
            max_value = np.iinfo(col_type).max if "int" in col_type_str else np.finfo(col_type).max
            try:
                if "int" in col_type_str:
                    x = x.fillna(0)
                return col_type(np.clip(x, min_value, max_value))
            except:
                return x.apply(lambda x_: self.as_number(x_, col_type, min_value, max_value))

    def apply_function_single(self, col: str, x):
        col_type = self.column_dtypes[col]
        if col_type == str:
            return str(x)
        else:
            col_type_str = str(col_type).lower()
            min_value = np.iinfo(col_type).min if "int" in col_type_str else np.finfo(col_type).min
            max_value = np.iinfo(col_type).max if "int" in col_type_str else np.finfo(col_type).max
            try:
                if "int" in col_type_str and str(x).lower() == "nan":
                    return col_type(0)
                else:
                    return col_type(np.clip(x, min_value, max_value))
            except:
                return self.as_number(x, col_type, min_value, max_value)

    @staticmethod
    def as_number(x, col_type, min_value, max_value):
        col_type_str = str(col_type).lower()
        try:
            return col_type(np.clip(x, min_value, max_value))
        except:
            return col_type(0) if "int" in col_type_str else col_type(np.nan)

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
                if "int" in str(col_type).lower():
                    s_[col] = col_type(0)
                elif "float" in str(col_type).lower():
                    s_[col] = col_type(np.nan)
                else:
                    s_[col] = col_type(None)
            else:
                # 调整数据类型
                s_[col] = self.apply_function_series(col, s_[col])
        # reduce mem usage
        if self.reduce_mem_usage:
            s_ = self.reduce_mem_usage_mode.transform(s_)
        if self.check_list_same(self.output_col_names, s_.columns.tolist()):
            return s_
        else:
            return s_[self.output_col_names]

    def transform_single(self, s: dict_type) -> dict_type:
        # 检验冲突
        self._check_miss_addition_columns(s.keys())
        # copy数据
        s_ = copy.copy(s)
        for col in self.output_col_names:
            col_type = self.column_dtypes.get(col)
            # 空值填充
            if col not in s_.keys():
                if "int" in str(col_type).lower():
                    s_[col] = col_type(0)
                elif "float" in str(col_type).lower():
                    s_[col] = col_type(np.nan)
                else:
                    s_[col] = col_type(None)
            else:
                # 调整数据类型
                s_[col] = self.apply_function_single(col, s_[col])
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


class ReName(PreprocessBase):
    """
    #换cols的名
    """

    def __init__(self, cols="all", **kwargs):
        super().__init__(cols=cols, **kwargs)

    def _transform(self, s: dataframe_type) -> dataframe_type:
        for col, new_col in self.cols:
            if col != new_col:
                s[new_col] = s[col]
                del s[col]
        return s

    def _transform_single(self, s: dict_type) -> dict_type:
        for col, new_col in self.cols:
            if col != new_col:
                s[new_col] = s[col]
                del s[col]
        return s

    def _get_params(self):
        return {}


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
