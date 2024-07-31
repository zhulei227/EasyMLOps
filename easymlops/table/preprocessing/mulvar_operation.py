"""
对多个变量的操作，输出一个变量
"""
from .core import *


class MulOperationBase(TablePipeObjectBase):
    """
    二元变量操作
    """

    def __init__(self, cols="all", output_col_name=None, drop_input=True, **kwargs):
        super().__init__(**kwargs)
        self.cols = cols
        self.output_col_name = output_col_name
        self.drop_input = drop_input

    def before_fit(self, s: dataframe_type, **kwargs) -> dataframe_type:
        s = super().before_fit(s, **kwargs)
        if str(self.cols) == "all" or self.cols is None or (type(self.cols) == list and len(self.cols) == 0):
            self.cols = s.columns.tolist()
        return s[self.cols]

    def udf_fit(self, s: dataframe_type, **kwargs):
        return self

    def udf_get_params(self):
        return {"cols": self.cols, "output_col_name": self.output_col_name, "drop_input": self.drop_input}

    def udf_set_params(self, params: dict):
        self.cols = params["cols"]
        self.output_col_name = params["output_col_name"]
        self.drop_input = params["drop_input"]


class Sum(MulOperationBase):
    def __init__(self, cols="all", output_col_name="sum", **kwargs):
        super().__init__(cols=cols, output_col_name=output_col_name, **kwargs)

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        s[self.output_col_name] = s[self.cols].sum(axis=1)
        if not self.drop_input:
            return s
        else:
            return s[[self.output_col_name]]

    def udf_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        rst = []
        for col in self.cols:
            rst.append(s[col])
        s[self.output_col_name] = np.sum(rst)
        if not self.drop_input:
            return s
        else:
            return {self.output_col_name: s[self.output_col_name]}

    def udf_get_params(self):
        return {}


class Mean(MulOperationBase):
    def __init__(self, cols="all", output_col_name="mean", **kwargs):
        super().__init__(cols=cols, output_col_name=output_col_name, **kwargs)

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        s[self.output_col_name] = s[self.cols].mean(axis=1)
        if not self.drop_input:
            return s
        else:
            return s[[self.output_col_name]]

    def udf_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        rst = []
        for col in self.cols:
            rst.append(s[col])
        s[self.output_col_name] = np.mean(rst)
        if not self.drop_input:
            return s
        else:
            return {self.output_col_name: s[self.output_col_name]}

    def udf_get_params(self):
        return {}


class Median(MulOperationBase):
    def __init__(self, cols="all", output_col_name="median", **kwargs):
        super().__init__(cols=cols, output_col_name=output_col_name, **kwargs)

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        s[self.output_col_name] = s[self.cols].median(axis=1)
        if not self.drop_input:
            return s
        else:
            return s[[self.output_col_name]]

    def udf_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        rst = []
        for col in self.cols:
            rst.append(s[col])
        s[self.output_col_name] = np.median(rst)
        if not self.drop_input:
            return s
        else:
            return {self.output_col_name: s[self.output_col_name]}

    def udf_get_params(self):
        return {}
