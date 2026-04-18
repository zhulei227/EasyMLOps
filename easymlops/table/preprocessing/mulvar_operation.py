"""
对多个变量的操作，输出一个变量
"""
from .core import *


class MulOperationBase(TablePipeObjectBase):
    """
    多元变量操作基类。
    
    支持对多个列进行聚合操作，输出一个结果列。
    
    Example:
        >>> pipe = Sum(cols=["price", "tax", "fee"])
    """
    
    def __init__(self, cols="all", output_col_name=None, drop_input=True, **kwargs):
        """
        初始化多元操作。
        
        Args:
            cols: 要操作的列
            output_col_name: 输出列名
            drop_input: 是否删除输入列
            **kwargs: 其他父类参数
        """
        super().__init__(**kwargs)
        self.cols = cols
        self.output_col_name = output_col_name
        self.drop_input = drop_input

    def before_fit(self, s: dataframe_type, **kwargs) -> dataframe_type:
        """fit前处理，确定要操作的列。"""
        s = super().before_fit(s, **kwargs)
        if str(self.cols) == "all" or self.cols is None or (type(self.cols) == list and len(self.cols) == 0):
            self.cols = s.columns.tolist()
        return s[self.cols]

    def udf_fit(self, s: dataframe_type, **kwargs):
        """训练阶段（无需训练，直接返回self）。"""
        return self

    def udf_get_params(self):
        """获取参数。"""
        return {"cols": self.cols, "output_col_name": self.output_col_name, "drop_input": self.drop_input}

    def udf_set_params(self, params: dict):
        """设置参数。"""
        self.cols = params["cols"]
        self.output_col_name = params["output_col_name"]
        self.drop_input = params["drop_input"]


class Sum(MulOperationBase):
    """
    求和操作。
    
    对多个列的值求和。
    
    Example:
        >>> pipe = Sum(cols=["price", "tax", "fee"], output_col_name="total")
    """
    
    def __init__(self, cols="all", output_col_name="sum", **kwargs):
        """
        初始化求和操作。
        
        Args:
            cols: 要操作的列
            output_col_name: 输出列名
            **kwargs: 其他父类参数
        """
        super().__init__(cols=cols, output_col_name=output_col_name, **kwargs)

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        """对数据进行求和转换。"""
        s[self.output_col_name] = s[self.cols].sum(axis=1)
        if not self.drop_input:
            return s
        else:
            return s[[self.output_col_name]]

    def udf_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        """对单条数据进行求和转换。"""
        rst = []
        for col in self.cols:
            rst.append(s[col])
        s[self.output_col_name] = np.sum(rst)
        if not self.drop_input:
            return s
        else:
            return {self.output_col_name: s[self.output_col_name]}

    def udf_get_params(self):
        """获取参数。"""
        return {}


class Mean(MulOperationBase):
    """
    求均值操作。
    
    对多个列的值求平均值。
    
    Example:
        >>> pipe = Mean(cols=["score1", "score2", "score3"], output_col_name="avg_score")
    """
    
    def __init__(self, cols="all", output_col_name="mean", **kwargs):
        """
        初始化求均值操作。
        
        Args:
            cols: 要操作的列
            output_col_name: 输出列名
            **kwargs: 其他父类参数
        """
        super().__init__(cols=cols, output_col_name=output_col_name, **kwargs)

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        """对数据进行求均值转换。"""
        s[self.output_col_name] = s[self.cols].mean(axis=1)
        if not self.drop_input:
            return s
        else:
            return s[[self.output_col_name]]

    def udf_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        """对单条数据进行求均值转换。"""
        rst = []
        for col in self.cols:
            rst.append(s[col])
        s[self.output_col_name] = np.mean(rst)
        if not self.drop_input:
            return s
        else:
            return {self.output_col_name: s[self.output_col_name]}

    def udf_get_params(self):
        """获取参数。"""
        return {}


class Median(MulOperationBase):
    """
    求中位数操作。
    
    对多个列的值求中位数。
    
    Example:
        >>> pipe = Median(cols=["price1", "price2", "price3"], output_col_name="median_price")
    """
    
    def __init__(self, cols="all", output_col_name="median", **kwargs):
        """
        初始化求中位数操作。
        
        Args:
            cols: 要操作的列
            output_col_name: 输出列名
            **kwargs: 其他父类参数
        """
        super().__init__(cols=cols, output_col_name=output_col_name, **kwargs)

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        """对数据进行求中位数转换。"""
        s[self.output_col_name] = s[self.cols].median(axis=1)
        if not self.drop_input:
            return s
        else:
            return s[[self.output_col_name]]

    def udf_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        """对单条数据进行求中位数转换。"""
        rst = []
        for col in self.cols:
            rst.append(s[col])
        s[self.output_col_name] = np.median(rst)
        if not self.drop_input:
            return s
        else:
            return {self.output_col_name: s[self.output_col_name]}

    def udf_get_params(self):
        """获取参数。"""
        return {}
