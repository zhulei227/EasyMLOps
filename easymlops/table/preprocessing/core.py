from easymlops.table.core import *
from ..perfopt import ReduceMemUsage


class PreprocessBase(TablePipeObjectBase):
    """
    预处理操作基类。
    
    所有预处理类的父类，提供列选择和数据转换的标准框架。
    支持对指定列进行批量或单条数据转换。
    
    Example:
        >>> class MyPreprocess(PreprocessBase):
        ...     def apply_function_series(self, col, x):
        ...         return x.upper()
    """
    
    def __init__(self, cols="all", **kwargs):
        """
        初始化预处理操作。
        
        cols输入格式: 
        1. cols="all", cols=None 表示对所有columns进行操作
        2. cols=["col1","col2"] 表示对col1和col2操作
        3. cols=[("col1","new_col1"),("col2","new_col2")] 表示对col1和col2操作，
           并将结果赋值给new_col1和new_col2，并不修改原始col1和col2
        
        Args:
            cols: 要操作的列
            **kwargs: 其他父类参数
        """
        super().__init__(**kwargs)
        self.cols = cols

    def apply_function_series(self, col: str, x: series_type):
        """
        对Series应用转换函数。
        
        Args:
            col: 当前列名称
            x: 当前列对应的值
            
        Returns:
            转换后的Series
        """
        raise Exception("need to implement")

    def apply_function_single(self, col: str, x):
        """
        对单个值应用转换函数。
        
        Args:
            col: 当前列名称
            x: 当前列对应的值
            
        Returns:
            转换后的值
        """
        raise Exception("need to implement")

    def before_fit(self, s: dataframe_type, **kwargs) -> dataframe_type:
        """
        fit前处理，确定要操作的列。
        
        Args:
            s: 输入数据
            **kwargs: 其他参数
            
        Returns:
            处理后的数据
        """
        s = super().before_fit(s, **kwargs)
        if str(self.cols) == "all" or self.cols is None or (type(self.cols) == list and len(self.cols) == 0):
            self.cols = []
            for col in s.columns.tolist():
                if self.prefix is None:
                    self.cols.append((col, col))
                else:
                    self.cols.append((col, f"{self.prefix}_{col}"))
        else:
            if type(self.cols) == list:
                if type(self.cols[0]) == tuple or type(self.cols[0]) == list:
                    pass
                else:
                    new_cols = []
                    for col in self.cols:
                        if self.prefix is None:
                            new_cols.append((col, col))
                        else:
                            new_cols.append((col, f"{self.prefix}_{col}"))
                    self.cols = new_cols
            else:
                raise Exception("cols should be None,'all' or a list")
        self.prefix = None
        return s

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        """对数据进行转换。"""
        for col, new_col in self.cols:
            if col in s.columns:
                s[new_col] = self.apply_function_series(col, s[col])
        return s

    def udf_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        """对单条数据进行转换。"""
        for col, new_col in self.cols:
            if col in s.keys():
                s[new_col] = self.apply_function_single(col, s[col])
        return s

    def udf_get_params(self):
        """获取参数。"""
        return {"cols": self.cols}

    def udf_set_params(self, params: dict):
        """设置参数。"""
        self.cols = params["cols"]


class FixInput(PreprocessBase):
    """
    固定输入数据的名称、顺序和数据类型。
    
    用于规范化输入数据的格式，确保数据符合预期。
    
    Example:
        >>> pipe = FixInput()
    """
    
    def __init__(self, cols="all", reduce_mem_usage=True, skip_check_transform_type=True, show_check_detail=False,
                 skip_check_transform_value=True, **kwargs):
        """
        初始化 FixInput。
        
        Args:
            cols: 要操作的列
            reduce_mem_usage: 是否缩减内存使用
            skip_check_transform_type: 跳过类型检测
            show_check_detail: 显示缺失和多余列的详细信息
            skip_check_transform_value: 跳过值检测
            **kwargs: 其他父类参数
        """
        super().__init__(cols=cols, skip_check_transform_type=skip_check_transform_type,
                         skip_check_transform_value=skip_check_transform_value, **kwargs)
        self.column_dtypes = dict()
        self.reduce_mem_usage = reduce_mem_usage
        self.reduce_mem_usage_mode = None
        self.show_check_detail = show_check_detail

    def udf_fit(self, s: dataframe_type, **kwargs):
        """
        训练阶段，记录数据类型和内存优化配置。
        
        Args:
            s: 输入数据
            **kwargs: 其他参数
            
        Returns:
            self
        """
        self.output_col_names = self.input_col_names
        # 记录数据类型
        self.column_dtypes = dict()
        for col, pandas_col_type in s.dtypes.to_dict().items():
            col_type = self.get_col_type(pandas_col_type)
            self.column_dtypes[col] = col_type
        # reduce mem usage
        if self.reduce_mem_usage:
            self.reduce_mem_usage_mode = ReduceMemUsage()
            self.reduce_mem_usage_mode.fit(s, **kwargs)
        return self

    def _check_miss_addition_columns(self, input_transform_columns):
        """检查缺失和多余的列。"""
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
        """对Series应用类型转换。"""
        col_type = self.column_dtypes[col]
        if col_type == str:
            return x.astype(str)
        else:
            col_type_str = str(col_type).lower()
            x = pd.to_numeric(x, errors="coerce")
            if "int" in col_type_str:
                x = x.fillna(0)
            min_value = np.iinfo(col_type).min if "int" in col_type_str else np.finfo(col_type).min
            max_value = np.iinfo(col_type).max if "int" in col_type_str else np.finfo(col_type).max
            return col_type(np.clip(x, min_value, max_value))

    def apply_function_single(self, col: str, x):
        """对单个值应用类型转换。"""
        col_type = self.column_dtypes[col]
        if col_type == str:
            return str(x)
        else:
            col_type_str = str(col_type).lower()
            x = pd.to_numeric(x, errors="coerce")
            if "int" in col_type_str and "nan" in str(x).lower():
                x = 0
            min_value = np.iinfo(col_type).min if "int" in col_type_str else np.finfo(col_type).min
            max_value = np.iinfo(col_type).max if "int" in col_type_str else np.finfo(col_type).max
            return col_type(np.clip(x, min_value, max_value))

    def transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        """批量数据转换，检查缺失列并应用类型转换。"""
        # 检查缺失
        self._check_miss_addition_columns(s.columns)
        # copy数据
        if self.copy_data:
            s_ = copy.deepcopy(s)
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
                    s_[col] = col_type("nan")
            else:
                # 调整数据类型
                s_[col] = self.apply_function_series(col, s_[col])
        if self.check_list_same(self.output_col_names, s_.columns.tolist()):
            final_s = s_
        else:
            final_s = s_[self.output_col_names]

        # reduce mem usage
        if self.reduce_mem_usage:
            final_s = self.reduce_mem_usage_mode.transform(final_s, **kwargs)
        return final_s

    def transform_single(self, s: dict_type, **kwargs) -> dict_type:
        """单条数据转换，检查缺失列并应用类型转换。"""
        # 检验冲突
        self._check_miss_addition_columns(s.keys())
        # copy数据
        s_ = copy.deepcopy(s)
        for col in self.output_col_names:
            col_type = self.column_dtypes.get(col)
            # 空值填充
            if col not in s_.keys():
                if "int" in str(col_type).lower():
                    s_[col] = col_type(0)
                elif "float" in str(col_type).lower():
                    s_[col] = col_type(np.nan)
                else:
                    s_[col] = col_type("nan")
            else:
                # 调整数据类型
                s_[col] = self.apply_function_single(col, s_[col])
        final_s = self.extract_dict(s_, self.output_col_names)
        # reduce mem usage
        if self.reduce_mem_usage:
            final_s = self.reduce_mem_usage_mode.transform_single(final_s, **kwargs)
        return final_s

    def udf_get_params(self) -> dict_type:
        """获取参数。"""
        params = {"column_dtypes": self.column_dtypes, "reduce_mem_usage": self.reduce_mem_usage}
        if self.reduce_mem_usage:
            params["reduce_mem_usage_mode"] = self.reduce_mem_usage_mode.get_params()
        return params

    def udf_set_params(self, params: dict_type):
        """设置参数。"""
        self.column_dtypes = params["column_dtypes"]
        self.reduce_mem_usage = params["reduce_mem_usage"]
        if self.reduce_mem_usage:
            self.reduce_mem_usage_mode = ReduceMemUsage()
            self.reduce_mem_usage_mode.set_params(params["reduce_mem_usage_mode"])

    def callback(self, callback_func, data, return_callback_result=False, *args, **kwargs):
        """
        回调函数接口
        """
        show_flag = self.show_check_detail
        if show_flag:
            self.show_check_detail = False
        result = callback_func(self, data, *args, **kwargs)
        self.show_check_detail = show_flag
        if return_callback_result:
            return result


class ReName(PreprocessBase):
    """
    修改col名称：[(col1,new_col1),(col2,new_col2)]
    """

    def __init__(self, cols="all", **kwargs):
        super().__init__(cols=cols, **kwargs)

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        for col, new_col in self.cols:
            if col != new_col:
                s[new_col] = s[col]
                del s[col]
        return s

    def udf_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        for col, new_col in self.cols:
            if col != new_col:
                s[new_col] = s[col]
                del s[col]
        return s

    def udf_get_params(self):
        return {}


class DropCols(PreprocessBase):
    """
    删掉特定的列:cols
    """

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        for col, _ in self.cols:
            if col in s.columns.tolist():
                del s[col]
        return s

    def udf_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        for col, _ in self.cols:
            if col in s.keys():
                s.pop(col)
        return s

    def udf_get_params(self):
        return {}


class SelectCols(PreprocessBase):
    """
    选择特定的cols
    """

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        selected_cols = [col for col, _ in self.cols]
        for col in selected_cols:
            if col not in s.columns:
                raise Exception("{} not in {}".format(col, s.columns.tolist))
        s = s[selected_cols]
        return s

    def udf_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        s_ = dict()
        selected_cols = [col for col, _ in self.cols]
        for col in selected_cols:
            if col not in s.keys():
                raise Exception("{} not in {}".format(col, s.keys()))
        for col in selected_cols:
            s_[col] = s[col]
        return s_

    def udf_get_params(self):
        return {}


class DoNoThing(TablePipeObjectBase):
    """
    顾名思义，啥也不做，透传
    """

    def udf_get_params(self):
        return {}
