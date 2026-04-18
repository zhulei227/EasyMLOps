from tqdm import tqdm

from easymlops.core import *
import pandas as pd
import copy
import numpy as np

dataframe_type = pd.DataFrame
series_type = pd.Series
dict_type = dict
GlobalNullValues = [np.nan, np.inf, -np.inf, None, "NULL", "NaN", "Nan", "None", "NONE", "none", "nan", "np.nan",
                    "null", "", " ", "inf", "-inf", "-np.inf", "np.inf", "Null"]
GlobalExtremeValues = [np.inf, -np.inf, 0.0, 0, -1, -1.0, 1.0, 1, -1e-7, 1e-7, np.iinfo(np.int64).min,
                       np.iinfo(np.int64).max, np.finfo(np.float64).min,
                       np.finfo(np.float64).max, "", "null", None, "1.0", "0.0", "-1.0", "-1", "NaN", "None"]


class TablePipeObjectBase(PipeObjectBase):
    """
    表格数据Pipe基类。
    
    继承自PipeObjectBase，专门用于处理表格数据(DataFrame)。
    提供了fit、transform、transform_single等核心方法的标准调用流程。
    
    Example:
        >>> class MyPipe(TablePipeObjectBase):
        ...     def udf_fit(self, s, **kwargs):
        ...         return self
        ...     def udf_transform(self, s, **kwargs):
        ...         return s
    """
    
    def __init__(self, name=None, transform_check_max_number_error=1e-5, skip_check_transform_type=False,
                 skip_check_transform_value=False, leak_check_transform_type=True, leak_check_transform_value=True,
                 copy_data=False, prefix=None, **kwargs):
        """
        初始化Table Pipe对象。
        
        Args:
            name: 模块名称，如果为空默认为self.__class__
            transform_check_max_number_error: 允许的最大数值误差
            skip_check_transform_type: 是否跳过类型检测（针对稀疏矩阵等特殊类型）
            skip_check_transform_value: 是否跳过数值相等检测
            leak_check_transform_type: 弱化类型检测
            leak_check_transform_value: 弱化值检测
            copy_data: transform阶段是否复制数据
            prefix: 为输出添加前缀
            **kwargs: 其他父类参数
        """
        super().__init__(name=name, **kwargs)
        self.input_col_names = None
        self.output_col_names = None
        self.transform_check_max_number_error = transform_check_max_number_error
        self.skip_check_transform_type = skip_check_transform_type
        self.skip_check_transform_value = skip_check_transform_value
        self.leak_check_transform_type = leak_check_transform_type
        self.leak_check_transform_value = leak_check_transform_value
        self.copy_data = copy_data
        self.prefix = prefix
        self.branch_pipes = []
        self.master_pipe = None
        self.parent_pipe = None

    def fit(self, s: dataframe_type, **kwargs):
        """
        训练阶段入口。
        
        依次调用: before_fit -> udf_fit -> after_fit
        
        Args:
            s: 输入数据
            **kwargs: 其他参数
            
        Returns:
            self
        """
        self.udf_fit(self.before_fit(s, **kwargs), **kwargs)
        return self.after_fit(s, **kwargs)

    def before_fit(self, s: dataframe_type, **kwargs) -> dataframe_type:
        """
        fit前预操作。
        
        Args:
            s: 输入数据
            **kwargs: 其他参数
            
        Returns:
            处理后的数据
        """
        s = super().before_fit(s, **kwargs)
        assert type(s) == dataframe_type
        self.input_col_names = s.columns.tolist()
        return s

    def udf_fit(self, s: dataframe_type, **kwargs):
        """
        用户自定义的fit实现。
        
        Args:
            s: 输入数据
            **kwargs: 其他参数
            
        Returns:
            self
        """
        return self

    def after_fit(self, s: dataframe_type = None, **kwargs):
        """
        fit后操作。
        
        Args:
            s: 输入数据
            **kwargs: 其他参数
            
        Returns:
            self
        """
        super().after_fit(s, **kwargs)
        for branch_pipe in self.branch_pipes:
            branch_pipe.fit(copy.deepcopy(s), **kwargs)
        self.get_params()
        return self

    def transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        """
        批量数据转换入口。
        
        依次调用: before_transform -> udf_transform -> after_transform
        
        Args:
            s: 输入数据
            **kwargs: 其他参数
            
        Returns:
            转换后的DataFrame
        """
        return self.after_transform(self.udf_transform(self.before_transform(s, **kwargs), **kwargs), **kwargs)

    def before_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        """
        transform前预操作。
        
        Args:
            s: 输入数据
            **kwargs: 其他参数
            
        Returns:
            处理后的数据
        """
        s = super().before_transform(s, **kwargs)
        assert type(s) == dataframe_type
        if self.copy_data:
            s_ = copy.deepcopy(s)
        else:
            s_ = s
        input_cols = s.columns.tolist()
        if self.check_list_same(input_cols, self.input_col_names):
            return s_
        else:
            return s_[self.input_col_names]

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        """
        用户自定义的transform实现。
        
        Args:
            s: 输入数据
            **kwargs: 其他参数
            
        Returns:
            转换后的数据
        """
        return s

    def after_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        """
        transform后操作。
        
        Args:
            s: 转换后的数据
            **kwargs: 其他参数
            
        Returns:
            处理后的数据
        """
        s = super().after_transform(s, **kwargs)
        if self.prefix is not None:
            s.columns = ["{}_{}".format(self.prefix, col) for col in s.columns]
        self.output_col_names = list(s.columns)
        for branch_pipe in self.branch_pipes:
            branch_pipe.transform(copy.deepcopy(s), **kwargs)
        return s

    def transform_single(self, s: dict_type, **kwargs) -> dict_type:
        """
        单条数据转换入口。
        
        依次调用: before_transform_single -> udf_transform_single -> after_transform_single
        
        Args:
            s: 输入数据字典
            **kwargs: 其他参数
            
        Returns:
            转换后的数据字典
        """
        return self.after_transform_single(
            self.udf_transform_single(self.before_transform_single(s, **kwargs), **kwargs), **kwargs)

    def before_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        """
        单条数据转换前预操作。
        
        Args:
            s: 输入数据字典
            **kwargs: 其他参数
            
        Returns:
            处理后的数据字典
        """
        s = super().before_transform_single(s, **kwargs)
        assert type(s) == dict_type
        return self.extract_dict(s, self.input_col_names)

    def udf_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        """
        用户自定义的单条数据转换实现。
        
        Args:
            s: 输入数据字典
            **kwargs: 其他参数
            
        Returns:
            转换后的数据字典
        """
        return s

    def after_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        """
        单条数据转换后操作。
        
        Args:
            s: 转换后的数据字典
            **kwargs: 其他参数
            
        Returns:
            处理后的数据字典
        """
        s = super().after_transform_single(s, **kwargs)
        if self.prefix is not None:
            new_s = dict()
            for col, value in s.items():
                new_s["{}_{}".format(self.prefix, col)] = value
            s = new_s
        s = self.extract_dict(s, self.output_col_names)
        for branch_pipe in self.branch_pipes:
            branch_pipe.transform_single(copy.deepcopy(s), **kwargs)
        return s

    def get_params(self) -> dict_type:
        """
        获取参数。
        
        Returns:
            参数字典
        """
        current_class = self.__class__
        all_params = [self.udf_get_params()]
        while issubclass(current_class, TablePipeObjectBase):
            all_params.append(super(current_class, self).udf_get_params())
            self.check_key_conflict(all_params[-1], all_params[-2], current_class, current_class.__base__)
            current_class = current_class.__base__
        all_params.reverse()
        combine_params = dict()
        for params in all_params:
            combine_params.update(params)
        return combine_params

    def udf_get_params(self) -> dict_type:
        """获取参数。"""
        return {"input_col_names": self.input_col_names,
                "output_col_names": self.output_col_names,
                "transform_check_max_number_error": self.transform_check_max_number_error,
                "skip_check_transform_type": self.skip_check_transform_type,
                "skip_check_transform_value": self.skip_check_transform_value,
                "copy_data": self.copy_data,
                "prefix": self.prefix,
                "branch_pipes": [branch_pipe.get_params() for branch_pipe in self.branch_pipes]}

    def set_params(self, params: dict_type):
        """
        设置参数。
        
        Args:
            params: 参数字典
        """
        current_class = self.__class__
        super_classes = []
        while issubclass(current_class, TablePipeObjectBase):
            super_classes.append(super(current_class, self))
            current_class = current_class.__base__
        super_classes.reverse()
        for super_class in super_classes:
            super_class.udf_set_params(params)
        self.udf_set_params(params)

    def udf_set_params(self, params: dict_type):
        """
        用户自定义的参数设置实现。
        
        Args:
            params: 参数字典
        """
        self.input_col_names = params["input_col_names"]
        self.output_col_names = params["output_col_names"]
        self.transform_check_max_number_error = params["transform_check_max_number_error"]
        self.skip_check_transform_type = params["skip_check_transform_type"]
        self.skip_check_transform_value = params["skip_check_transform_value"]
        self.copy_data = params["copy_data"]
        self.prefix = params["prefix"]
        for index in range(len(params["branch_pipes"])):
            self.branch_pipes[index].set_params(params["branch_pipes"][index])

    @staticmethod
    def extract_dict(s: dict_type, keys: list) -> dict_type:
        """
        从字典中提取指定的键。
        
        Args:
            s: 源字典
            keys: 要提取的键列表
            
        Returns:
            新字典
        """
        new_s = dict()
        for key in keys:
            new_s[key] = s[key]
        return new_s

    @staticmethod
    def check_list_same(list1: list, list2: list):
        """
        比较两个列表元素是否完全相同。
        
        Args:
            list1: 列表1
            list2: 列表2
            
        Returns:
            是否相同
        """
        flag = True
        if len(list1) != len(list2):
            flag = False
        if flag:
            for idx in range(len(list1)):
                item1 = list1[idx]
                item2 = list2[idx]
                if item1 != item2:
                    flag = False
                    break
        return flag

    @staticmethod
    def check_key_conflict(param1: dict_type, param2: dict_type, class1, class2):
        """
        检测两个参数字典中是否有键冲突。
        
        Args:
            param1: 参数字典1
            param2: 参字典典2
            class1: 类1
            class2: 类2
        """
        same_param_names = list(set(param1.keys()) & set(param2.keys()))
        if len(same_param_names) > 0:
            print("the {} and {} use same parameter names \033[1;43m[{}]\033[0m,please check if conflict,".format
                  (class1, class2, same_param_names))

    @staticmethod
    def get_col_type(pandas_col_type):
        """
        获取pandas列类型对应的numpy类型。
        
        Args:
            pandas_col_type: pandas数据类型
            
        Returns:
            numpy数据类型
        """
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

    @staticmethod
    def get_match_values_index(x: series_type, values):
        """
        获取值匹配的索引。
        
        Args:
            x: 数据Series
            values: 要匹配的值列表
            
        Returns:
            布尔索引Series
        """
        x_ = copy.deepcopy(x)
        index_map = dict()
        for value in values:
            index_map[value] = True
        try:
            index = x_.map(index_map).fillna(False)
        except:
            index = x_.apply(lambda x: index_map.get(x, False))
        return index.astype(bool)

    def set_master_pipe(self, master_pipe):
        """
        设置master pipe。
        
        Args:
            master_pipe: master pipe对象
        """
        self.master_pipe = master_pipe

    def get_master_pipe(self):
        """
        获取master pipe。
        
        Returns:
            master pipe对象
        """
        return self.master_pipe

    def set_branch_pipe(self, pipe_obj):
        """
        添加branch pipe。
        
        Args:
            pipe_obj: pipe对象
        """
        pipe_obj.set_master_pipe(self)
        pipe_obj.set_parent_pipe(self)
        self.branch_pipes.append(pipe_obj)

    def get_branch_pipe(self, index):
        """
        获取指定索引的branch pipe。
        
        Args:
            index: 索引（整数或字符串）
            
        Returns:
            branch pipe对象
        """
        if type(index) == int:
            return self.branch_pipes[index]
        elif type(index) == str:
            for branch_pipe in self.branch_pipes:
                if branch_pipe.name == index:
                    return branch_pipe
        else:
            raise Exception(f"can't support the index:{index}")

    def remove_branch_pipe(self, index):
        """
        移除指定索引的branch pipe。
        
        Args:
            index: 索引（整数或字符串）
        """
        hint_index = None
        if type(index) == int:
            hint_index = index
        elif type(index) == str:
            for branch_pipe in self.branch_pipes:
                if branch_pipe.name == index:
                    hint_index = index
                    break
        if hint_index is None:
            raise Exception(f"can't find the index:{index}")
        else:
            self.branch_pipes.pop(hint_index)

    def set_parent_pipe(self, parent_pipe=None):
        """
        设置父pipe。
        
        Args:
            parent_pipe: 父pipe对象
        """
        self.parent_pipe = parent_pipe

    def get_parent_pipe(self):
        """
        获取直接父pipe。
        
        Returns:
            父pipe对象
        """
        return self.parent_pipe

    def get_all_parent_pipes(self):
        """
        获取所有父pipe列表。
        
        Returns:
            父pipe列表
        """
        all_parent_pipes = []
        if self.parent_pipe is not None:
            all_parent_pipes.append(self.parent_pipe)
        while all_parent_pipes[-1].get_parent_pipe() is not None:
            all_parent_pipes.append(all_parent_pipes[-1].get_parent_pipe())
        all_parent_pipes.reverse()
        return all_parent_pipes

    def transform_all_parent(self, x, show_process=False, **kwargs):
        """
        对所有父pipe顺序执行transform。
        
        Args:
            x: 输入数据
            show_process: 是否显示进度
            **kwargs: 其他参数
            
        Returns:
            转换后的数据
        """
        x_ = copy.deepcopy(x)
        all_parent_pipes = self.get_all_parent_pipes()
        for current_parent_pipe in tqdm(all_parent_pipes) if show_process else all_parent_pipes:
            if show_process:
                print(current_parent_pipe.name)
            x_ = current_parent_pipe.transform(x_, **kwargs)
        return x_

    def transform_single_all_parent(self, x, show_process=False, **kwargs):
        """
        对所有父pipe顺序执行transform_single。
        
        Args:
            x: 输入数据字典
            show_process: 是否显示进度
            **kwargs: 其他参数
            
        Returns:
            转换后的数据字典
        """
        x_ = copy.deepcopy(x)
        all_parent_pipes = self.get_all_parent_pipes()
        for current_parent_pipe in tqdm(all_parent_pipes) if show_process else all_parent_pipes:
            if show_process:
                print(current_parent_pipe.name)
            x_ = current_parent_pipe.transform_single(x_, **kwargs)
        return x_
