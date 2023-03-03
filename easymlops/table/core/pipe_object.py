from tqdm import tqdm

from easymlops.core import *
import pandas as pd
import copy
import numpy as np

dataframe_type = pd.DataFrame
series_type = pd.Series
dict_type = dict
GlobalNullValues = [np.nan, np.inf, None, "NULL", "NaN", "Nan", "None", "NONE", "none", "nan", "np.nan",
                    "null", "", " ", "inf", "np.inf", "Null"]
GlobalExtremeValues = [np.inf, 0.0, 0, -1, -1.0, 1.0, 1, -1e-7, 1e-7, np.iinfo(np.int64).min,
                       np.iinfo(np.int64).max, np.finfo(np.float64).min,
                       np.finfo(np.float64).max, "", "null", None, "1.0", "0.0", "-1.0", "-1", "NaN", "None"]


class TablePipeObjectBase(PipeObjectBase):
    """
    Table型模型的PipeBase
    """

    def __init__(self, name=None, transform_check_max_number_error=1e-5, skip_check_transform_type=False,
                 skip_check_transform_value=False, leak_check_transform_type=True, leak_check_transform_value=True,
                 copy_data=False, prefix=None, **kwargs):
        """

        :param name: 模块名称，如果为空默认为self.__class__
        :param transform_check_max_number_error: 在check_transform_function时允许的最大数值误差
        :param skip_check_transform_type: 在check_transform_function时是否跳过类型检测(针对一些特殊数据类型，比如稀疏矩阵)
        :param skip_check_transform_value: 是否跳过数值相等检测（目前只有FixInput模块要跳，其他都不会跳过）
        :param leak_check_transform_type: 弱化类型检测，比如将int32和int64都视为int类型
        :param leak_check_transform_value: 弱化检测值，将None视为相等
        :param copy_data: transform阶段是否要copy一次数据
        :param prefix: 是否为输出添加一个前缀,默认None,不添加
        :param kwargs:
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
        self.master_pipe = None  # 当前pipe作为branch pipe时所绑定的主master的pipe引用，这个对象无需保存，待申明时指定
        self.parent_pipe = None  # 父类对象不保存，实际部署时会重新申明使得保存的parent pipe失效

    def fit(self, s: dataframe_type, **kwargs):
        """
        fit:依次调用before_fit,_fit,after_fit

        :param s:
        :param kwargs:
        :return:
        """
        self.udf_fit(self.before_fit(s, **kwargs))
        return self.after_fit(s, **kwargs)

    def before_fit(self, s: dataframe_type, **kwargs) -> dataframe_type:
        """
        fit前预操作

        :param s:
        :param kwargs:
        :return:
        """
        s = super().before_fit(s, **kwargs)
        assert type(s) == dataframe_type
        self.input_col_names = s.columns.tolist()
        return s

    def udf_fit(self, s: dataframe_type, **kwargs):
        """

        :param s:
        :param kwargs:
        :return:
        """
        return self

    def after_fit(self, s: dataframe_type = None, **kwargs):
        """
        fit后操作

        :param s:
        :param kwargs:
        :return:
        """
        super().after_fit(s, **kwargs)
        # 执行branch pipe
        for branch_pipe in self.branch_pipes:
            branch_pipe.fit(copy.deepcopy(s), **kwargs)
        # 检验参数冲突
        self.get_params()
        return self

    def transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        """
        批量接口:依次调用before_transform,_transform,after_transform

        :param s:
        :param kwargs:
        :return:
        """
        return self.after_transform(self.udf_transform(self.before_transform(s, **kwargs), **kwargs), **kwargs)

    def before_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        """
        udf_transform之前调用

        :param s:
        :param kwargs:
        :return:
        """
        s = super().before_transform(s, **kwargs)
        assert type(s) == dataframe_type
        # 是否copy数据, copy可以避免修改input
        if self.copy_data:
            s_ = copy.deepcopy(s)
        else:
            s_ = s
        # 对其训练时的input columns
        input_cols = s.columns.tolist()
        if self.check_list_same(input_cols, self.input_col_names):
            return s_
        else:
            return s_[self.input_col_names]

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        """

        :param s:
        :param kwargs:
        :return:
        """
        return s

    def after_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        """
        udf_transform之后调用

        :param s:
        :param kwargs:
        :return:
        """
        s = super().after_transform(s, **kwargs)
        # 是否改名
        if self.prefix is not None:
            s.columns = ["{}_{}".format(self.prefix, col) for col in s.columns]
        # 保留output columns
        self.output_col_names = list(s.columns)
        # 执行branch pipe
        for branch_pipe in self.branch_pipes:
            branch_pipe.transform(copy.deepcopy(s), **kwargs)
        return s

    def transform_single(self, s: dict_type, **kwargs) -> dict_type:
        """
        单条数据接口:调用顺序before_transform_single,_transform_single,after_transform_single

        :param s:
        :param kwargs:
        :return:
        """
        return self.after_transform_single(
            self.udf_transform_single(self.before_transform_single(s, **kwargs), **kwargs), **kwargs)

    def before_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        """
        udf_transform_single之前调用

        :param s:
        :param kwargs:
        :return:
        """
        s = super().before_transform_single(s, **kwargs)
        assert type(s) == dict_type
        return self.extract_dict(s, self.input_col_names)

    def udf_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        """
        单条数据接口，生产部署用

        :param s:
        :param kwargs:
        :return:
        """
        return s

    def after_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        """
        udf_transform_single之后调用

        :param s:
        :param kwargs:
        :return:
        """
        s = super().after_transform_single(s, **kwargs)
        # 改名
        if self.prefix is not None:
            new_s = dict()
            for col, value in s.items():
                new_s["{}_{}".format(self.prefix, col)] = value
            s = new_s
        # 输出指定columns
        s = self.extract_dict(s, self.output_col_names)
        # 执行branch pipe
        for branch_pipe in self.branch_pipes:
            branch_pipe.transform_single(copy.deepcopy(s), **kwargs)
        return s

    def get_params(self) -> dict_type:
        """

        :return:
        """
        # 逐步获取父类的_get_params
        current_class = self.__class__
        all_params = [self.udf_get_params()]
        while issubclass(current_class, TablePipeObjectBase):
            all_params.append(super(current_class, self).udf_get_params())
            self.check_key_conflict(all_params[-1], all_params[-2], current_class, current_class.__base__)
            current_class = current_class.__base__
        all_params.reverse()
        # 逆向聚合参数
        combine_params = dict()
        for params in all_params:
            combine_params.update(params)
        # 获取当前参数
        return combine_params

    def udf_get_params(self) -> dict_type:
        """

        :return:
        """
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

        :param params:
        :return:
        """
        # 逐步获取父类class
        current_class = self.__class__
        super_classes = []
        while issubclass(current_class, TablePipeObjectBase):
            super_classes.append(super(current_class, self))
            current_class = current_class.__base__
        super_classes.reverse()
        for super_class in super_classes:
            super_class.udf_set_params(params)
        # 设置当前类
        self.udf_set_params(params)

    def udf_set_params(self, params: dict_type):
        """
        :param params:
        :return:
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
        从s中抽取指定的keys

        :param s:
        :param keys:
        :return:
        """
        new_s = dict()
        for key in keys:
            new_s[key] = s[key]
        return new_s

    @staticmethod
    def check_list_same(list1: list, list2: list):
        """
        比较俩列表元素是否完全相同

        :param list1:
        :param list2:
        :return:
        """
        flag = True
        if len(list1) != len(list2):
            flag = False
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
        检测param1和param2中的key是否有冲突

        :param param1:
        :param param2:
        :param class1:
        :param class2:
        :return:
        """
        same_param_names = list(set(param1.keys()) & set(param2.keys()))
        if len(same_param_names) > 0:
            print("the {} and {} use same parameter names \033[1;43m[{}]\033[0m,please check if conflict,".format
                  (class1, class2, same_param_names))

    @staticmethod
    def get_col_type(pandas_col_type):
        """
        获取当前col的数据类型

        :param pandas_col_type:
        :return:
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
        获取当前x的值是否匹配values
        """
        x_ = copy.deepcopy(x)
        index_map = dict()
        for value in values:
            index_map[value] = True
        index = x_.map(index_map).fillna(False)
        return index.astype(np.bool)

    def set_master_pipe(self, master_pipe):
        """
        设置master_pipe，这个函数是被pipeline调用的，当前pipe的角色是branch pipe

        :param master_pipe:
        :return:
        """
        self.master_pipe = master_pipe

    def get_master_pipe(self):
        """
        获取绑定的master pipe

        :return:
        """
        return self.master_pipe

    def set_branch_pipe(self, pipe_obj):
        """
        设置 branch pipe

        :param pipe_obj:
        :return:
        """
        pipe_obj.set_master_pipe(self)
        self.branch_pipes.append(pipe_obj)

    def get_branch_pipe(self, index):
        """
        获取指定index的branch pipe

        :param index:
        :return:
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
        移除指定index的branch pipe

        :param index:
        :return:
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
        设置前置的pipe，被pipeline调用

        :param parent_pipe:
        :return:
        """
        self.parent_pipe = parent_pipe

    def get_parent_pipe(self):
        """
        获取直接前置的pipe

        :return:
        """
        return self.parent_pipe

    def get_all_parent_pipes(self):
        """
        获取所有前置pipe 列表

        :return:
        """
        # 找到所有的父类(逆序保存)
        all_parent_pipes = []
        if self.parent_pipe is not None:
            all_parent_pipes.append(self.parent_pipe)
        while all_parent_pipes[-1].get_parent_pipe() is not None:
            all_parent_pipes.append(all_parent_pipes[-1].get_parent_pipe())
        # 改顺序
        all_parent_pipes.reverse()
        return all_parent_pipes

    def transform_all_parent(self, x, show_process=False, **kwargs):
        """
        将所有前置的pipe顺序执行一次transform

        :param x:
        :param show_process:
        :param kwargs:
        :return:
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
        将所有前置的pipe顺序执行一次transform_single

        :param x:
        :param show_process:
        :param kwargs:
        :return:
        """
        x_ = copy.deepcopy(x)
        all_parent_pipes = self.get_all_parent_pipes()
        for current_parent_pipe in tqdm(all_parent_pipes) if show_process else all_parent_pipes:
            if show_process:
                print(current_parent_pipe.name)
            x_ = current_parent_pipe.transform(x_, **kwargs)
        return x_
