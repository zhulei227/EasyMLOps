import copy
import numpy as np
import pandas
import datetime
import warnings

import pandas as pd

warnings.filterwarnings("ignore")
dataframe_type = pandas.core.frame.DataFrame
series_type = pandas.core.series.Series
dict_type = dict


def check_dict_type(func):
    def wrapper(*args, **kwargs):
        assert type(args[1]) == dict_type
        return func(*args, **kwargs)

    return wrapper


def check_dataframe_type(func):
    def wrapper(*args, **kwargs):
        assert type(args[1]) == dataframe_type
        return func(*args, **kwargs)

    return wrapper


def fit_wrapper(func):
    def wrapper(*args, **kwargs):
        print("Deprecated this function [{}] for better flexibility,please use [{}] replace".format("fit_wrapper",
                                                                                                    "_fit"))
        args = list(args)
        pipe_object = args[0]
        data: dataframe_type = args[1]
        # 调用before_fit
        data = pipe_object.before_fit(data)
        args[1] = data
        args = tuple(args)
        # 调用fit
        pipe_object = func(*args, **kwargs)
        # 调用after_fit
        pipe_object.after_fit()
        return pipe_object

    return wrapper


def transform_wrapper(func):
    def wrapper(*args, **kwargs):
        print("Deprecated this function [{}] for better flexibility,please use [{}] replace".format("transform_wrapper",
                                                                                                    "_transform"))
        args = list(args)
        pipe_object = args[0]
        data: dataframe_type = args[1]
        # 调用before_transform
        data = pipe_object.before_transform(data)
        args[1] = data
        args = tuple(args)
        # 调用transform
        data = func(*args, **kwargs)
        # 调用after_transform
        return pipe_object.after_transform(data)

    return wrapper


def transform_single_wrapper(func):
    def wrapper(*args, **kwargs):
        print("Deprecated this function [{}] for better flexibility,please use [{}] replace".format(
            "transform_single_wrapper",
            "_transform_single"))
        args = list(args)
        pipe_object = args[0]
        data: dict_type = args[1]
        # 调用before_transform_single
        data = pipe_object.before_transform_single(data)
        args[1] = data
        args = tuple(args)
        # 调用transform
        data = func(*args, **kwargs)
        # 调用after_transform
        return pipe_object.after_transform_single(data)

    return wrapper


class SuperPipeObject(object):
    """
    name:模块名称，如果为空默认为self.__class__
    input_col_names:输入数据的columns
    output_col_names:输出数据的columns
    transform_check_max_number_error:在check_transform_function时允许的最大数值误差
    skip_check_transform_type:在check_transform_function时是否跳过类型检测(针对一些特殊数据类型，比如稀疏矩阵)
    leak_check_transform_type:弱化类型检测，比如将int32和int64都视为int类型
    leak_check_transform_value:弱化检测值，将None视为相等
    copy_transform_data:transform阶段是否要copy一次数据
    prefix:是否为输出添加一个前缀,默认None,不添加
    """

    def __init__(self, name=None, transform_check_max_number_error=1e-5, skip_check_transform_type=False,
                 skip_check_transform_value=False, leak_check_transform_type=True, leak_check_transform_value=True,
                 copy_transform_data=True, prefix=None, **kwargs):
        super().__init__(**kwargs)
        if name is None:
            name = self.__class__
        self.name = name
        self.input_col_names = None
        self.output_col_names = None
        self.transform_check_max_number_error = transform_check_max_number_error
        self.skip_check_transform_type = skip_check_transform_type
        self.skip_check_transform_value = skip_check_transform_value
        self.leak_check_transform_type = leak_check_transform_type
        self.leak_check_transform_value = leak_check_transform_value
        self.copy_transform_data = copy_transform_data
        self.prefix = prefix

    def fit(self, s: dataframe_type):
        """
        fit:依次调用before_fit,_fit,after_fit
        """
        self._fit(self.before_fit(s))
        return self.after_fit()

    def before_fit(self, s: dataframe_type) -> dataframe_type:
        """
        fit前预操作
        """
        assert type(s) == dataframe_type
        self.input_col_names = s.columns.tolist()
        return s

    def _fit(self, s: dataframe_type):
        return self

    def after_fit(self):
        """
        fit后操作
        """
        return self

    def transform(self, s: dataframe_type) -> dataframe_type:
        """
        批量接口:依次调用before_transform,_transform,after_transform
        """
        return self.after_transform(self._transform(self.before_transform(s)))

    def before_transform(self, s: dataframe_type) -> dataframe_type:
        assert type(s) == dataframe_type
        if self.copy_transform_data:
            s_ = s[self.input_col_names]
        else:
            s = s[self.input_col_names]
            s_ = s
        return s_

    def _transform(self, s: dataframe_type) -> dataframe_type:
        return s

    def after_transform(self, s: dataframe_type) -> dataframe_type:
        # 是否改名
        if self.prefix is not None:
            s.columns = ["{}_{}".format(self.prefix, col) for col in s.columns]
        # 保留output columns
        self.output_col_names = list(s.columns)
        return s

    @staticmethod
    def extract_dict(s: dict_type, keys: list) -> dict_type:
        new_s = dict()
        for key in keys:
            new_s[key] = s[key]
        return new_s

    def transform_single(self, s: dict_type) -> dict_type:
        """
        当条数据接口:调用顺序before_transform_single,_transform_single,after_transform_single
        """
        return self.after_transform_single(self._transform_single(self.before_transform_single(s)))

    def before_transform_single(self, s: dict_type) -> dict_type:
        assert type(s) == dict_type
        return self.extract_dict(s, self.input_col_names)

    def _transform_single(self, s: dict_type) -> dict_type:
        """
        当条数据接口，生产用
        """
        return s

    def after_transform_single(self, s: dict_type) -> dict_type:
        # 改名
        if self.prefix is not None:
            new_s = dict()
            for col, value in s.items():
                new_s["{}_{}".format(self.prefix, col)] = value
            s = new_s
        # 输出指定columns
        return self.extract_dict(s, self.output_col_names)

    def _run_batch_single_transform(self, s_):
        """
        分别获取batch_transform和single_transform预测
        """
        s = copy.copy(s_)
        batch_transform = self.transform(s)  # 注意:transform可能会修改s自身数据
        single_transform = []
        single_operate_times = []
        s = copy.copy(s_)
        for record in s.to_dict("record"):
            start_time = datetime.datetime.now()
            single_transform.append(self.transform_single(record))
            end_time = datetime.datetime.now()
            single_operate_times.append((end_time - start_time).microseconds / 1000)
        single_transform = pandas.DataFrame(single_transform)
        # 统一数据类型
        for col in single_transform.columns:
            single_transform[col] = single_transform[col].astype(batch_transform[col].dtype)
        return batch_transform, single_transform, single_operate_times

    def _check_shape(self, batch_transform, single_transform):
        """
        检测shape是否一致
        """
        if batch_transform.shape != single_transform.shape:
            raise Exception(
                "({})  module output shape error , batch shape is {} , single  shape is {}".format(
                    self.name, batch_transform.shape, single_transform.shape))

    def _check_columns(self, batch_transform, single_transform):
        """
        检测输出的column是否一致
        """
        for col in batch_transform.columns:
            if col not in single_transform.columns:
                raise Exception(
                    "({})  module output column error,the batch output column {} not in single output".format(
                        self.name, col))

    def _check_data_type(self, batch_transform, single_transform):
        """
        检测数据类型是否一致
        """
        for col in batch_transform.columns:
            if not self.skip_check_transform_type and not self._leak_check_type_is_same(batch_transform[col].dtype,
                                                                                        single_transform[col].dtype):
                raise Exception(
                    "({})  module output type error,the column {} in batch is {},while in single is {}".format(
                        self.name, col, batch_transform[col].dtype, single_transform[col].dtype))

    def _check_data_same(self, batch_transform, single_transform):
        """
        检测数值是否一致
        """
        for col in batch_transform.columns:
            col_type = str(batch_transform[col].dtype)
            batch_col_values = batch_transform[col].values
            single_col_values = single_transform[col].values
            if not self.skip_check_transform_value and ("int" in col_type or "float" in col_type):
                # 数值数据检测
                try:
                    batch_col_values = batch_col_values.to_dense()  # 转换为dense
                except:
                    pass
                try:
                    single_col_values = single_col_values.to_dense()  # 转换为dense
                except:
                    pass
                error_index = np.argwhere(
                    np.abs(batch_col_values - single_col_values) > self.transform_check_max_number_error)
                if len(error_index) > 0:
                    error_info = pd.DataFrame(
                        {"error_index": np.reshape(error_index[:3], (-1,)),
                         "batch_transform": np.reshape(batch_col_values[error_index][:3], (-1,)),
                         "single_transform": np.reshape(single_col_values[error_index][:3], (-1,))})
                    # 再做一次弱检测
                    if self._leak_check_value_is_same(error_info["batch_transform"], error_info["single_transform"]):
                        format_info = """
        ----------------------------------------------------
        ({})  module output value is unsafe,in col \033[1;43m[{}]\033[0m,current transform_check_max_number_error is {},
        the top {} error info is \n {}
        ----------------------------------------------------"
                                """
                        print(format_info.format(self.name, col, self.transform_check_max_number_error,
                                                 min(3, len(error_info)), error_info))
                    else:
                        raise Exception(
                            "({}) module output value error,in col [{}],current transform_check_max_number_error is {},"
                            "the top {} error info is \n {}".format(
                                self.name, col, self.transform_check_max_number_error, min(3, len(error_info)),
                                error_info))
            elif not self.skip_check_transform_value:
                # 离散数据检测
                error_index = np.argwhere(batch_col_values != single_col_values)
                if len(error_index) > 0:
                    error_info = pd.DataFrame(
                        {"error_index": np.reshape(error_index[:3], (-1,)),
                         "batch_transform": np.reshape(batch_col_values[error_index][:3], (-1,)),
                         "single_transform": np.reshape(single_col_values[error_index][:3], (-1,))})
                    # 再做一次弱检测
                    if self._leak_check_value_is_same(error_info["batch_transform"], error_info["single_transform"]):
                        format_info = """
        ----------------------------------------------------
        ({})  module output value is unsafe,in col \033[1;43m[{}]\033[0m
        the top {} error info is \n {}
        ----------------------------------------------------"
                                """
                        print(format_info.format(self.name, col, min(3, len(error_info)), error_info))
                    else:
                        raise Exception(
                            "({})  module output value error,in col [{}] ,the top {} error info is \n {}".format(
                                self.name, col, min(3, len(error_info)), error_info))

    def check_transform_function(self, s_):
        """
        自动测试批量接口和单条数据接口
        """
        # 运行batch和single transform
        batch_transform, single_transform, single_operate_times = self._run_batch_single_transform(s_)
        # 检验1:输出shape是否一致
        self._check_shape(batch_transform, single_transform)
        # 检验2:输出名称是否一致
        self._check_columns(batch_transform, single_transform)
        # 检验3:数据类型是否一致
        self._check_data_type(batch_transform, single_transform)
        # 检验4:数值是否一致
        self._check_data_same(batch_transform, single_transform)
        # 打印运行成功信息
        print("({})  module transform check [success], single transform speed:[{}]ms/it".format(self.name, np.round(
            np.mean(single_operate_times), 2)))
        return batch_transform

    def _leak_check_type_is_same(self, type1, type2):
        if type1 == type2:
            return True
        # 弱化检测，比如int32与int64都视为int类型
        if self.leak_check_transform_type:
            type1 = str(type1)
            type2 = str(type2)
            if "int" in type1 and "int" in type2:
                return True
            elif "float" in type1 and "float" in type2:
                return True
            elif ("object" in type1 or "category" in type1 or "str" in type1) and (
                    "object" in type2 or "category" in type2 or "str" in type2):
                return True
        return False

    def _leak_check_value_is_same(self, ser1, ser2):
        # 弱化检测
        if self.leak_check_transform_value and np.sum(ser1.astype(str) != ser2.astype(str)) == 0:
            return True
        else:
            return False

    def get_params(self):
        return self._get_params()

    def _get_params(self) -> dict_type:
        return {"name": self.name,
                "input_col_names": self.input_col_names,
                "output_col_names": self.output_col_names,
                "transform_check_max_number_error": self.transform_check_max_number_error,
                "skip_check_transform_type": self.skip_check_transform_type,
                "skip_check_transform_value": self.skip_check_transform_value,
                "copy_transform_data": self.copy_transform_data,
                "prefix": self.prefix}

    def set_params(self, params: dict_type):
        self._set_params(params)

    def _set_params(self, params: dict_type):
        self.name = params["name"]
        self.input_col_names = params["input_col_names"]
        self.output_col_names = params["output_col_names"]
        self.transform_check_max_number_error = params["transform_check_max_number_error"]
        self.skip_check_transform_type = params["skip_check_transform_type"]
        self.skip_check_transform_value = params["skip_check_transform_value"]
        self.copy_transform_data = params["copy_transform_data"]
        self.prefix = params["prefix"]


class PipeObject(SuperPipeObject):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def after_fit(self):
        super().after_fit()
        self.get_params()
        return self

    def get_params(self) -> dict_type:
        # 逐步获取父类的_get_params
        current_class = self.__class__
        all_params = [self._get_params()]
        while issubclass(current_class, PipeObject):
            all_params.append(super(current_class, self)._get_params())
            self.check_key_conflict(all_params[-1], all_params[-2], current_class, current_class.__base__)
            current_class = current_class.__base__
        all_params.reverse()
        # 逆向聚合参数
        combine_params = dict()
        for params in all_params:
            combine_params.update(params)
        # 获取当前参数
        return combine_params

    def _get_params(self):
        return {}

    def set_params(self, params: dict_type):
        # 逐步获取父类class
        current_class = self.__class__
        super_classes = []
        while issubclass(current_class, PipeObject):
            super_classes.append(super(current_class, self))
            current_class = current_class.__base__
        super_classes.reverse()
        for super_class in super_classes:
            super_class._set_params(params)
        # 设置当前类
        self._set_params(params)

    def _set_params(self, params: dict_type):
        pass

    @staticmethod
    def check_key_conflict(param1: dict_type, param2: dict_type, class1, class2):
        # 检测name是否冲突
        same_param_names = list(set(param1.keys()) & set(param2.keys()))
        if len(same_param_names) > 0:
            print("the {} and {} use same parameter names \033[1;43m[{}]\033[0m,please check if conflict,".format
                  (class1, class2, same_param_names))
