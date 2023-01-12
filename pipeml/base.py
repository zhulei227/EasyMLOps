import copy
import numpy as np
import pandas
import datetime
import warnings

import pandas as pd

warnings.filterwarnings("ignore")
series_type = pandas.core.series.Series
dataframe_type = pandas.core.frame.DataFrame
dict_type = dict


def check_series_type(func):
    def wrapper(*args, **kwargs):
        assert type(args[1]) == series_type
        return func(*args, **kwargs)

    return wrapper


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


class PipeObject(object):
    """
    name:模块名称，如果为空默认为self.__class__
    input_col_names:输入数据的columns
    output_col_names:输出数据的columns
    transform_check_max_number_error:在auto_check_transform时允许的最大数值误差
    skip_check_transform_type:在auto_check_transform时是否跳过类型检测(针对一些特殊数据类型，比如稀疏矩阵)
    leak_check_transform_type:弱化类型检测，比如将int32和int64都视为int类型
    leak_check_transform_value:弱化检测值，将None视为相等
    """

    def __init__(self, name=None, transform_check_max_number_error=1e-5, skip_check_transform_type=False,
                 skip_check_transform_value=False, leak_check_transform_type=True, leak_check_transform_value=True):
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

    def fit(self, s):
        return self

    def transform(self, s):
        """
        批量接口
        """
        raise Exception("need to implement")

    def transform_single(self, s: dict_type):
        """
        当条数据接口，生产用
        """
        raise Exception("need to implement")

    def auto_check_transform(self, s_):
        """
        自动测试批量接口和单条数据接口
        """
        # 预测数据
        s = copy.copy(s_)
        batch_transform = self.transform(s)  # 注意:transform可能会修改s自身数据
        single_transform = []
        single_operate_times = []
        s = copy.copy(s_)
        if type(s) == dataframe_type:
            for record in s.to_dict("record"):
                start_time = datetime.datetime.now()
                single_transform.append(self.transform_single(record))
                end_time = datetime.datetime.now()
                single_operate_times.append((end_time - start_time).microseconds / 1000)
            single_transform = pandas.DataFrame(single_transform)
        else:  # series_type
            for value in s.values:
                start_time = datetime.datetime.now()
                single_transform.append(self.transform_single({self.input_col_names[0]: value}))
                end_time = datetime.datetime.now()
                single_operate_times.append((end_time - start_time).microseconds / 1000)
            single_transform = pandas.DataFrame(single_transform)
        # 都转换为dataframe,方便统一检测
        new_batch_transform = batch_transform
        if type(batch_transform) == series_type:
            new_batch_transform = pandas.DataFrame({self.output_col_names[0]: batch_transform.values})
        # 检验
        # 检验1:输出shape是否一致
        if new_batch_transform.shape != single_transform.shape:
            raise Exception(
                "({})  module shape error , batch shape is {} , single  shape is {}".format(
                    self.name, new_batch_transform.shape, single_transform.shape))

        for col in new_batch_transform.columns:
            # 检验2:输出名称是否一致
            if col not in single_transform.columns:
                raise Exception(
                    "({})  module column error,the batch output column {} not in single output".format(
                        self.name, col))
            # 检验3:数据类型是否一致
            if not self.skip_check_transform_type and not self._leak_check_type_is_same(new_batch_transform[col].dtype,
                                                                                        single_transform[col].dtype):
                raise Exception(
                    "({})  module type error,the column {} in batch is {},while in single is {}".format(
                        self.name, col, new_batch_transform[col].dtype, single_transform[col].dtype))
            # 检验4:数值是否一致
            col_type = str(new_batch_transform[col].dtype)
            batch_col_values = new_batch_transform[col].values
            single_col_values = single_transform[col].values
            if not self.skip_check_transform_value and ("int" in col_type or "float" in col_type):
                try:
                    batch_col_values = batch_col_values.to_dense()  # 转换为dense
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
({})  module value is unsafe,in col \033[1;43m[{}]\033[0m,current transform_check_max_number_error is {},
the top {} error info is \n {}
----------------------------------------------------"
                        """
                        print(format_info.format(self.name, col, self.transform_check_max_number_error,
                                                 min(3, len(error_info)), error_info))
                    else:
                        raise Exception(
                            "({})  module value error,in col [{}],current transform_check_max_number_error is {} ,"
                            "the top {} error info is \n {}".format(
                                self.name, col, self.transform_check_max_number_error, min(3, len(error_info)),
                                error_info))
            elif not self.skip_check_transform_value:
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
({})  module value is unsafe,in col \033[1;43m[{}]\033[0m
the top {} error info is \n {}
----------------------------------------------------"
                        """
                        print(format_info.format(self.name, col, min(3, len(error_info)), error_info))
                    else:
                        raise Exception(
                            "({})  module value error,in col [{}] ,the top {} error info is \n {}".format(
                                self.name, col, min(3, len(error_info)), error_info))
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

    def get_params(self) -> dict_type:
        return {"name": self.name,
                "input_col_names": self.input_col_names,
                "output_col_names": self.output_col_names,
                "transform_check_max_number_error": self.transform_check_max_number_error,
                "skip_check_transform_type": self.skip_check_transform_type,
                "skip_check_transform_value": self.skip_check_transform_value}

    def set_params(self, params: dict_type):
        self.name = params["name"]
        self.input_col_names = params["input_col_names"]
        self.output_col_names = params["output_col_names"]
        self.transform_check_max_number_error = params["transform_check_max_number_error"]
        self.skip_check_transform_type = params["skip_check_transform_type"]
        self.skip_check_transform_value = params["skip_check_transform_value"]
