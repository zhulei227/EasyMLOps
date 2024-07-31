from .pipeline import *
from easymlops.table.utils import *
import copy
import datetime
import pandas as pd

"""
对Table pipe的性能，一致性，空值，极端值，类型反转等进行检测
"""


def run_batch_single_transform(module: PipeBase, s_):
    """
    分别获取检测离线批量和在线单条的预测结果，以及统计运算性能

    :param module: 调用的pipe
    :param s_: 被调用数据
    :return:
    """
    s = copy.deepcopy(s_)
    batch_transform = module.transform(s)  # 注意:transform可能会修改s自身数据
    # 做shuffle检测
    if np.sum(s_.index != batch_transform.index) > 0:
        raise Exception(f"({module})  module's transform function will shuffle data's index,please fix it")
    single_transform = []
    single_operate_times = []
    s = copy.deepcopy(s_)
    detector = CpuMemDetector()
    detector.start()
    records = s.to_dict("records")
    if len(records) == 0:  # 全空
        records = [{}] * batch_transform.shape[0]
    for record in records:
        start_time = datetime.datetime.now()
        single_transform.append(module.transform_single(record))
        end_time = datetime.datetime.now()
        single_operate_times.append((end_time - start_time).microseconds / 1000)
    detector.end()
    max_cpu_percent, min_used_mem, max_used_mem = detector.get_status()
    single_transform = pd.DataFrame(single_transform, index=batch_transform.index)
    # 统一数据类型
    for col in single_transform.columns:
        single_transform[col] = single_transform[col].astype(batch_transform[col].dtype)
    return batch_transform, single_transform, single_operate_times, max_cpu_percent, min_used_mem, max_used_mem


def check_shape(module: PipeBase, batch_transform, single_transform):
    """
    检测离线批量和在线单条的输出的shape是否一致

    :param module:
    :param batch_transform:
    :param single_transform:
    :return:
    """
    if batch_transform.shape != single_transform.shape:
        raise Exception(
            "({})  module output shape error , batch shape is {} , single  shape is {}".format(
                module.name, batch_transform.shape, single_transform.shape))


def check_columns(self, batch_transform, single_transform):
    """
    检测离线批量和在线单条的输出的columns是否一致

    :param self:
    :param batch_transform:
    :param single_transform:
    :return:
    """
    for col in batch_transform.columns:
        if col not in single_transform.columns:
            raise Exception(
                "({})  module output column error,the batch output column {} not in single output".format(
                    self.name, col))


def check_data_type(module: PipeBase, batch_transform, single_transform):
    """
    检测离线批量和在线单条的输出的数据类型是否一致

    :param module:
    :param batch_transform:
    :param single_transform:
    :return:
    """
    for col in batch_transform.columns:
        if not module.skip_check_transform_type and not leak_check_type_is_same(module, batch_transform[col].dtype,
                                                                                single_transform[col].dtype):
            raise Exception(
                "({})  module output type error,the column {} in batch is {},while in single is {}".format(
                    module.name, col, batch_transform[col].dtype, single_transform[col].dtype))


def check_data_same(module: PipeBase, batch_transform, single_transform):
    """
    检测离线批量和在线单条的输出的数值是否一致

    :param module:
    :param batch_transform:
    :param single_transform:
    :return:
    """
    for col in batch_transform.columns:
        col_type = str(batch_transform[col].dtype)
        batch_col_values = batch_transform[col].values
        single_col_values = single_transform[col].values
        if not module.skip_check_transform_value and ("int" in col_type or "float" in col_type):
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
                np.abs(batch_col_values * 1.0 - single_col_values * 1.0) > module.transform_check_max_number_error)
            if len(error_index) > 0:
                error_info = pd.DataFrame(
                    {"error_index": np.reshape(error_index[:3], (-1,)),
                     "batch_transform": np.reshape(batch_col_values[error_index][:3], (-1,)),
                     "single_transform": np.reshape(single_col_values[error_index][:3], (-1,))})
                # 再做一次弱检测
                if leak_check_value_is_same(module, error_info["batch_transform"], error_info["single_transform"]):
                    format_info = """
        ----------------------------------------------------
        ({})  module output value is unsafe,in col \033[1;43m[{}]\033[0m,current transform_check_max_number_error is {},
        the top {} error info is \n {}
        ----------------------------------------------------"
                                """
                    print(format_info.format(module.name, col, module.transform_check_max_number_error,
                                             min(3, len(error_info)), error_info))
                else:
                    raise Exception(
                        "({}) module output value error,in col [{}],current transform_check_max_number_error is {},"
                        "the top {} error info is \n {}".format(
                            module.name, col, module.transform_check_max_number_error, min(3, len(error_info)),
                            error_info))
        elif not module.skip_check_transform_value:
            # 离散数据检测
            error_index = np.argwhere(batch_col_values != single_col_values)
            if len(error_index) > 0:
                error_info = pd.DataFrame(
                    {"error_index": np.reshape(error_index[:3], (-1,)),
                     "batch_transform": np.reshape(batch_col_values[error_index][:3], (-1,)),
                     "single_transform": np.reshape(single_col_values[error_index][:3], (-1,))})
                # 再做一次弱检测
                if leak_check_value_is_same(module, error_info["batch_transform"], error_info["single_transform"]):
                    format_info = """
        ----------------------------------------------------
        ({})  module output value is unsafe,in col \033[1;43m[{}]\033[0m
        the top {} error info is \n {}
        ----------------------------------------------------"
                                """
                    print(format_info.format(module.name, col, min(3, len(error_info)), error_info))
                else:
                    raise Exception(
                        "({})  module output value error,in col [{}] ,the top {} error info is \n {}".format(
                            module.name, col, min(3, len(error_info)), error_info))


def check_transform_function(module: PipeBase, s_):
    """
    测试离线批量和在线单条的shape,输出名称,数据类型,数值是否一致

    :param module:
    :param s_:
    :return:
    """
    # 运行batch和single transform
    batch_transform, single_transform, single_operate_times \
        , max_cpu_percent, min_used_mem, max_used_mem = run_batch_single_transform(module, s_)
    # 检验1:输出shape是否一致
    check_shape(module, batch_transform, single_transform)
    # 检验2:输出名称是否一致
    check_columns(module, batch_transform, single_transform)
    # 检验3:数据类型是否一致
    check_data_type(module, batch_transform, single_transform)
    # 检验4:数值是否一致
    check_data_same(module, batch_transform, single_transform)
    # 打印运行成功信息
    print(
        "({}) module check [transform] complete,speed:[{}ms]/it,cpu:[{}%],memory:[{}K]".format(
            module.name, np.round(np.mean(single_operate_times), 2), int(max_cpu_percent),
            int(max_used_mem - min_used_mem)))
    return batch_transform


def leak_check_type_is_same(module: PipeBase, type1, type2):
    """
    弱化数据类型后，是否一致，比如int32和int64看作同一种数据类型

    :param module:
    :param type1:
    :param type2:
    :return:
    """
    if type1 == type2:
        return True
    # 弱化检测，比如int32与int64都视为int类型
    if module.leak_check_transform_type:
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


def leak_check_value_is_same(module: PipeBase, ser1, ser2):
    """
    所有数值的检测

    :param module:
    :param ser1:
    :param ser2:
    :return:
    """
    if module.leak_check_transform_value and np.sum(ser1.astype(str) != ser2.astype(str)) == 0:
        return True
    else:
        return False


def check_transform_function_pipeline(module: TSPipeLine, x, sample=1000, return_x=False):
    """
    check_transform_function函数的pipeline版本检测

    :param module:
    :param x:
    :param sample:
    :param return_x:
    :return:
    """
    x_ = copy.deepcopy(x[:min(sample, len(x))])
    for model in module.models:
        if issubclass(model.__class__, TSPipeLine):
            # 如果是Pipe类型，则返回transform后的x供下一个Pipe模块调用
            x_ = check_transform_function_pipeline(model, x_, return_x=True)
        else:
            # 非Pipe以及其子类，默认都会返回transform后的x
            x_ = check_transform_function(model, x_)
    if return_x:
        return x_


def run_transform_and_check(module: TSPipeLine, x_, check_col, check_type, check_value):
    """
    分别跑transform和transform_single后做检测输出是否一致

    :param module:
    :param x_:
    :param check_col:
    :param check_type:
    :param check_value:
    :return:
    """
    x = copy.deepcopy(x_)
    try:
        batch_transform, single_transform, single_operate_times, max_cpu_percent, min_used_mem, max_used_mem = \
            run_batch_single_transform(module, x)
    except Exception as e:
        print(e)
        raise Exception("column: \033[1;43m[{}]\033[0m check {} fail, "
                        "if input \033[1;43m[{}]\033[0m, "
                        "there will be error!".format(check_col, check_type, check_value))
    try:
        check_data_same(module, batch_transform, single_transform)
    except Exception as e:
        print(e)
        print("column: \033[1;43m[{}]\033[0m check {} fail, "
              "if input \033[1;43m[{}]\033[0m, "
              "the batch and single transform function will have different final output"
              .format(check_col, check_type, check_value))
    return batch_transform, single_transform, single_operate_times, max_cpu_percent, min_used_mem, max_used_mem


def check_two_batch_transform_same(module: TSPipeLine, cur_batch_transform, pre_batch_transform, check_col,
                                   check_type,
                                   check_cur_value, check_pre_value):
    """
    检验俩数据是否一致

    :param module:
    :param cur_batch_transform:
    :param pre_batch_transform:
    :param check_col:
    :param check_type:
    :param check_cur_value:
    :param check_pre_value:
    :return:
    """
    try:
        check_data_same(module, cur_batch_transform, pre_batch_transform)
    except Exception as e:
        print(e)
        print("column: \033[1;43m[{}]\033[0m check {} fail, "
              "when input \033[1;43m[{}]\033[0m or \033[1;43m[{}]\033[0m, "
              "there will be different final output".format(check_col, check_type, check_cur_value,
                                                            check_pre_value))
