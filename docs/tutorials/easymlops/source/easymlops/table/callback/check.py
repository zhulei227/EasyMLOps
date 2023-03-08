from easymlops.table.core import TablePipeObjectBase, TablePipeLine, dataframe_type
from easymlops.table.utils import *
import copy
import datetime
import pandas as pd

"""
对Table pipe的性能，一致性，空值，极端值，类型反转等进行检测
"""


def run_batch_single_transform(module: TablePipeObjectBase, s_):
    """
    分别获取batch_transform和single_transform预测结果
    """
    s = copy.deepcopy(s_)
    batch_transform = module.transform(s)  # 注意:transform可能会修改s自身数据
    single_transform = []
    single_operate_times = []
    s = copy.deepcopy(s_)
    detector = CpuMemDetector()
    detector.start()
    for record in s.to_dict("record"):
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


def check_shape(module: TablePipeObjectBase, batch_transform, single_transform):
    """
    检测shape是否一致
    """
    if batch_transform.shape != single_transform.shape:
        raise Exception(
            "({})  module output shape error , batch shape is {} , single  shape is {}".format(
                module.name, batch_transform.shape, single_transform.shape))


def check_columns(self, batch_transform, single_transform):
    """
    检测输出的column是否一致
    """
    for col in batch_transform.columns:
        if col not in single_transform.columns:
            raise Exception(
                "({})  module output column error,the batch output column {} not in single output".format(
                    self.name, col))


def check_data_type(module: TablePipeObjectBase, batch_transform, single_transform):
    """
    检测数据类型是否一致
    """
    for col in batch_transform.columns:
        if not module.skip_check_transform_type and not leak_check_type_is_same(module, batch_transform[col].dtype,
                                                                                single_transform[col].dtype):
            raise Exception(
                "({})  module output type error,the column {} in batch is {},while in single is {}".format(
                    module.name, col, batch_transform[col].dtype, single_transform[col].dtype))


def check_data_same(module: TablePipeObjectBase, batch_transform, single_transform):
    """
    检测数值是否一致
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


def check_transform_function(module: TablePipeObjectBase, s_):
    """
    自动测试批量接口和单条数据接口
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


def leak_check_type_is_same(module: TablePipeObjectBase, type1, type2):
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


def leak_check_value_is_same(module: TablePipeObjectBase, ser1, ser2):
    # 弱化检测
    if module.leak_check_transform_value and np.sum(ser1.astype(str) != ser2.astype(str)) == 0:
        return True
    else:
        return False


def check_transform_function_pipeline(module: TablePipeLine, x, sample=1000, return_x=False):
    x_ = copy.deepcopy(x[:min(sample, len(x))])
    for model in module.models:
        if issubclass(model.__class__, TablePipeLine):
            # 如果是Pipe类型，则返回transform后的x供下一个Pipe模块调用
            x_ = check_transform_function_pipeline(model, x_, return_x=True)
        else:
            # 非Pipe以及其子类，默认都会返回transform后的x
            x_ = check_transform_function(model, x_)
    if return_x:
        return x_


def run_transform_and_check(module: TablePipeLine, x_, check_col, check_type, check_value):
    """
    分别跑transform和transform_single后做检测输出是否一致
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


def check_two_batch_transform_same(module: TablePipeLine, cur_batch_transform, pre_batch_transform, check_col,
                                   check_type,
                                   check_cur_value, check_pre_value):
    """
    检验俩数据是否一致
    """
    try:
        check_data_same(module, cur_batch_transform, pre_batch_transform)
    except Exception as e:
        print(e)
        print("column: \033[1;43m[{}]\033[0m check {} fail, "
              "when input \033[1;43m[{}]\033[0m or \033[1;43m[{}]\033[0m, "
              "there will be different final output".format(check_col, check_type, check_cur_value,
                                                            check_pre_value))


def check_null_value(module: TablePipeLine, x: dataframe_type, sample=100,
                     null_values=None):
    """
    检验空值情况下的数据一致性
    """
    if null_values is None:
        null_values = [None, np.nan, "null", "NULL", "nan", "NaN", "", "none", "None", " "]
    cols = x.columns.tolist()
    for col in cols:
        total_single_operate_times = []
        x_ = copy.deepcopy(x[:min(sample, len(x))])
        # 真删除
        del x_[col]
        pre_null = "__delete__"
        pre_batch_transform, pre_single_transform, single_operate_times, max_cpu_percent, min_used_mem, max_used_mem = \
            run_transform_and_check(module, x_, check_col=col, check_type="null", check_value=pre_null)
        total_single_operate_times.extend(single_operate_times)
        # 检测后面各类null值
        for null_value in null_values:
            cur_null = null_value
            x_ = copy.deepcopy(x[:min(sample, len(x))])
            x_[col] = null_value
            cur_batch_transform, cur_single_transform, single_operate_times, max_cpu_percent, min_used_mem, max_used_mem = \
                run_transform_and_check(module, x_, check_col=col, check_type="null", check_value=cur_null)

            # 上一个空和当前空对比
            check_two_batch_transform_same(module, cur_batch_transform, pre_batch_transform, check_col=col,
                                           check_type="null", check_cur_value=cur_null,
                                           check_pre_value=pre_null)
            pre_batch_transform, pre_single_transform, pre_null = cur_batch_transform, cur_single_transform, cur_null
            total_single_operate_times.extend(single_operate_times)
        print(
            "column:[{}] check [null value] complete,speed:[{}ms]/it,cpu:[{}%],memory:[{}K]"
                .format(col, np.round(np.mean(total_single_operate_times), 2), max_cpu_percent,
                        int(max_used_mem - min_used_mem)))


def check_extreme_value(module: TablePipeLine, x: dataframe_type, sample=100,
                        number_extreme_values=None,
                        category_extreme_values=None):
    """
    检验输入极端值的情况下，还能否有正常的output
    """
    if number_extreme_values is None:
        number_extreme_values = [np.inf, 0.0, -1, 1, -1e-7, 1e-7, np.iinfo(np.int64).min,
                                 np.iinfo(np.int64).max, np.finfo(np.float64).min,
                                 np.finfo(np.float64).max]
    if category_extreme_values is None:
        category_extreme_values = ["", "null", None, "1.0", "0.0", "-1.0", "-1", "none", "NaN", "None"]
    cols = x.columns.tolist()
    for col in cols:
        total_single_operate_times = []
        total_min_used_mem = np.iinfo(np.int64).max
        total_max_used_mem = np.iinfo(np.int64).min
        total_max_cpu_percent = 0
        if "int" in str(x[col].dtype).lower() or "float" in str(x[col].dtype).lower():
            extreme_values = number_extreme_values
        else:
            extreme_values = category_extreme_values
        # 检测后面各类null值
        for extreme_value in extreme_values:
            x_ = copy.deepcopy(x[:min(sample, len(x))])
            x_[col] = extreme_value
            batch_transform, single_transform, single_operate_times, max_cpu_percent, min_used_mem, max_used_mem = \
                run_transform_and_check(module, x_, check_col=col, check_type="extreme", check_value=extreme_value)

            total_single_operate_times.extend(single_operate_times)
            total_max_cpu_percent = max(total_max_cpu_percent, max_cpu_percent)
            total_min_used_mem = min(total_min_used_mem, min_used_mem)
            total_max_used_mem = max(total_max_used_mem, max_used_mem)
        print("column:[{}] check [extreme value] complete,speed:[{}ms]/it,cpu:[{}%],memory:[{}K]"
              .format(col, np.round(np.mean(total_single_operate_times), 2), int(total_max_cpu_percent),
                      int(total_max_used_mem - total_min_used_mem)))
    # 全局测试
    # 1.纯空测试
    module.transform_single({})
    # 2.其余各类值全部赋值测试
    total_single_operate_times = []
    total_min_used_mem = np.iinfo(np.int64).max
    total_max_used_mem = np.iinfo(np.int64).min
    total_max_cpu_percent = 0
    for extreme_value in number_extreme_values + category_extreme_values:
        x_ = copy.deepcopy(x[:min(sample, len(x))])
        for col in x_.columns:
            x_[col] = extreme_value
        batch_transform, single_transform, single_operate_times, max_cpu_percent, min_used_mem, max_used_mem = \
            run_transform_and_check(module, x_, check_col="__all__", check_type="extreme", check_value=extreme_value)
        total_single_operate_times.extend(single_operate_times)
        total_max_cpu_percent = max(total_max_cpu_percent, max_cpu_percent)
        total_min_used_mem = min(total_min_used_mem, min_used_mem)
        total_max_used_mem = max(total_max_used_mem, max_used_mem)
    print("column:[__all__] check [extreme value] complete,speed:[{}ms]/it,cpu:[{}%],memory:[{}K]"
          .format(np.round(np.mean(total_single_operate_times), 2), int(total_max_cpu_percent),
                  int(total_max_used_mem - total_min_used_mem)))


def check_inverse_dtype(module: TablePipeLine, x: dataframe_type, sample=100,
                        number_inverse_values=None,
                        category_inverse_values=None):
    """
    检验反转数据类型，能否有正常的output
    """
    if number_inverse_values is None:
        number_inverse_values = ["", "null", None, "1.0", "0.0", "-1.0", "-1"]
    if category_inverse_values is None:
        category_inverse_values = [0.0, -1, 1, -1e-7, 1e-7, np.iinfo(np.int64).min,
                                   np.iinfo(np.int64).max, np.finfo(np.float64).min, np.finfo(np.float64).max]
    cols = x.columns.tolist()
    for col in cols:
        total_single_operate_times = []
        total_min_used_mem = np.iinfo(np.int64).max
        total_max_used_mem = np.iinfo(np.int64).min
        total_max_cpu_percent = 0
        if "int" in str(x[col].dtype).lower() or "float" in str(x[col].dtype).lower():
            inverse_values = number_inverse_values
        else:
            inverse_values = category_inverse_values
        # 检测后面各类inverse值
        for inverse_value in inverse_values:
            x_ = copy.deepcopy(x[:min(sample, len(x))])
            x_[col] = inverse_value
            batch_transform, single_transform, single_operate_times, max_cpu_percent, min_used_mem, max_used_mem = \
                run_transform_and_check(module, x_, check_type="inverse", check_col=col, check_value=inverse_value)
            total_single_operate_times.extend(single_operate_times)
            total_max_cpu_percent = max(total_max_cpu_percent, max_cpu_percent)
            total_min_used_mem = min(total_min_used_mem, min_used_mem)
            total_max_used_mem = max(total_max_used_mem, max_used_mem)
        print("column:[{}] check [inverse type] complete,speed:[{}ms]/it,cpu:[{}%],memory:[{}K]"
              .format(col, np.round(np.mean(total_single_operate_times), 2), int(total_max_cpu_percent),
                      int(total_max_used_mem - total_min_used_mem)))


def check_int_trans_float(module: TablePipeLine, x: dataframe_type, sample=100):
    """
    将原始int数据类型转为float做测试
    """
    cols = x.columns.tolist()
    x_ = copy.deepcopy(x[:min(sample, len(x))])
    base_batch_transform, _, _, _, _, _ = \
        run_transform_and_check(module, x_, check_col="__base__", check_type="int trans float",
                                check_value="float type data")
    for col in cols:
        if "int" in str(x[col].dtype).lower():
            x_ = copy.deepcopy(x[:min(sample, len(x))])
            x_[col] = x_[col].astype(float)
            total_single_operate_times = []
            total_min_used_mem = np.iinfo(np.int64).max
            total_max_used_mem = np.iinfo(np.int64).min
            total_max_cpu_percent = 0
            float_batch_transform, _, single_operate_times, max_cpu_percent, min_used_mem, max_used_mem = \
                run_transform_and_check(module, x_, check_col=col, check_type="int trans float",
                                        check_value="float type data")
            total_single_operate_times.extend(single_operate_times)
            total_max_cpu_percent = max(total_max_cpu_percent, max_cpu_percent)
            total_min_used_mem = min(total_min_used_mem, min_used_mem)
            total_max_used_mem = max(total_max_used_mem, max_used_mem)
            # float和base对比
            check_two_batch_transform_same(module, base_batch_transform, float_batch_transform, check_col=col,
                                           check_type="int trans float", check_cur_value="__int__",
                                           check_pre_value="__float__")
            print(
                "column:[{}] check [int trans float] complete,speed:[{}ms]/it,cpu:[{}%],memory:[{}K]"
                    .format(col, np.round(np.mean(total_single_operate_times), 2), int(total_max_cpu_percent),
                            int(total_max_used_mem - total_min_used_mem)))
