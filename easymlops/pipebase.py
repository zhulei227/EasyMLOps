from .base import *
import copy
from tqdm import tqdm
import pickle


class Pipe(PipeObject):
    def __init__(self, run_to_layer=None, **kwargs):
        super().__init__(**kwargs)
        self.models = []
        self.run_to_layer = run_to_layer

    def pipe(self, model):
        self.models.append(model)
        return self

    def __getitem__(self, target_layer):
        for current_layer_deep, model in enumerate(self.models):
            if self._match_layer(current_layer_deep, model.name, target_layer):
                return model

    def fit(self, x, show_process=False):
        x_ = copy.deepcopy(x)
        if show_process:
            for model in tqdm(self.models):
                print(model.name)
                x_ = model.fit(x_).transform(x_)
        else:
            for model in self.models:
                x_ = model.fit(x_).transform(x_)
        return self

    def _match_layer(self, current_layer_deep, current_layer_name, target_layer):
        if target_layer is not None:
            if type(target_layer) == int and target_layer < 0:
                target_layer = len(self.models) + target_layer
            if type(target_layer) == int and current_layer_deep == target_layer:
                return True
            if type(target_layer) == str and current_layer_name == target_layer:
                return True
        return False

    def transform(self, x, show_process=False, run_to_layer=None):
        run_to_layer = self.run_to_layer if run_to_layer is None else run_to_layer
        x_ = copy.deepcopy(x)
        if show_process:
            for current_layer_deep, model in tqdm(enumerate(self.models)):
                print(model.name)
                x_ = model.transform(x_)
                if self._match_layer(current_layer_deep, model.name, run_to_layer):
                    break
        else:
            for current_layer_deep, model in enumerate(self.models):
                x_ = model.transform(x_)
                if self._match_layer(current_layer_deep, model.name, run_to_layer):
                    break
        return x_

    def transform_single(self, x, show_process=False, run_to_layer=None, logger=None, log_base_dict: dict_type = None):
        run_to_layer = self.run_to_layer if run_to_layer is None else run_to_layer
        x_ = copy.deepcopy(x)
        self._save_log(logger, log_base_dict, 0, x)
        if show_process:
            for current_layer_deep, model in tqdm(enumerate(self.models)):
                print(model.name)
                x_ = model.transform_single(x_)
                self._save_log(logger, log_base_dict, current_layer_deep + 1, x_)
                if self._match_layer(current_layer_deep, model.name, run_to_layer):
                    break
        else:
            for current_layer_deep, model in enumerate(self.models):
                x_ = model.transform_single(x_)
                self._save_log(logger, log_base_dict, current_layer_deep + 1, x_)
                if self._match_layer(current_layer_deep, model.name, run_to_layer):
                    break
        return x_

    @staticmethod
    def _save_log(logger, log_base_dict, step, info):
        """
        :param logger: 主要提供,logger.info方法，最好异步，不阻塞主程序
        :param log_base_dict: 续保保存基本信息，比如涉及到该条数据的id信息等
        :param step: 所在pipeline的第几步
        :param info: 该步的输出
        :return:
        """
        if log_base_dict is None:
            log_base_dict = dict()
        if logger is not None:
            log_info = copy.deepcopy(log_base_dict)
            log_info.update({"step": step, "info": copy.deepcopy(info)})
            logger.info(log_info)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump([model.get_params() for model in self.models], f)

    def load(self, path):
        with open(path, "rb") as f:
            for i, params in enumerate(pickle.load(f)):
                self.models[i].set_params(params)

    def get_params(self) -> dict:
        params = [model.get_params() for model in self.models]
        return {"params": params}

    def set_params(self, params: dict):
        params = params["params"]
        for i, param in enumerate(params):
            self.models[i].set_params(param)

    def check_transform_function(self, x, sample=1000, return_x=False):
        x_ = copy.deepcopy(x[:min(sample, len(x))])
        for model in self.models:
            if issubclass(model.__class__, Pipe):
                # 如果是Pipe类型，则返回transform后的x供下一个Pipe模块调用
                x_ = model.check_transform_function(x_, return_x=True)
            else:
                # 非Pipe以及其子类，默认都会返回transform后的x
                x_ = model.check_transform_function(x_)
        if return_x:
            return x_

    def _switch_show_check_detail(self, types=None, show_open=False):
        if types is not None:
            pipe_list = []
            pipe_list.extend(self.models)
            # 指定类的
            while len(pipe_list) > 0:
                model = pipe_list.pop()
                for model_type in types:
                    if str(model_type).lower() in str(model.__class__).lower():
                        model.show_check_detail = show_open
                if hasattr(model, "models") and issubclass(model.models[0].__class__, PipeObject):
                    pipe_list.extend(model.models)

    def _run_transform_and_check(self, x_, check_col, check_type, check_value, skip_show_check_detail_types=None):
        """
        分别跑transform和transform_single后做检测输出是否一致
        """
        x = copy.deepcopy(x_)
        self._switch_show_check_detail(types=skip_show_check_detail_types, show_open=False)
        try:
            batch_transform, single_transform, single_operate_times = \
                self._run_batch_single_transform(x)
        except Exception as e:
            print(e)
            raise Exception("column: \033[1;43m[{}]\033[0m check {} fail, "
                            "if input \033[1;43m[{}]\033[0m, "
                            "there will be error!".format(check_col, check_type, check_value))
        try:
            self._check_data_same(batch_transform, single_transform)
        except Exception as e:
            print(e)
            print("column: \033[1;43m[{}]\033[0m check {} fail, "
                  "if input \033[1;43m[{}]\033[0m, "
                  "the batch and single transform function will have different final output"
                  .format(check_col, check_type, check_value))
        # 恢复
        self._switch_show_check_detail(types=skip_show_check_detail_types, show_open=True)
        return batch_transform, single_transform, single_operate_times

    def _check_two_batch_transform_same(self, cur_batch_transform, pre_batch_transform, check_col, check_type,
                                        check_cur_value, check_pre_value):
        """
        检验俩数据是否一致
        """
        try:
            self._check_data_same(cur_batch_transform, pre_batch_transform)
        except Exception as e:
            print(e)
            print("column: \033[1;43m[{}]\033[0m check {} fail, "
                  "when input \033[1;43m[{}]\033[0m or \033[1;43m[{}]\033[0m, "
                  "there will be different final output".format(check_col, check_type, check_cur_value,
                                                                check_pre_value))

    def check_null_value(self, x: dataframe_type, sample=100,
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
            pre_batch_transform, pre_single_transform, single_operate_times = \
                self._run_transform_and_check(x_, check_col=col, check_type="null", check_value=pre_null,
                                              skip_show_check_detail_types=["FixInput"])
            total_single_operate_times.extend(single_operate_times)
            # 检测后面各类null值
            for null_value in null_values:
                cur_null = null_value
                x_ = copy.deepcopy(x[:min(sample, len(x))])
                x_[col] = null_value
                cur_batch_transform, cur_single_transform, single_operate_times = \
                    self._run_transform_and_check(x_, check_col=col, check_type="null", check_value=cur_null,
                                                  skip_show_check_detail_types=["FixInput"])

                # 上一个空和当前空对比
                self._check_two_batch_transform_same(cur_batch_transform, pre_batch_transform, check_col=col,
                                                     check_type="null", check_cur_value=cur_null,
                                                     check_pre_value=pre_null)
                pre_batch_transform, pre_single_transform, pre_null = cur_batch_transform, cur_single_transform, cur_null
                total_single_operate_times.extend(single_operate_times)
            print("column: [{}] check null value complete, total single transform speed:[{}]ms/it".format(col, np.round(
                np.mean(total_single_operate_times), 2)))

    def check_extreme_value(self, x: dataframe_type, sample=100,
                            number_extreme_values=None,
                            category_extreme_values=None):
        """
        检验输入极端值的情况下，还能否有正常的output
        """
        if number_extreme_values is None:
            number_extreme_values = [np.inf, 0.0, -1, 1, -1e-7, 1e-7, np.finfo(np.float64).min,
                                     np.finfo(np.float64).max]
        if category_extreme_values is None:
            category_extreme_values = ["", "null", None, "1.0", "0.0", "-1.0", "-1", "none", "NaN", "None"]
        cols = x.columns.tolist()
        for col in cols:
            total_single_operate_times = []
            if "int" in str(x[col].dtype).lower() or "float" in str(x[col].dtype).lower():
                extreme_values = number_extreme_values
            else:
                extreme_values = category_extreme_values
            # 检测后面各类null值
            for extreme_value in extreme_values:
                x_ = copy.deepcopy(x[:min(sample, len(x))])
                x_[col] = extreme_value
                batch_transform, single_transform, single_operate_times = \
                    self._run_transform_and_check(x_, check_col=col, check_type="extreme", check_value=extreme_value)

                total_single_operate_times.extend(single_operate_times)
            print("column: [{}] check extreme value complete, total single transform speed:[{}]ms/it"
                  .format(col, np.round(np.mean(total_single_operate_times), 2)))
        # 全局测试
        # 1.纯空测试
        self._switch_show_check_detail(types=["FixInput"], show_open=False)
        self.transform_single({})
        self._switch_show_check_detail(types=["FixInput"], show_open=True)
        # 2.其余各类值全部赋值测试
        total_single_operate_times = []
        for extreme_value in number_extreme_values + category_extreme_values:
            x_ = copy.deepcopy(x[:min(sample, len(x))])
            for col in x_.columns:
                x_[col] = extreme_value
            batch_transform, single_transform, single_operate_times = \
                self._run_transform_and_check(x_, check_col="__all__", check_type="extreme", check_value=extreme_value)
            total_single_operate_times.extend(single_operate_times)
        print("[__all__] columns set the same extreme value complete,total single transform speed:[{}]ms/it"
              .format(np.round(np.mean(total_single_operate_times), 2)))

    def check_inverse_dtype(self, x: dataframe_type, sample=100,
                            number_inverse_values=None,
                            category_inverse_values=None):
        """
        检验反转数据类型，能否有正常的output
        """
        if number_inverse_values is None:
            number_inverse_values = ["", "null", None, "1.0", "0.0", "-1.0", "-1"]
        if category_inverse_values is None:
            category_inverse_values = [0.0, -1, 1, -1e-7, 1e-7, np.finfo(np.float64).min, np.finfo(np.float64).max]
        cols = x.columns.tolist()
        for col in cols:
            total_single_operate_times = []
            if "int" in str(x[col].dtype).lower() or "float" in str(x[col].dtype).lower():
                inverse_values = number_inverse_values
            else:
                inverse_values = category_inverse_values
            # 检测后面各类inverse值
            for inverse_value in inverse_values:
                x_ = copy.deepcopy(x[:min(sample, len(x))])
                x_[col] = inverse_value
                batch_transform, single_transform, single_operate_times = \
                    self._run_transform_and_check(x_, check_type="inverse", check_col=col, check_value=inverse_value)
                total_single_operate_times.extend(single_operate_times)
            print("column: [{}] check inverse value complete, total single transform speed:[{}]ms/it"
                  .format(col, np.round(np.mean(total_single_operate_times), 2)))

    def check_int_trans_float(self, x: dataframe_type, sample=100):
        """
        将原始int数据类型转为float做测试
        """
        cols = x.columns.tolist()
        x_ = copy.deepcopy(x[:min(sample, len(x))])
        base_batch_transform, _, _ = \
            self._run_transform_and_check(x_, check_col="__base__", check_type="int trans float",
                                          check_value="float type data")
        for col in cols:
            if "int" in str(x[col].dtype).lower():
                x_ = copy.deepcopy(x[:min(sample, len(x))])
                x_[col] = x_[col].astype(float)
                total_single_operate_times = []
                float_batch_transform, _, single_operate_times = \
                    self._run_transform_and_check(x_, check_col=col, check_type="int trans float",
                                                  check_value="float type data")
                total_single_operate_times.extend(single_operate_times)

                # float和base对比
                self._check_two_batch_transform_same(base_batch_transform, float_batch_transform, check_col=col,
                                                     check_type="int trans float", check_cur_value="__int__",
                                                     check_pre_value="__float__")
                print("column: [{}] check int trans float value complete, total single transform speed:[{}]ms/it"
                      .format(col, np.round(np.mean(total_single_operate_times), 2)))

    def auto_test(self, x, sample=100):
        check_transform_function_describe = """
###################################################################
 1.一致性测试和性能测试:check_transform_function                      
###################################################################"""
        print(check_transform_function_describe)
        self.check_transform_function(x, sample=sample)
        check_null_value_describe = """
#########################################################################################
 2.空值测试:check_null_value                                                           
 null_values=[None, np.nan, "null", "NULL", "nan", "NaN", "", "none", "None", " "](默认)
########################################################################################"""
        print(check_null_value_describe)
        self.check_null_value(x, sample=sample)
        check_extreme_value_describe = """
############################################################################################################
 3.极端值测试:check_extreme_value                                                                            
 number_extreme_values = [np.inf, 0.0, -1, 1, -1e-7, 1e-7, np.finfo(np.float64).min, np.finfo(np.float64).max](默认)
 category_extreme_values = ["", "null", None, "1.0", "0.0", "-1.0", "-1", "NaN", "None"](默认)                            
###########################################################################################################"""
        print(check_extreme_value_describe)
        self.check_extreme_value(x, sample=sample)
        check_inverse_dtype_describe = """
###############################################################################################################
 4.数据类型反转测试:check_inverse_dtype                                                                          
 category_inverse_values = [0.0, -1, 1, -1e-7, 1e-7, np.finfo(np.float64).min, np.finfo(np.float64).max](默认)
 number_inverse_values = ["", "null", None, "1.0", "0.0", "-1.0", "-1"](默认)                                
#############################################################################################################"""
        print(check_inverse_dtype_describe)
        self.check_inverse_dtype(x, sample=sample)

        check_int_trans_float_describe = """
############################################
 5.int数据转float测试:check_int_trans_float                                                                                            
############################################"""
        print(check_int_trans_float_describe)
        self.check_int_trans_float(x, sample=sample)
