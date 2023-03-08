import easymlops.table.ensemble
from easymlops.table.core import TablePipeObjectBase, dict_type
import copy
from tqdm import tqdm
import pickle


class TablePipeLine(TablePipeObjectBase):
    def __init__(self, run_to_layer=None, **kwargs):
        super().__init__(**kwargs)
        self.models = []
        self.run_to_layer = run_to_layer

    def set_dependency(self, model):
        """
        设置依赖关系
        """
        # 添加前后依赖
        if len(self.models) > 0:
            model.set_parent_pipe(self.models[-1])
        else:
            model.set_parent_pipe(self.get_parent_pipe())
            return
        # 如果当前添加的model是pipeline或者Parallel,需要递归设置
        need_update_parent_pipes = []
        # 如果当前model也是pipeline
        if issubclass(model.__class__, TablePipeLine) and len(model.models) > 0:
            need_update_parent_pipes.append(model.models[0])
        # 如果当前model是Parallel
        if issubclass(model.__class__, easymlops.table.ensemble.Parallel) and len(model.pipe_objects) > 0:
            need_update_parent_pipes.extend(model.pipe_objects)
        while len(need_update_parent_pipes) > 0:
            pipe_ = need_update_parent_pipes.pop(0)
            pipe_.set_parent_pipe(self.models[-1])
            if issubclass(pipe_.__class__, TablePipeLine) and len(pipe_.models) > 0:
                need_update_parent_pipes.append(pipe_.models[0])
            # 如果当前model是Parallel
            if issubclass(pipe_.__class__, easymlops.table.ensemble.Parallel) and len(pipe_.pipe_objects) > 0:
                need_update_parent_pipes.extend(pipe_.pipe_objects)

    def pipe(self, model):
        self.set_dependency(model)
        self.models.append(model)
        return self

    def __getitem__(self, index):
        # 切片方式
        if isinstance(index, slice):
            target_models = self.models[index]
            new_pipe = copy.deepcopy(self)
            new_pipe.models = []
            for model in target_models:
                new_pipe.pipe(model)
            return new_pipe
        # 读取指定层方式
        elif type(index) == int or type(index) == str or type(index) == list or type(index) == tuple:
            return self._match_index_pipe(match_index=index)
        else:
            raise Exception("{} indices should be int,list,tuple,str or slice".format(self.name))

    def fit(self, x, show_process=False, **kwargs):
        # 注意:最后一层无需transform
        x_ = copy.deepcopy(x)
        run_idx = range(len(self.models))
        if show_process:
            run_idx = tqdm(run_idx)
        for idx in run_idx:
            model = self.models[idx]
            if show_process:
                print(model.name)
            if idx == len(self.models) - 1:
                model.fit(x_, **kwargs)
                # pipeline的最后一个pipe也要做transform，不过只处理一条数据，为的是调用transform过程中需要保留的一些参数
                model.transform(copy.deepcopy(x_[:1]), **kwargs)
            else:
                x_ = model.fit(x_, **kwargs).transform(x_, **kwargs)
        return self

    def _match_index_pipe(self, match_index) -> TablePipeObjectBase:
        if type(match_index) == int:
            return self.models[match_index]
        if type(match_index) == tuple:
            match_index = list(match_index)
        if type(match_index) == list:
            # 逐层获取
            pipes_ = []
            pipes_.extend(self.models)
            current_pipe = None
            while len(match_index) > 0:
                index = match_index.pop(0)
                current_pipe = pipes_[index]
                pipes_ = []
                if issubclass(current_pipe.__class__, TablePipeLine):
                    pipes_.extend(current_pipe.models)
                if issubclass(current_pipe.__class__, easymlops.table.ensemble.Parallel):
                    pipes_.extend(current_pipe.pipe_objects)
            return current_pipe
        if type(match_index) == str:
            pipes_ = []
            pipes_.extend(self.models)
            while len(pipes_) > 0:
                ipipe = pipes_.pop(0)
                if ipipe.name == match_index:
                    return ipipe
                if issubclass(ipipe.__class__, easymlops.table.ensemble.Parallel):
                    pipes_.extend(ipipe.pipe_objects)
                if issubclass(ipipe.__class__, TablePipeLine):
                    pipes_.extend(ipipe.models)
        raise Exception(f"can't match the index:{match_index}")

    def transform(self, x, show_process=False, run_to_layer=None, **kwargs):
        run_to_layer = self.run_to_layer if run_to_layer is None else run_to_layer
        x_ = copy.deepcopy(x)
        # 处理run_to_layer
        if run_to_layer is not None:
            pipe_ = self._match_index_pipe(run_to_layer)
            run_models = pipe_.get_all_parent_pipes()
            run_models.append(pipe_)
        else:
            run_models = self.models
        for model in tqdm(run_models) if show_process else run_models:
            if show_process:
                print(model.name)
            x_ = model.transform(x_, **kwargs)
        return x_

    def transform_single(self, x, show_process=False, run_to_layer=None, logger=None, prefix="step",
                         log_base_dict: dict_type = None, storage_base_dict: dict_type = None, **kwargs):
        run_to_layer = self.run_to_layer if run_to_layer is None else run_to_layer
        x_ = copy.deepcopy(x)

        # 处理run_to_layer
        if run_to_layer is not None:
            pipe_ = self._match_index_pipe(run_to_layer)
            run_models = pipe_.get_all_parent_pipes()
            run_models.append(pipe_)
            run_models = enumerate(run_models)
        else:
            run_models = enumerate(self.models)
        for current_layer_deep, model in tqdm(run_models) if show_process else run_models:
            if show_process:
                print(model.name)
            if issubclass(model.__class__, TablePipeLine):
                x_ = model.transform_single(x_, show_process=show_process,
                                            logger=logger,
                                            prefix="{}-{}".format(prefix, current_layer_deep),
                                            log_base_dict=log_base_dict, storage_base_dict=storage_base_dict,
                                            **kwargs)
            else:
                x_ = model.transform_single(x_, logger=logger,
                                            log_base_dict=log_base_dict,
                                            storage_base_dict=storage_base_dict, **kwargs)
                self._save_log(logger=logger, log_base_dict=log_base_dict,
                               step="{}-{}".format(prefix, current_layer_deep), transform=x_, pipe_name=model.name)
        return x_

    @staticmethod
    def _save_log(logger, log_base_dict, step, transform, pipe_name):
        """
        :param logger: 主要提供,logger.info方法，最好异步，不阻塞主程序
        :param log_base_dict: 续保保存基本信息，比如涉及到该条数据的id信息等
        :param step: 所在pipeline的第几步
        :param transform: 该步的输出
        :param pipe_name:pipe模块名称
        :return:
        """
        if log_base_dict is None:
            log_base_dict = dict()
        if logger is not None:
            log_info = {"step": step, "pipe_name": pipe_name, "transform": copy.deepcopy(transform)}
            log_info.update(log_base_dict)
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

    def auto_test(self, x, sample=100):
        from easymlops.table.callback import check_transform_function_pipeline, check_null_value, check_extreme_value, \
            check_inverse_dtype, check_int_trans_float
        check_transform_function_describe = """
###################################################################
 1.一致性测试和性能测试:check_transform_function                      
###################################################################"""
        print(check_transform_function_describe)
        self.callback(check_transform_function_pipeline, x, sample=sample)
        check_null_value_describe = """
#########################################################################################
 2.空值测试:check_null_value                                                           
 null_values=[None, np.nan, "null", "NULL", "nan", "NaN", "", "none", "None", " "](默认)
########################################################################################"""
        print(check_null_value_describe)
        self.callback(check_null_value, x, sample=sample)
        check_extreme_value_describe = """
############################################################################################################
 3.极端值测试:check_extreme_value                                                                            
 number_extreme_values = [np.inf, 0.0, -1, 1, -1e-7, 1e-7, np.finfo(np.float64).min, np.finfo(np.float64).max](默认)
 category_extreme_values = ["", "null", None, "1.0", "0.0", "-1.0", "-1", "NaN", "None"](默认)                            
###########################################################################################################"""
        print(check_extreme_value_describe)
        self.callback(check_extreme_value, x, sample=sample)
        check_inverse_dtype_describe = """
###############################################################################################################
 4.数据类型反转测试:check_inverse_dtype                                                                          
 category_inverse_values = [0.0, -1, 1, -1e-7, 1e-7, np.finfo(np.float64).min, np.finfo(np.float64).max](默认)
 number_inverse_values = ["", "null", None, "1.0", "0.0", "-1.0", "-1"](默认)                                
#############################################################################################################"""
        print(check_inverse_dtype_describe)
        self.callback(check_inverse_dtype, x, sample=sample)

        check_int_trans_float_describe = """
############################################
 5.int数据转float测试:check_int_trans_float                                                                                            
############################################"""
        print(check_int_trans_float_describe)
        self.callback(check_int_trans_float, x, sample=sample)
