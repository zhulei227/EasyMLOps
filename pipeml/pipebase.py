from .base import *
import copy
from tqdm import tqdm
import pickle


class Pipe(PipeObject):
    def __init__(self):
        PipeObject.__init__(self)
        self.models = []

    def pipe(self, model):
        self.models.append(model)
        return self

    def fit(self, x, show_process=False):
        x_ = copy.copy(x)
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
        x_ = copy.copy(x)
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
        x_ = copy.copy(x)
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
            log_info = copy.copy(log_base_dict)
            log_info.update({"step": step, "info": copy.copy(info)})
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

    def auto_check_transform(self, x):
        x_ = copy.copy(x)
        for model in self.models:
            x_ = model.auto_check_transform(x_)