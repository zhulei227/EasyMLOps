import copy

import numpy as np
import pandas as pd

from easymlops.table.core import *
from easymlops.table.utils import PandasUtils


class Parallel(TablePipeObjectBase):
    """
    并行模块。
    
    接受相同的数据，并行执行多个pipe object，然后将结果按列合并输出。
    列名相同的后续结果会覆盖前面的结果。
    
    Example:
        >>> parallel = Parallel(pipe_objects=[pipe1, pipe2, pipe3])
    """
    
    def __init__(self, pipe_objects=None, drop_input_data=True, skip_check_transform_type=True, **kwargs):
        """
        初始化并行模块。
        
        Args:
            pipe_objects: 要并行执行的pipe对象列表
            drop_input_data: 是否删除输入数据
            skip_check_transform_type: 跳过类型检测
            **kwargs: 其他父类参数
        """
        super().__init__(skip_check_transform_type=skip_check_transform_type, **kwargs)
        self.pipe_objects = pipe_objects
        for obj in self.pipe_objects:
            obj.set_parent_pipe(self.get_parent_pipe())
        self.drop_input_data = drop_input_data

    def __getitem__(self, target_pipe_model):
        """通过索引或名称获取子pipe对象。"""
        return self._match_index_pipe(match_index=target_pipe_model)

    def _match_index_pipe(self, match_index) -> TablePipeObjectBase:
        """
        根据索引匹配pipe对象。
        
        Args:
            match_index: 索引，可以是int、tuple、list或str
            
        Returns:
            匹配的pipe对象
        """
        if type(match_index) == int:
            return self.pipe_objects[match_index]
        if type(match_index) == tuple:
            match_index = list(match_index)
        if type(match_index) == list:
            pipes_ = []
            pipes_.extend(self.pipe_objects)
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
            pipes_.extend(self.pipe_objects)
            while len(pipes_) > 0:
                ipipe = pipes_.pop(0)
                if ipipe.name == match_index:
                    return ipipe
                if issubclass(ipipe.__class__, easymlops.table.ensemble.Parallel):
                    pipes_.extend(ipipe.pipe_objects)
                if issubclass(ipipe.__class__, TablePipeLine):
                    pipes_.extend(ipipe.models)
        raise Exception(f"can't match the index:{match_index}")

    def udf_fit(self, s: dataframe_type, **kwargs):
        """并行训练所有子pipe对象。"""
        for obj in self.pipe_objects:
            s_ = copy.deepcopy(s)
            obj.fit(s_, **kwargs)
        return self

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        """并行转换数据并合并结果。"""
        dfs = []
        for index in range(len(self.pipe_objects)):
            s_ = copy.deepcopy(s)
            df = self.pipe_objects[index].transform(s_, **kwargs)
            df.index = s.index
            dfs.append(df)
        return PandasUtils.concat_duplicate_columns(dfs)

    def udf_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        """单条数据并行转换。"""
        result = dict()
        for index in range(len(self.pipe_objects)):
            s_ = copy.deepcopy(s)
            result.update(self.pipe_objects[index].transform_single(s_, **kwargs))
        return result

    def transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        """批量数据转换。"""
        output = self.after_transform(self.udf_transform(self.before_transform(s, **kwargs), **kwargs), **kwargs)
        if self.drop_input_data:
            return output
        else:
            return PandasUtils.concat_duplicate_columns([s, output])

    def transform_single(self, s: dict_type, **kwargs) -> dict_type:
        """单条数据转换。"""
        output = self.after_transform_single(
            self.udf_transform_single(self.before_transform_single(s, **kwargs), **kwargs), **kwargs)
        if self.drop_input_data:
            return output
        else:
            s.update(output)
            return s

    def udf_get_params(self):
        """获取参数。"""
        return {"pipe_objects_params": [obj.get_params() for obj in self.pipe_objects],
                "drop_input_data": self.drop_input_data}

    def udf_set_params(self, params: dict_type):
        """设置参数。"""
        self.drop_input_data = params["drop_input_data"]
        pipe_objects_params = params["pipe_objects_params"]
        for i in range(len(pipe_objects_params)):
            pipe_params = pipe_objects_params[i]
            obj = self.pipe_objects[i]
            obj.set_params(pipe_params)


class SeedAveraging(TablePipeObjectBase):
    """
    同构模型多种子训练并平均预测（Kaggle 标配 bagging，v0.2.5 新增）。

    对 base_pipe 深拷贝 n_seeds 份，向 native_init_params 注入不同种子后分别
    训练，transform 时对各组预测取算术平均（分类=平均概率，回归=平均预测值）。

    Args:
        base_pipe: 已配置好的模型 pipe 模板（须在构造时绑定 y），作为每组训练副本
        n_seeds: 种子数量
        seed_start: 起始种子（第 i 个模型种子 = seed_start + i）
        seed_param: 注入 native_init_params 的种子参数名
                    （lightgbm 用 "seed"，sklearn 模型用 "random_state"）
        **kwargs: 其他父类参数

    Example:
        >>> model = SeedAveraging(
        ...     base_pipe=LGBMClassification(y=y_train, native_init_params={"learning_rate": 0.05}),
        ...     n_seeds=5)
        >>> model.fit(train_df)
        >>> pred = model.transform(test_df)   # 平均后的概率列（列名=类别）
    """

    def __init__(self, base_pipe=None, n_seeds=5, seed_start=42, seed_param="seed", **kwargs):
        super().__init__(**kwargs)
        self.base_pipe = base_pipe
        self.n_seeds = int(n_seeds)
        self.seed_start = int(seed_start)
        self.seed_param = seed_param
        self.seed_pipes_ = []

    def udf_fit(self, s, **kwargs):
        if self.base_pipe is None:
            raise Exception("SeedAveraging requires base_pipe (with y bound at construction)")
        self.seed_pipes_ = []
        for i in range(self.n_seeds):
            m = copy.deepcopy(self.base_pipe)
            if hasattr(m, "native_init_params"):
                m.native_init_params[self.seed_param] = self.seed_start + i
            m.fit(s, **kwargs)
            self.seed_pipes_.append(m)
        return self

    @staticmethod
    def _average_frames(frames):
        result = frames[0].copy()
        for frame in frames[1:]:
            result = result + frame
        return result / float(len(frames))

    def udf_transform(self, s, **kwargs):
        if not self.seed_pipes_:
            raise Exception("SeedAveraging is not fitted, call fit before transform")
        outs = [m.transform(s, **kwargs) for m in self.seed_pipes_]
        return self._average_frames(outs)

    def udf_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        if not self.seed_pipes_:
            raise Exception("SeedAveraging is not fitted, call fit before transform")
        outs = [m.transform_single(copy.deepcopy(s), **kwargs) for m in self.seed_pipes_]
        return {k: float(np.mean([o[k] for o in outs])) for k in outs[0].keys()}

    def udf_get_params(self) -> dict_type:
        return {"base_class": self.base_pipe.__class__ if self.base_pipe is not None else None,
                "base_params": self.base_pipe.get_params() if self.base_pipe is not None else None,
                "n_seeds": self.n_seeds, "seed_start": self.seed_start, "seed_param": self.seed_param,
                "seed_pipes_params": [m.get_params() for m in self.seed_pipes_]}

    def udf_set_params(self, params: dict_type):
        self.n_seeds = params["n_seeds"]
        self.seed_start = params["seed_start"]
        self.seed_param = params["seed_param"]
        if params["base_class"] is not None:
            self.base_pipe = params["base_class"]()
            self.base_pipe.set_params(params["base_params"])
        else:
            self.base_pipe = None
        self.seed_pipes_ = []
        for pipe_params in params["seed_pipes_params"]:
            m = params["base_class"]()
            m.set_params(pipe_params)
            self.seed_pipes_.append(m)


class PseudoLabelPipe(TablePipeObjectBase):
    """
    伪标签半监督（pseudo-labeling，v0.2.5 新增）。

    先在有标签数据上训练 base_pipe 副本，对无标签数据预测；分类任务保留
    置信度 >= confidence 的样本并打上伪标签，回归任务直接用预测值作为伪标签；
    将伪标签样本拼回训练集重新训练（可迭代 n_rounds 轮）。transform 委托给
    最终模型。

    Args:
        base_pipe: 已配置好的模型 pipe 模板（须在构造时绑定 y）
        confidence: 分类伪标签的置信度阈值（max prob >= confidence 才保留）
        n_rounds: 伪标签迭代轮数
        unlabeled: 无标签数据（也可在 fit 时通过 unlabeled= 传入）
        **kwargs: 其他父类参数

    Example:
        >>> model = PseudoLabelPipe(
        ...     base_pipe=LGBMClassification(y=y_train), confidence=0.9, n_rounds=1)
        >>> model.fit(train_df, unlabeled=test_df)   # test_df 无标签
        >>> pred = model.transform(test_df)
    """

    def __init__(self, base_pipe=None, confidence=0.9, n_rounds=1, unlabeled=None, **kwargs):
        super().__init__(**kwargs)
        self.base_pipe = base_pipe
        self.confidence = float(confidence)
        self.n_rounds = int(n_rounds)
        self.unlabeled = unlabeled
        self.model_ = None

    def _is_classification(self):
        return getattr(self.base_pipe, "num_class", None) is not None

    def _new_model(self):
        m = copy.deepcopy(self.base_pipe)
        if hasattr(m, "weight"):
            m.weight = None   # 扩充后样本长度变化, 原权重失效
        return m

    def udf_fit(self, s, **kwargs):
        if self.base_pipe is None:
            raise Exception("PseudoLabelPipe requires base_pipe (with y bound at construction)")
        unlabeled = kwargs.pop("unlabeled", None)
        if unlabeled is None:
            unlabeled = self.unlabeled
        self.model_ = self._new_model()
        self.model_.fit(s, **kwargs)
        if unlabeled is None or len(unlabeled) == 0 or self.n_rounds < 1:
            return self
        cols = self.model_.cols
        feats = unlabeled[cols]
        for _ in range(self.n_rounds):
            preds = self.model_.transform(feats)
            if self._is_classification():
                conf = preds.max(axis=1)
                pseudo_labels = preds.idxmax(axis=1)
                mask = conf >= self.confidence
                if int(mask.sum()) == 0:
                    break
                pseudo_ids = pseudo_labels[mask].map(self.model_.label2id)
                y_ext = pd.concat([self.base_pipe.y.reset_index(drop=True),
                                   pseudo_ids.reset_index(drop=True)], axis=0,
                                  ignore_index=True)
            else:
                mask = pd.Series(True, index=preds.index)
                pred_col = preds.columns[0]
                y_ext = pd.concat([self.base_pipe.y.reset_index(drop=True),
                                   preds[mask][pred_col].reset_index(drop=True)], axis=0,
                                  ignore_index=True)
            ext_feats = feats.loc[mask].reset_index(drop=True)
            train_ext = pd.concat([s.reset_index(drop=True), ext_feats], axis=0)
            self.model_ = self._new_model()
            self.model_.y = y_ext
            self.model_.fit(train_ext, **kwargs)
        return self

    def udf_transform(self, s, **kwargs):
        if self.model_ is None:
            raise Exception("PseudoLabelPipe is not fitted, call fit before transform")
        return self.model_.transform(s, **kwargs)

    def udf_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        if self.model_ is None:
            raise Exception("PseudoLabelPipe is not fitted, call fit before transform")
        return self.model_.transform_single(s, **kwargs)

    def udf_get_params(self) -> dict_type:
        return {"base_class": self.base_pipe.__class__ if self.base_pipe is not None else None,
                "base_params": self.base_pipe.get_params() if self.base_pipe is not None else None,
                "base_y": self.base_pipe.y if self.base_pipe is not None else None,
                "confidence": self.confidence, "n_rounds": self.n_rounds,
                "model_params": self.model_.get_params() if self.model_ is not None else None}

    def udf_set_params(self, params: dict_type):
        self.confidence = params["confidence"]
        self.n_rounds = params["n_rounds"]
        if params["base_class"] is not None:
            self.base_pipe = params["base_class"]()
            self.base_pipe.set_params(params["base_params"])
            self.base_pipe.y = params["base_y"]
        else:
            self.base_pipe = None
        if params["model_params"] is not None:
            self.model_ = params["base_class"]()
            self.model_.set_params(params["model_params"])
        else:
            self.model_ = None
