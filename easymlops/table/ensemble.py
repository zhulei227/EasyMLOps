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


class StackingClassification(TablePipeObjectBase):
    """
    分类 OOF 堆叠（v0.2.7 新增，源自 Kaggle: model stacking with OOF meta features）。

    流程:
        1. StratifiedKFold 分折: 每折用"排除该折"训练的各 base 模型预测该折概率,
           拼成 OOF 概率矩阵(行数=训练集, 列=类别);
        2. 用 OOF 概率矩阵训练 meta 模型;
        3. 各 base 模型在全量数据重训(最终模型), transform 时
           base 概率 → meta 模型 → 输出最终类别概率。

    Args:
        base_pipes: 已配置好(构造时绑定 y)的分类模型 pipe 列表(基学习器)
        meta_pipe: 元模型 pipe(如 LogisticRegressionClassification, 构造时绑定 y)
        cv: 折数(>=2)
        random_state: 分折随机种子
        **kwargs: 其他父类参数

    Example:
        >>> from easymlops.table.ensemble import StackingClassification
        >>> from easymlops.table.classification import LGBMClassification, LogisticRegressionClassification
        >>> stack = StackingClassification(
        ...     base_pipes=[LGBMClassification(y=y_train, use_faster_predictor=False),
        ...                 LogisticRegressionClassification(y=y_train)],
        ...     meta_pipe=LogisticRegressionClassification(y=y_train), cv=5)
        >>> stack.fit(train_df)
        >>> probs = stack.transform(test_df)
    """

    def __init__(self, base_pipes=None, meta_pipe=None, cv=5, random_state=42, **kwargs):
        super().__init__(**kwargs)
        self.base_pipes = list(base_pipes) if base_pipes else []
        self.meta_pipe = meta_pipe
        self.cv = max(2, int(cv))
        self.random_state = random_state
        self.final_bases_ = []
        self.meta_model_ = None
        self.class_order_ = []

    def _check(self):
        if not self.base_pipes or self.meta_pipe is None:
            raise Exception("StackingClassification requires base_pipes and meta_pipe")
        for base in self.base_pipes:
            if getattr(base, "num_class", None) is None:
                raise Exception("base_pipes must be classification model pipes with y bound")
        if getattr(self.meta_pipe, "num_class", None) is None:
            raise Exception("meta_pipe must be a classification model pipe with y bound")

    def udf_fit(self, s: dataframe_type, **kwargs):
        self._check()
        from sklearn.model_selection import StratifiedKFold
        y_ids = np.asarray(self.base_pipes[0].y, dtype=np.int64)
        class_order = [self.base_pipes[0].id2label[i] for i in range(self.base_pipes[0].num_class)]
        self.class_order_ = class_order
        splitter = StratifiedKFold(n_splits=self.cv, shuffle=True,
                                   random_state=self.random_state)
        oof = np.zeros((len(s), len(class_order)), dtype=np.float64)
        for train_idx, valid_idx in splitter.split(s, y_ids):
            s_tr = s.iloc[train_idx].reset_index(drop=True)
            s_va = s.iloc[valid_idx].reset_index(drop=True)
            y_tr = pd.Series(y_ids[train_idx]).reset_index(drop=True)
            for base in self.base_pipes:
                fold_model = copy.deepcopy(base)
                fold_model.y = y_tr
                fold_model.fit(s_tr)
                probs = fold_model.transform(s_va)
                probs = probs[class_order]   # 按统一类别顺序对齐
                oof[valid_idx] += probs.to_numpy(dtype=np.float64)
        oof /= max(len(self.base_pipes), 1)

        # meta 模型: 用 OOF 概率矩阵训练
        meta_model = copy.deepcopy(self.meta_pipe)
        meta_model.y = pd.Series(y_ids)
        meta_model.fit(pd.DataFrame(oof, columns=class_order))

        # 最终 base 模型: 全量重训
        self.final_bases_ = []
        for base in self.base_pipes:
            final_model = copy.deepcopy(base)
            final_model.fit(s)
            self.final_bases_.append(final_model)
        self.meta_model_ = meta_model
        return self

    def _base_probs(self, s: dataframe_type) -> dataframe_type:
        frames = [base.transform(s) for base in self.final_bases_]
        combined = frames[0].copy()
        for frame in frames[1:]:
            combined = combined + frame
        return combined / float(len(frames))

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        if self.meta_model_ is None:
            raise Exception("StackingClassification is not fitted, call fit before transform")
        probs = self._base_probs(s)[self.class_order_]
        return self.meta_model_.transform(probs)

    def udf_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        if self.meta_model_ is None:
            raise Exception("StackingClassification is not fitted, call fit before transform")
        combined = {}
        for base in self.final_bases_:
            for key, value in base.transform_single(copy.deepcopy(s)).items():
                combined[key] = combined.get(key, 0.0) + float(value)
        averaged = {k: v / len(self.final_bases_) for k, v in combined.items()}
        return self.meta_model_.transform_single(averaged)

    def udf_get_params(self) -> dict_type:
        return {"cv": self.cv, "random_state": self.random_state,
                "class_order_": self.class_order_,
                "base_classes": [b.__class__ for b in self.base_pipes],
                "base_params": [b.get_params() for b in self.base_pipes],
                "meta_class": self.meta_pipe.__class__ if self.meta_pipe else None,
                "meta_params": self.meta_pipe.get_params() if self.meta_pipe else None,
                "final_bases_params": [b.get_params() for b in self.final_bases_],
                "meta_model_params": self.meta_model_.get_params() if self.meta_model_ else None}

    def udf_set_params(self, params: dict_type):
        self.cv = params["cv"]
        self.random_state = params["random_state"]
        self.class_order_ = params["class_order_"]
        self.base_pipes = []
        for cls, bp in zip(params["base_classes"], params["base_params"]):
            m = cls()
            m.set_params(bp)
            self.base_pipes.append(m)
        self.meta_pipe = None
        if params["meta_class"] is not None:
            self.meta_pipe = params["meta_class"]()
            self.meta_pipe.set_params(params["meta_params"])
        self.final_bases_ = []
        for cls, bp in zip(params["base_classes"], params["final_bases_params"]):
            m = cls()
            m.set_params(bp)
            self.final_bases_.append(m)
        self.meta_model_ = None
        if params["meta_model_params"] is not None and params["meta_class"] is not None:
            self.meta_model_ = params["meta_class"]()
            self.meta_model_.set_params(params["meta_model_params"])

from easymlops.table.classification import ClassificationBase
from easymlops.table.strategy.hill_climbing.climber import ClimberCV
from easymlops.table.utils import EvalFunction


def _binary_auc_metric(y_true, probs):
    """模块级 AUC(可 pickle, 供爬山法多进程使用)"""
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y_true, probs)


class AugmentationPipe(TablePipeObjectBase):
    """
    训练集数据增强包装器（v0.2.10 新增，源自 Kaggle: mixup/SMOTE augmentation）：
    按(method)对训练数据增强后训练 base_pipe，transform/transform_single 委托。

    方法:
        - "smote":       分类；少数类样本与其 k 近邻少数类样本数值列插值合成新样本
                         （类别列/标签取自种子样本），至 sampling_ratio×少数类数量
        - "oversample":  分类；少数类随机过采样至 sampling_ratio×少数类数量
        - "mixup":       回归；随机样本对数值列凸组合(lam~Beta(alpha,alpha))，标签同比例混合
        - "gaussian":    回归；数值列加 N(0, noise_std×列标准差) 噪声，标签不变

    注意: GBDT 类模型对插值增强基本无感，增强主要使 NN/线性模型受益。

    Args:
        base_pipe: 已配置好的模型 pipe 模板（须在构造时绑定 y）
        method: "smote" / "oversample" / "mixup" / "gaussian"
        sampling_ratio: smote/oversample 的增强倍率（1.0=平衡至多数类）
        k_neighbors: smote 近邻数
        alpha: mixup 的 Beta 分布参数
        n_augment: mixup 合成样本数
        noise_std: gaussian 噪声强度（相对列标准差的倍数）
        random_state: 随机种子
        **kwargs: 其他父类参数

    Example:
        >>> from easymlops.table.ensemble import AugmentationPipe
        >>> model = AugmentationPipe(
        ...     base_pipe=TabularMLPClassification(y=y_train, epochs=10),
        ...     method="smote", sampling_ratio=1.0)
        >>> model.fit(train_df)
    """

    def __init__(self, base_pipe=None, method="smote", sampling_ratio=1.0, k_neighbors=5,
                 alpha=0.2, n_augment=None, noise_std=0.1, random_state=42, **kwargs):
        super().__init__(**kwargs)
        self.base_pipe = base_pipe
        self.method = method
        if method not in ("smote", "oversample", "mixup", "gaussian"):
            raise Exception("method must be smote/oversample/mixup/gaussian")
        self.sampling_ratio = float(sampling_ratio)
        self.k_neighbors = max(1, int(k_neighbors))
        self.alpha = float(alpha)
        self.n_augment = n_augment
        self.noise_std = float(noise_std)
        self.random_state = int(random_state)
        self.model_ = None


    def _is_classification(self):
        return getattr(self.base_pipe, "num_class", None) is not None

    def _numeric_cols(self, s: dataframe_type):
        return [c for c in s.columns if np.issubdtype(s[c].dtype, np.number)]

    def _augment_smote(self, s: dataframe_type, y: series_type):
        from sklearn.neighbors import NearestNeighbors
        num_cols = self._numeric_cols(s)
        y_values = y.reset_index(drop=True)
        counts = y_values.value_counts()
        minority = counts.idxmin()
        majority_count = counts.max()
        target_count = int(counts[minority] * self.sampling_ratio) - counts[minority]
        if target_count <= 0:
            return s, y
        minority_idx = y_values[y_values == minority].index.tolist()
        minority_rows = s.loc[minority_idx].reset_index(drop=True)
        x_num = minority_rows[num_cols].fillna(0).to_numpy(dtype=float)
        k = min(self.k_neighbors, len(x_num) - 1)
        if k < 1:
            return s, y
        nn = NearestNeighbors(n_neighbors=k + 1).fit(x_num)
        rng = np.random.RandomState(self.random_state)
        synth_rows = []
        for _ in range(target_count):
            i = rng.randint(0, len(x_num))
            neigh = nn.kneighbors(x_num[i:i + 1], return_distance=False)[0][1:]
            j = neigh[rng.randint(0, len(neigh))]
            lam = rng.rand()
            row = minority_rows.iloc[i].copy()
            for c in num_cols:
                v_i, v_j = x_num[i][num_cols.index(c)], x_num[j][num_cols.index(c)]
                row[c] = v_i + lam * (v_j - v_i)
            synth_rows.append(row)
        aug_s = pd.concat([s.reset_index(drop=True),
                           pd.DataFrame(synth_rows).reindex(columns=s.columns)],
                          axis=0, ignore_index=True)
        aug_y = pd.concat([y_values, pd.Series([minority] * len(synth_rows))],
                          axis=0, ignore_index=True)
        return aug_s, aug_y

    def _augment_oversample(self, s: dataframe_type, y: series_type):
        y_values = y.reset_index(drop=True)
        counts = y_values.value_counts()
        minority = counts.idxmin()
        target_count = int(counts[minority] * self.sampling_ratio) - counts[minority]
        if target_count <= 0:
            return s, y
        rng = np.random.RandomState(self.random_state)
        minority_idx = y_values[y_values == minority].index.tolist()
        picked = rng.choice(minority_idx, size=target_count, replace=True)
        aug_s = pd.concat([s.reset_index(drop=True), s.iloc[picked].reset_index(drop=True)],
                          axis=0, ignore_index=True)
        aug_y = pd.concat([y_values, y_values.iloc[picked].reset_index(drop=True)],
                          axis=0, ignore_index=True)
        return aug_s, aug_y

    def _augment_mixup(self, s: dataframe_type, y: series_type):
        num_cols = self._numeric_cols(s)
        n_aug = self.n_augment if self.n_augment is not None else len(s)
        rng = np.random.RandomState(self.random_state)
        idx_a = rng.randint(0, len(s), n_aug)
        idx_b = rng.randint(0, len(s), n_aug)
        lams = rng.beta(self.alpha, self.alpha, n_aug)
        a_num = s.iloc[idx_a][num_cols].fillna(0).to_numpy(dtype=float)
        b_num = s.iloc[idx_b][num_cols].fillna(0).to_numpy(dtype=float)
        mix_num = lams.reshape(-1, 1) * a_num + (1 - lams.reshape(-1, 1)) * b_num
        mix_df = s.iloc[idx_a].reset_index(drop=True).copy()
        for k, c in enumerate(num_cols):
            mix_df[c] = mix_num[:, k]
        y_values = y.reset_index(drop=True)
        mix_y = lams * y_values.iloc[idx_a].to_numpy(dtype=float) + \
            (1 - lams) * y_values.iloc[idx_b].to_numpy(dtype=float)
        aug_s = pd.concat([s.reset_index(drop=True), mix_df], axis=0, ignore_index=True)
        aug_y = pd.concat([y_values, pd.Series(mix_y)], axis=0, ignore_index=True)
        return aug_s, aug_y

    def _augment_gaussian(self, s: dataframe_type, y: series_type):
        num_cols = self._numeric_cols(s)
        rng = np.random.RandomState(self.random_state)
        noise_df = s.copy()
        for c in num_cols:
            std = float(s[c].std() or 0.0)
            noise_df[c] = s[c] + rng.normal(0, self.noise_std * std, len(s))
        return noise_df, y

    def udf_fit(self, s: dataframe_type, **kwargs):
        if self.base_pipe is None:
            raise Exception("AugmentationPipe requires base_pipe (with y bound at construction)")
        np.random.seed(self.random_state)
        y = self.base_pipe.y.reset_index(drop=True)
        s_work = s.reset_index(drop=True)
        if self.method == "smote":
            if not self._is_classification():
                raise Exception("method='smote' requires a classification base_pipe")
            aug_s, aug_y = self._augment_smote(s_work, y)
        elif self.method == "oversample":
            if not self._is_classification():
                raise Exception("method='oversample' requires a classification base_pipe")
            aug_s, aug_y = self._augment_oversample(s_work, y)
        elif self.method == "mixup":
            if self._is_classification():
                raise Exception("method='mixup' requires a regression base_pipe "
                                "(mixed labels are not valid class labels)")
            aug_s, aug_y = self._augment_mixup(s_work, y)
        else:
            aug_s, aug_y = self._augment_gaussian(s_work, y)

        self.model_ = copy.deepcopy(self.base_pipe)
        self.model_.y = aug_y
        self.model_.fit(aug_s, **kwargs)
        return self

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        if self.model_ is None:
            raise Exception("AugmentationPipe is not fitted, call fit before transform")
        return self.model_.transform(s, **kwargs)

    def udf_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        if self.model_ is None:
            raise Exception("AugmentationPipe is not fitted, call fit before transform")
        return self.model_.transform_single(s, **kwargs)

    def udf_get_params(self) -> dict_type:
        return {"base_class": self.base_pipe.__class__ if self.base_pipe is not None else None,
                "base_params": self.base_pipe.get_params() if self.base_pipe is not None else None,
                "method": self.method, "sampling_ratio": self.sampling_ratio,
                "k_neighbors": self.k_neighbors, "alpha": self.alpha,
                "n_augment": self.n_augment, "noise_std": self.noise_std,
                "random_state": self.random_state,
                "model_params": self.model_.get_params() if self.model_ is not None else None}

    def udf_set_params(self, params: dict_type):
        self.method = params["method"]
        self.sampling_ratio = params["sampling_ratio"]
        self.k_neighbors = params["k_neighbors"]
        self.alpha = params["alpha"]
        self.n_augment = params["n_augment"]
        self.noise_std = params["noise_std"]
        self.random_state = params["random_state"]
        if params["base_class"] is not None:
            self.base_pipe = params["base_class"]()
            self.base_pipe.set_params(params["base_params"])
        else:
            self.base_pipe = None
        if params["model_params"] is not None and self.base_pipe is not None:
            self.model_ = params["base_class"]()
            self.model_.set_params(params["model_params"])
        else:
            self.model_ = None



class HillClimbingStackingClassification(ClassificationBase):
    """
    分类版爬山法堆叠（v0.2.10 新增，源自 Kaggle: hill climbing ensemble）。

    输入为各基模型输出的概率列（来自 Parallel(模型组)），对每个类别独立运行
    ClimberCV 爬山（StratifiedKFold 防泄漏、AUC 最大化默认），逐列贪心加权混合；
    输出为各类别混合后的概率（跨类归一化），与 ClassificationBase 输出契约一致。

    Args:
        y: 目标变量（构造时绑定）
        eval_function: 逐类二值评估函数（默认 "auc"，可传 callable(y_bin, probs)）
        eval_function_direct: "maximize"/"minimize"
        allow_negative_weights: 是否允许负权重
        precision: 权重搜索步长
        n_splits: StratifiedKFold 折数
        random_state / shuffle: 分折参数
        **kwargs: 其他父类参数

    Example:
        >>> from easymlops.table.ensemble import Parallel, HillClimbingStackingClassification
        >>> from easymlops.table.classification import LGBMClassification, LogisticRegressionClassification
        >>> pipeline.pipe(Parallel(pipe_objects=[
        ...     LGBMClassification(y=y_train, prefix="lgbm", use_faster_predictor=False),
        ...     LogisticRegressionClassification(y=y_train, prefix="lr")]))
        >>> pipeline.pipe(HillClimbingStackingClassification(y=y_train, eval_function="auc"))
    """

    def __init__(self, y: series_type = None, eval_function="auc", eval_function_direct="maximize",
                 cols="all", allow_negative_weights=True, precision=0.001,
                 score_decimal_places=6, n_splits=5, random_state=42, shuffle=True,
                 skip_check_transform_type=True, drop_input_data=True,
                 support_sparse_input=False, **kwargs):
        super().__init__(y=y, cols=cols, skip_check_transform_type=skip_check_transform_type,
                         drop_input_data=drop_input_data,
                         support_sparse_input=support_sparse_input, **kwargs)
        if type(eval_function) == str and \
                eval_function.lower() not in ("auc", "roc_auc"):
            eval_function = EvalFunction.get(eval_function)
        self.eval_function = eval_function
        self.eval_function_direct = eval_function_direct
        self.allow_negative_weights = allow_negative_weights
        self.precision = precision
        self.score_decimal_places = score_decimal_places
        self.n_splits = n_splits
        self.random_state = random_state
        self.shuffle = shuffle
        self.climbers_ = {}

    def udf_fit(self, s: dataframe_type, **kwargs):
        from sklearn.model_selection import StratifiedKFold
        self.climbers_ = {}
        if isinstance(self.eval_function, str) and \
                self.eval_function.lower() in ("auc", "roc_auc"):
            eval_fn = _binary_auc_metric
        else:
            eval_fn = self.eval_function
        y_ids = self.y
        for class_id in range(self.num_class):
            y_bin = (np.asarray(y_ids, dtype=np.int64) == class_id).astype(float)
            climber = ClimberCV(
                objective=self.eval_function_direct,
                eval_metric=eval_fn,
                allow_negative_weights=self.allow_negative_weights,
                precision=self.precision,
                score_decimal_places=self.score_decimal_places,
                verbose=False,
                n_jobs=1,
                cv=StratifiedKFold(n_splits=self.n_splits, random_state=self.random_state,
                                   shuffle=self.shuffle)
            )
            climber.fit(s.copy(), pd.Series(y_bin))
            self.climbers_[class_id] = climber
        return self

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        if not self.climbers_:
            raise Exception("HillClimbingStackingClassification is not fitted")
        probs = {}
        for class_id, climber in self.climbers_.items():
            probs[class_id] = climber.predict(s)
        prob_df = pd.DataFrame(probs, index=s.index)
        # 跨类归一化(负权重可能出现负值, clip 到 0)
        prob_df = prob_df.clip(lower=0.0)
        total = prob_df.sum(axis=1).replace(0.0, 1.0)
        prob_df = prob_df.div(total, axis=0)
        prob_df.columns = [self.id2label[i] for i in range(self.num_class)]
        return prob_df

    def udf_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        result_df = self.transform(pd.DataFrame([s]))
        return {col: float(row) for col, row in result_df.iloc[0].items()}

    def udf_get_params(self) -> dict_type:
        return {"eval_function": self.eval_function,
                "eval_function_direct": self.eval_function_direct,
                "allow_negative_weights": self.allow_negative_weights,
                "precision": self.precision, "n_splits": self.n_splits,
                "random_state": self.random_state, "shuffle": self.shuffle,
                "climbers": self.climbers_,
                "y": self.y}

    def udf_set_params(self, params: dict):
        self.eval_function = params["eval_function"]
        self.eval_function_direct = params["eval_function_direct"]
        self.allow_negative_weights = params["allow_negative_weights"]
        self.precision = params["precision"]
        self.n_splits = params["n_splits"]
        self.random_state = params["random_state"]
        self.shuffle = params["shuffle"]
        self.climbers_ = params["climbers"]
        self.y = params["y"]
