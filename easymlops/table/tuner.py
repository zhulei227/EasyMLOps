# -*- coding: utf-8 -*-
"""
超参搜索 pipe（v0.2.5 新增，源自 Kaggle 优胜方案高频技术：贝叶斯超参优化）。

约定:
    - 惰性导入 optuna（未安装时给出明确报错）
    - 搜索空间作用于模型 pipe 的 native_init_params（如 LGBM 的 learning_rate/num_leaves）
    - CV 评估后用最优参数在全量数据重训，transform 委托最优模型
"""
import copy

import numpy as np
import pandas as pd

from easymlops.table.core import *

dataframe_type = pd.DataFrame
dict_type = dict


class OptunaTuner(TablePipeObjectBase):
    """
    Optuna 超参搜索 pipe（v0.2.5 新增）。

    对模型类(如 LGBMRegression/LGBMClassification)的 native_init_params 做
    贝叶斯超参搜索：每轮 trial 采样一组参数 → 构造模型 → CV 评估 → 以最优参数
    在全量数据重训。transform/transform_single 委托最优模型。

    Args:
        model_class: 模型 pipe 类（如 LGBMRegression；构造须支持 **kwargs）
        fixed_params: 构造模型的固定参数（须包含 y；建议 use_faster_predictor=False）
        search_space: dict[param_name -> callable(trial) -> value]，
                      作用于 native_init_params，
                      如 {"learning_rate": lambda t: t.suggest_float("learning_rate", 0.01, 0.2)}
        n_trials: 搜索轮数
        cv: CV 折数（>=2；分类用 StratifiedKFold）
        task: "regression"（metric=RMSE）/ "classification"（metric=AUC，仅支持 0/1 二分类）
        seed: 分折与采样随机种子
        **kwargs: 其他父类参数

    Example:
        >>> from easymlops.table.tuner import OptunaTuner
        >>> from easymlops.table.regression import LGBMRegression
        >>> tuner = OptunaTuner(
        ...     model_class=LGBMRegression,
        ...     fixed_params={"y": y_train, "use_faster_predictor": False},
        ...     search_space={
        ...         "learning_rate": lambda t: t.suggest_float("learning_rate", 0.01, 0.2),
        ...         "num_leaves": lambda t: t.suggest_int("num_leaves", 8, 64),
        ...     },
        ...     n_trials=20, cv=3, task="regression")
        >>> tuner.fit(train_df)
        >>> print(tuner.best_params_)
    """

    def __init__(self, model_class=None, fixed_params=None, search_space=None,
                 n_trials=20, cv=3, task="regression", seed=42, **kwargs):
        super().__init__(**kwargs)
        self.model_class = model_class
        self.fixed_params = fixed_params or {}
        self.search_space = search_space or {}
        self.n_trials = int(n_trials)
        self.cv = int(cv)
        self.task = task
        self.seed = seed
        self.model_ = None
        self.best_params_ = None
        self.best_score_ = None

    def _build_model(self, y_values, native_params):
        """用(折叠)标签与采样的 native 参数构造模型 pipe。"""
        if self.model_class is None:
            raise Exception("OptunaTuner requires model_class")
        fixed = copy.deepcopy(self.fixed_params)
        fixed["y"] = pd.Series(y_values).reset_index(drop=True)
        merged_native = copy.deepcopy(fixed.get("native_init_params") or {})
        merged_native.update(native_params)
        fixed["native_init_params"] = merged_native
        return self.model_class(**fixed)

    @staticmethod
    def _get_pred_values(pred: dataframe_type) -> np.ndarray:
        """从模型输出提取一维预测值（回归取首列；二分类取类别 1 的概率列）。"""
        for col in (1, "1"):
            if col in pred.columns:
                return pred[col].to_numpy(dtype=float)
        return pred.iloc[:, 0].to_numpy(dtype=float)

    def _cv_score(self, s: dataframe_type, native_params) -> float:
        """CV 评估，返回 optuna 最小化方向的目标值。"""
        from sklearn.model_selection import KFold, StratifiedKFold
        y = np.asarray(self.fixed_params["y"])
        if self.task == "classification" and len(np.unique(y)) <= 10:
            splitter = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.seed)
            splits = splitter.split(s, y)
        else:
            splitter = KFold(n_splits=self.cv, shuffle=True, random_state=self.seed)
            splits = splitter.split(s)
        scores = []
        for train_idx, valid_idx in splits:
            model = self._build_model(y[train_idx], native_params)
            model.fit(s.iloc[train_idx].reset_index(drop=True))
            pred = model.transform(s.iloc[valid_idx].reset_index(drop=True))
            y_valid = y[valid_idx]
            values = self._get_pred_values(pred)
            if self.task == "classification":
                from sklearn.metrics import roc_auc_score
                scores.append(-roc_auc_score(y_valid, values))
            else:
                scores.append(float(np.sqrt(np.mean((values - y_valid) ** 2))))
        return float(np.mean(scores))

    def udf_fit(self, s: dataframe_type, **kwargs):
        try:
            import optuna
        except ImportError:
            raise Exception("OptunaTuner requires optuna: pip install optuna")
        if self.model_class is None:
            raise Exception("OptunaTuner requires model_class")
        if self.task == "classification" and len(np.unique(np.asarray(self.fixed_params["y"]))) > 2:
            raise Exception("task='classification' metric=AUC only supports binary (y in {0,1})")
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            native_params = {name: fn(trial) for name, fn in self.search_space.items()}
            return self._cv_score(s, native_params)

        study = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.TPESampler(seed=self.seed))
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        self.best_params_ = study.best_params
        self.best_score_ = study.best_value
        y = np.asarray(self.fixed_params["y"])
        self.model_ = self._build_model(y, self.best_params_)
        self.model_.fit(s, **kwargs)
        return self

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        if self.model_ is None:
            raise Exception("OptunaTuner is not fitted, call fit before transform")
        return self.model_.transform(s, **kwargs)

    def udf_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        if self.model_ is None:
            raise Exception("OptunaTuner is not fitted, call fit before transform")
        return self.model_.transform_single(s, **kwargs)

    def udf_get_params(self) -> dict_type:
        return {"model_class": self.model_class, "fixed_params": self.fixed_params,
                "search_space": self.search_space, "n_trials": self.n_trials,
                "cv": self.cv, "task": self.task, "seed": self.seed,
                "best_params_": self.best_params_, "best_score_": self.best_score_,
                "model_params": self.model_.get_params() if self.model_ is not None else None}

    def udf_set_params(self, params: dict_type):
        self.model_class = params["model_class"]
        self.fixed_params = params["fixed_params"]
        self.search_space = params["search_space"]
        self.n_trials = params["n_trials"]
        self.cv = params["cv"]
        self.task = params["task"]
        self.seed = params["seed"]
        self.best_params_ = params["best_params_"]
        self.best_score_ = params["best_score_"]
        if params["model_params"] is not None and self.model_class is not None:
            self.model_ = self.model_class()
            self.model_.set_params(params["model_params"])
        else:
            self.model_ = None
