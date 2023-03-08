from easymlops.table.core import *
import pandas as pd
from easymlops.table.utils import PandasUtils


class ClassificationBase(TablePipeObjectBase):
    def __init__(self, y: series_type = None, cols="all", skip_check_transform_type=True, drop_input_data=True,
                 native_init_params=None, native_fit_params=None, support_sparse_input=False,
                 **kwargs):
        super().__init__(skip_check_transform_type=skip_check_transform_type, **kwargs)
        if cols is None or type(cols) == str:
            self.cols = []
        self.support_sparse_input = support_sparse_input
        self.drop_input_data = drop_input_data
        self.y = copy.deepcopy(y)
        self.id2label = {}
        self.label2id = {}
        self.num_class = None
        if self.y is not None:
            for idx, label in enumerate(self.y.value_counts().index):
                self.id2label[idx] = label
                self.label2id[label] = idx
            self.y = self.y.apply(lambda x: self.label2id.get(x))
            self.num_class = len(self.id2label)
        # 底层模型自带参数
        self.native_init_params = native_init_params
        self.native_fit_params = native_fit_params
        if self.native_init_params is None:
            self.native_init_params = dict()
        if self.native_fit_params is None:
            self.native_fit_params = dict()

    def before_fit(self, s: dataframe_type, **kwargs) -> dataframe_type:
        s = super().before_fit(s, **kwargs)
        assert self.y is not None
        if len(self.cols) == 0:
            self.cols = s.columns.tolist()
        assert type(self.cols) == list
        if self.check_list_same(s.columns.tolist(), self.cols):
            return s
        else:
            return s[self.cols]

    def before_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        s = super().before_transform(s, **kwargs)
        if self.check_list_same(s.columns.tolist(), self.cols):
            return s
        else:
            return s[self.cols]

    def transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        s_ = self.after_transform(self.udf_transform(self.before_transform(s, **kwargs), **kwargs), **kwargs)
        if not self.drop_input_data:
            s_ = PandasUtils.concat_duplicate_columns([s, s_])
        return s_

    def before_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        s = super().before_transform_single(s, **kwargs)
        return self.extract_dict(s, self.cols)

    def transform_single(self, s: dict_type, **kwargs) -> dict_type:
        s_ = copy.deepcopy(s)
        s_ = self.after_transform_single(
            self.udf_transform_single(self.before_transform_single(s_, **kwargs), **kwargs), **kwargs)
        if not self.drop_input_data:
            s.update(s_)
            return s
        else:
            return s_

    def udf_fit(self, s, **kwargs):
        return self

    def udf_transform(self, s, **kwargs):
        return s

    def udf_transform_single(self, s: dict_type, **kwargs):
        input_dataframe = pd.DataFrame([s])
        input_dataframe = input_dataframe[self.cols]
        return self.udf_transform(input_dataframe, **kwargs).to_dict("record")[0]

    def udf_get_params(self) -> dict_type:
        return {"id2label": self.id2label, "label2id": self.label2id, "num_class": self.num_class, "cols": self.cols,
                "drop_input_data": self.drop_input_data, "native_init_params": self.native_init_params,
                "native_fit_params": self.native_fit_params, "support_sparse_input": self.support_sparse_input}

    def udf_set_params(self, params: dict):
        self.id2label = params["id2label"]
        self.label2id = params["label2id"]
        self.num_class = params["num_class"]
        self.cols = params["cols"]
        self.drop_input_data = params["drop_input_data"]
        self.native_init_params = params["native_init_params"]
        self.native_fit_params = params["native_fit_params"]
        self.support_sparse_input = params["support_sparse_input"]


class LGBMClassification(ClassificationBase):
    """
    input type:pandas.dataframe
    input like:
    | 0 | 1 | 2 | 3 |
    |0.2|0.3|0.4|0.5|
    |0.3|0.2|0.6|0.1|
    label type:pandas:series
    label like:
    |label|
    |good|
    |bad|
    -------------------------
    output type:pandas.dataframe
    output like:
    |good|bad|
    |0.5 |0.5|
    |0.5 |0.5|
    """

    def __init__(self, y=None, verbose=-1, objective="multiclass",
                 use_faster_predictor=True, support_sparse_input=False, dataset_params=None, **kwargs):
        super().__init__(y=y, support_sparse_input=support_sparse_input, **kwargs)
        self.native_init_params.update({
            'objective': objective,
            'num_class': self.num_class,
            'verbose': verbose
        })
        self.lgb_model = None
        self.use_faster_predictor = use_faster_predictor
        self.lgb_model_faster_predictor_params = None
        self.lgb_model_faster_predictor = None
        self.dataset_params = dataset_params
        if self.dataset_params is None:
            self.dataset_params = dict()

    def udf_fit(self, s: dataframe_type, **kwargs):
        import lightgbm as lgb
        if self.support_sparse_input:
            s_ = PandasUtils.pd2csr(s)
        else:
            s_ = s
        self.lgb_model = lgb.train(params=self.native_init_params,
                                   train_set=lgb.Dataset(data=s_, label=self.y, feature_name=list(s.columns),
                                                         **self.dataset_params),
                                   **self.native_fit_params)
        if self.use_faster_predictor:
            from easymlops.table.utils import FasterLgbMulticlassPredictor
            self.lgb_model_faster_predictor_params = self.lgb_model.dump_model()
            self.lgb_model_faster_predictor = FasterLgbMulticlassPredictor(
                model=self.lgb_model_faster_predictor_params, cache_num=10)
        return self

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        if self.support_sparse_input:
            s_ = PandasUtils.pd2csr(s)
        else:
            s_ = s
        result = pd.DataFrame(self.lgb_model.predict(s_),
                              columns=[self.id2label.get(i) for i in range(self.num_class)], index=s.index)
        return result

    def udf_transform_single(self, s: dict_type, **kwargs):
        if self.use_faster_predictor:
            return self.lgb_model_faster_predictor.predict(s).get("score")
        else:
            input_dataframe = pd.DataFrame([s])
            input_dataframe = input_dataframe[self.cols]
            return self.udf_transform(input_dataframe, **kwargs).to_dict("record")[0]

    def udf_get_params(self) -> dict_type:
        params = {"lgb_model": self.lgb_model, "use_faster_predictor": self.use_faster_predictor}
        if self.use_faster_predictor:
            params["lgb_model_faster_predictor_params"] = self.lgb_model_faster_predictor_params
        return params

    def udf_set_params(self, params: dict_type):
        self.lgb_model = params["lgb_model"]
        self.use_faster_predictor = params["use_faster_predictor"]
        if self.use_faster_predictor:
            from easymlops.table.utils import FasterLgbMulticlassPredictor
            self.lgb_model_faster_predictor_params = params["lgb_model_faster_predictor_params"]
            self.lgb_model_faster_predictor = FasterLgbMulticlassPredictor(
                model=self.lgb_model_faster_predictor_params, cache_num=10)

    def get_contrib(self, s: dict_type) -> dict_type:
        """
        获取sabaas特征重要性
        """
        assert self.use_faster_predictor
        return self.lgb_model_faster_predictor.predict(s).get("contrib")


class LogisticRegressionClassification(ClassificationBase):
    """
    input type:pandas.dataframe
    input like:
    | 0 | 1 | 2 | 3 |
    |0.2|0.3|0.4|0.5|
    |0.3|0.2|0.6|0.1|
    label type:pandas:series
    label like:
    |label|
    |good|
    |bad|
    -------------------------
    output type:pandas.dataframe
    output like:
    |good|bad|
    |0.5 |0.5|
    |0.5 |0.5|
    """

    def __init__(self, y=None, multi_class="multinomial", solver="newton-cg", max_iter=1000,
                 **kwargs):
        super().__init__(y=y, **kwargs)
        self.native_init_params.update({"multi_class": multi_class, "solver": solver, "max_iter": max_iter})
        self.lr = None

    def udf_fit(self, s: dataframe_type, **kwargs):
        if self.support_sparse_input:
            s_ = PandasUtils.pd2csr(s)
        else:
            s_ = s
        from sklearn.linear_model import LogisticRegression
        self.lr = LogisticRegression(**self.native_init_params)
        self.lr.fit(s_, self.y, **self.native_fit_params)
        return self

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        if self.support_sparse_input:
            s_ = PandasUtils.pd2csr(s)
        else:
            s_ = s
        result = pd.DataFrame(self.lr.predict_proba(s_),
                              columns=[self.id2label.get(i) for i in range(self.num_class)], index=s.index)
        return result

    def udf_get_params(self) -> dict:
        return {"lr": self.lr}

    def udf_set_params(self, params: dict):
        self.lr = params["lr"]


class SVMClassification(ClassificationBase):
    """
    input type:pandas.dataframe
    input like:
    | 0 | 1 | 2 | 3 |
    |0.2|0.3|0.4|0.5|
    |0.3|0.2|0.6|0.1|
    label type:pandas:series
    label like:
    |label|
    |good|
    |bad|
    -------------------------
    output type:pandas.dataframe
    output like:
    |good|bad|
    |0.5 |0.5|
    |0.5 |0.5|
    """

    def __init__(self, y=None, gamma=2, c=1, kernel="rbf", **kwargs):
        super().__init__(y=y, **kwargs)
        self.native_init_params.update(
            {"kernel": kernel, "decision_function_shape": "ovo", "gamma": gamma, "C": c, "probability": True})
        self.svm = None

    def udf_fit(self, s: dataframe_type, **kwargs):
        if self.support_sparse_input:
            s_ = PandasUtils.pd2csr(s)
        else:
            s_ = s
        from sklearn.svm import SVC
        self.svm = SVC(**self.native_init_params)
        self.svm.fit(s_, self.y, **self.native_fit_params)
        return self

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        if self.support_sparse_input:
            s_ = PandasUtils.pd2csr(s)
        else:
            s_ = s
        result = pd.DataFrame(self.svm.predict_proba(s_),
                              columns=[self.id2label.get(i) for i in range(self.num_class)], index=s.index)
        return result

    def udf_get_params(self) -> dict_type:
        return {"svm": self.svm}

    def udf_set_params(self, params: dict_type):
        self.svm = params["svm"]


class DecisionTreeClassification(ClassificationBase):
    """
    input type:pandas.dataframe
    input like:
    | 0 | 1 | 2 | 3 |
    |0.2|0.3|0.4|0.5|
    |0.3|0.2|0.6|0.1|
    label type:pandas:series
    label like:
    |label|
    |good|
    |bad|
    -------------------------
    output type:pandas.dataframe
    output like:
    |good|bad|
    |0.5 |0.5|
    |0.5 |0.5|
    """

    def __init__(self, y=None, criterion="gini", max_depth=3, min_samples_leaf=16, **kwargs):
        super().__init__(y=y, **kwargs)
        self.native_init_params.update(
            {"criterion": criterion, "max_depth": max_depth, "min_samples_leaf": min_samples_leaf})
        self.tree = None

    def udf_fit(self, s: dataframe_type, **kwargs):
        if self.support_sparse_input:
            s_ = PandasUtils.pd2csr(s)
        else:
            s_ = s
        from sklearn.tree import DecisionTreeClassifier
        self.tree = DecisionTreeClassifier(**self.native_init_params)
        self.tree.fit(s_, self.y, **self.native_fit_params)
        self.input_col_names = s.columns.tolist()
        return self

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        if self.support_sparse_input:
            s_ = PandasUtils.pd2csr(s)
        else:
            s_ = s
        result = pd.DataFrame(self.tree.predict_proba(s_),
                              columns=[self.id2label.get(i) for i in range(self.num_class)], index=s.index)
        return result

    def udf_get_params(self) -> dict_type:
        return {"tree": self.tree}

    def udf_set_params(self, params: dict_type):
        self.tree = params["tree"]


class RandomForestClassification(ClassificationBase):
    """
    input type:pandas.dataframe
    input like:
    | 0 | 1 | 2 | 3 |
    |0.2|0.3|0.4|0.5|
    |0.3|0.2|0.6|0.1|
    label type:pandas:series
    label like:
    |label|
    |good|
    |bad|
    -------------------------
    output type:pandas.dataframe
    output like:
    |good|bad|
    |0.5 |0.5|
    |0.5 |0.5|
    """

    def __init__(self, y=None, n_estimators=128, criterion="gini", max_depth=3, min_samples_leaf=16, **kwargs):
        super().__init__(y=y, **kwargs)
        self.native_init_params.update({"n_estimators": n_estimators, "criterion": criterion, "max_depth": max_depth,
                                        "min_samples_leaf": min_samples_leaf})
        self.tree = None

    def udf_fit(self, s: dataframe_type, **kwargs):
        if self.support_sparse_input:
            s_ = PandasUtils.pd2csr(s)
        else:
            s_ = s
        from sklearn.ensemble import RandomForestClassifier
        self.tree = RandomForestClassifier(**self.native_init_params)
        self.tree.fit(s_, self.y, **self.native_fit_params)
        return self

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        if self.support_sparse_input:
            s_ = PandasUtils.pd2csr(s)
        else:
            s_ = s
        result = pd.DataFrame(self.tree.predict_proba(s_),
                              columns=[self.id2label.get(i) for i in range(self.num_class)], index=s.index)
        return result

    def udf_get_params(self) -> dict_type:
        return {"tree": self.tree}

    def udf_set_params(self, params: dict_type):
        self.tree = params["tree"]


class KNeighborsClassification(ClassificationBase):
    """
    input type:pandas.dataframe
    input like:
    | 0 | 1 | 2 | 3 |
    |0.2|0.3|0.4|0.5|
    |0.3|0.2|0.6|0.1|
    label type:pandas:series
    label like:
    |label|
    |good|
    |bad|
    -------------------------
    output type:pandas.dataframe
    output like:
    |good|bad|
    |0.5 |0.5|
    |0.5 |0.5|
    """

    def __init__(self, y=None, n_neighbors=5, **kwargs):
        super().__init__(y=y, **kwargs)
        self.native_init_params.update(self.native_fit_params)
        self.native_init_params.update({"n_neighbors": n_neighbors})
        self.knn = None

    def udf_fit(self, s: dataframe_type, **kwargs):
        if self.support_sparse_input:
            s_ = PandasUtils.pd2csr(s)
        else:
            s_ = s
        from sklearn.neighbors import KNeighborsClassifier
        self.knn = KNeighborsClassifier(**self.native_init_params)
        self.knn.fit(s_, self.y)
        return self

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        if self.support_sparse_input:
            s_ = PandasUtils.pd2csr(s)
        else:
            s_ = s
        result = pd.DataFrame(self.knn.predict_proba(s_),
                              columns=[self.id2label.get(i) for i in range(self.num_class)], index=s.index)
        return result

    def udf_get_params(self) -> dict_type:
        return {"knn": self.knn}

    def udf_set_params(self, params: dict_type):
        self.knn = params["knn"]


class GaussianNBClassification(ClassificationBase):
    """
    input type:pandas.dataframe
    input like:
    | 0 | 1 | 2 | 3 |
    |0.2|0.3|0.4|0.5|
    |0.3|0.2|0.6|0.1|
    label type:pandas:series
    label like:
    |label|
    |good|
    |bad|
    -------------------------
    output type:pandas.dataframe
    output like:
    |good|bad|
    |0.5 |0.5|
    |0.5 |0.5|
    """

    def __init__(self, y=None, **kwargs):
        super().__init__(y=y, **kwargs)
        self.nb = None

    def udf_fit(self, s: dataframe_type, **kwargs):
        from sklearn.naive_bayes import GaussianNB
        self.nb = GaussianNB(**self.native_init_params)
        self.nb.fit(PandasUtils.pd2dense(s), self.y, **self.native_fit_params)
        return self

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        result = pd.DataFrame(self.nb.predict_proba(PandasUtils.pd2dense(s)),
                              columns=[self.id2label.get(i) for i in range(self.num_class)], index=s.index)
        return result

    def udf_get_params(self) -> dict_type:
        return {"nb": self.nb}

    def udf_set_params(self, params: dict_type):
        self.nb = params["nb"]


class MultinomialNBClassification(ClassificationBase):
    """
    input type:pandas.dataframe
    input like:
    | 0 | 1 | 2 | 3 |
    |0.2|0.3|0.4|0.5|
    |0.3|0.2|0.6|0.1|
    label type:pandas:series
    label like:
    |label|
    |good|
    |bad|
    -------------------------
    output type:pandas.dataframe
    output like:
    |good|bad|
    |0.5 |0.5|
    |0.5 |0.5|
    """

    def __init__(self, y=None, **kwargs):
        super().__init__(y=y, **kwargs)
        self.nb = None

    def udf_fit(self, s: dataframe_type, **kwargs):
        if self.support_sparse_input:
            s_ = PandasUtils.pd2csr(s)
        else:
            s_ = s
        from sklearn.naive_bayes import MultinomialNB
        self.nb = MultinomialNB(**self.native_init_params)
        self.nb.fit(s_, self.y, **self.native_fit_params)
        return self

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        if self.support_sparse_input:
            s_ = PandasUtils.pd2csr(s)
        else:
            s_ = s
        result = pd.DataFrame(self.nb.predict_proba(s_),
                              columns=[self.id2label.get(i) for i in range(self.num_class)], index=s.index)
        return result

    def udf_get_params(self) -> dict_type:
        return {"nb": self.nb}

    def udf_set_params(self, params: dict_type):
        self.nb = params["nb"]


class BernoulliNBClassification(ClassificationBase):
    """
    input type:pandas.dataframe
    input like:
    | 0 | 1 | 2 | 3 |
    |0.2|0.3|0.4|0.5|
    |0.3|0.2|0.6|0.1|
    label type:pandas:series
    label like:
    |label|
    |good|
    |bad|
    -------------------------
    output type:pandas.dataframe
    output like:
    |good|bad|
    |0.5 |0.5|
    |0.5 |0.5|
    """

    def __init__(self, y=None, **kwargs):
        super().__init__(y=y, **kwargs)
        self.nb = None

    def udf_fit(self, s: dataframe_type, **kwargs):
        if self.support_sparse_input:
            s_ = PandasUtils.pd2csr(s)
        else:
            s_ = s
        from sklearn.naive_bayes import BernoulliNB
        self.nb = BernoulliNB(**self.native_init_params)
        self.nb.fit(s_, self.y, **self.native_fit_params)
        return self

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        if self.support_sparse_input:
            s_ = PandasUtils.pd2csr(s)
        else:
            s_ = s
        result = pd.DataFrame(self.nb.predict_proba(s_),
                              columns=[self.id2label.get(i) for i in range(self.num_class)], index=s.index)
        return result

    def udf_get_params(self) -> dict_type:
        return {"nb": self.nb}

    def udf_set_params(self, params: dict_type):
        self.nb = params["nb"]
