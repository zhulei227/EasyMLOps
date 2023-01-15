from ..base import *
import scipy.sparse as sp
import numpy as np
import pandas as pd


class ClassificationPipeObject(PipeObject):
    def __init__(self, y: series_type = None, name=None, transform_check_max_number_error=1e-5,
                 skip_check_transform_type=True, copy_transform_data=True):
        super().__init__(name, transform_check_max_number_error, skip_check_transform_type,
                         copy_transform_data=copy_transform_data)
        self.y = copy.copy(y)
        self.id2label = {}
        self.label2id = {}
        self.num_class = None
        if self.y is not None:
            for idx, label in enumerate(self.y.value_counts().index):
                self.id2label[idx] = label
                self.label2id[label] = idx
            self.y = self.y.apply(lambda x: self.label2id.get(x))
            self.num_class = len(self.id2label)

    @fit_wrapper
    def fit(self, s):
        return self

    @transform_wrapper
    def transform(self, s):
        return s

    @transform_single_wrapper
    def transform_single(self, s: dict_type):
        input_dataframe = pd.DataFrame([s])
        input_dataframe = input_dataframe[self.input_col_names]
        return self.transform(input_dataframe).to_dict("record")[0]

    def _get_params(self) -> dict_type:
        return {"id2label": self.id2label, "label2id": self.label2id, "num_class": self.num_class}

    def _set_params(self, params: dict):
        self.id2label = params["id2label"]
        self.label2id = params["label2id"]
        self.num_class = params["num_class"]

    @staticmethod
    def pd2csr(data: dataframe_type):
        try:
            sparse_matrix = sp.csr_matrix(data.sparse.to_coo().tocsr(), dtype=np.float32)
        except:
            sparse_matrix = sp.csr_matrix(data, dtype=np.float32)
        return sparse_matrix

    @staticmethod
    def pd2dense(data: dataframe_type):
        return data.values


class LGBMClassification(ClassificationPipeObject):
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

    def __init__(self, y: series_type = None, max_depth=3, num_boost_round=256, name=None,
                 transform_check_max_number_error=1e-5,
                 skip_check_transform_type=True, copy_transform_data=True):
        super().__init__(y=y, name=name, transform_check_max_number_error=transform_check_max_number_error,
                         skip_check_transform_type=skip_check_transform_type,
                         copy_transform_data=copy_transform_data)
        self.max_depth = max_depth
        self.num_boost_round = num_boost_round
        self.lgb_model = None

    @fit_wrapper
    def fit(self, s: dataframe_type) -> dataframe_type:
        import lightgbm as lgb
        s_ = self.pd2csr(s)
        params = {
            'objective': 'multiclass',
            'num_class': self.num_class,
            'max_depth': self.max_depth,
            'verbose': -1
        }
        self.lgb_model = lgb.train(params=params, train_set=lgb.Dataset(data=s_, label=self.y),
                                   num_boost_round=self.num_boost_round)
        return self

    @transform_wrapper
    def transform(self, s: dataframe_type) -> dataframe_type:
        s = self.pd2csr(s)
        result = pandas.DataFrame(self.lgb_model.predict(s),
                                  columns=[self.id2label.get(i) for i in range(self.num_class)])
        return result

    def _get_params(self) -> dict_type:
        return {"lgb_model": self.lgb_model}

    def _set_params(self, params: dict_type):
        self.lgb_model = params["lgb_model"]


class LogisticRegressionClassification(ClassificationPipeObject):
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

    def __init__(self, y: series_type = None, multi_class="multinomial", solver="newton-cg", max_iter=1000, name=None,
                 transform_check_max_number_error=1e-5,
                 skip_check_transform_type=True, copy_transform_data=True):
        super().__init__(y=y, name=name, transform_check_max_number_error=transform_check_max_number_error,
                         skip_check_transform_type=skip_check_transform_type,
                         copy_transform_data=copy_transform_data)
        self.multi_class = multi_class
        self.solver = solver
        self.max_iter = max_iter
        self.lr = None

    @fit_wrapper
    def fit(self, s: dataframe_type) -> dataframe_type:
        from sklearn.linear_model import LogisticRegression
        s_ = self.pd2csr(s)
        self.lr = LogisticRegression(multi_class=self.multi_class, solver=self.solver, max_iter=self.max_iter)
        self.lr.fit(s_, self.y)
        return self

    @transform_wrapper
    def transform(self, s: dataframe_type) -> dataframe_type:
        result = pandas.DataFrame(self.lr.predict_proba(s),
                                  columns=[self.id2label.get(i) for i in range(self.num_class)])
        return result

    def _get_params(self) -> dict:
        return {"lr": self.lr}

    def _set_params(self, params: dict):
        self.lr = params["lr"]


class SVMClassification(ClassificationPipeObject):
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

    def __init__(self, y: series_type = None, gamma=2, c=1, kernel="rbf", name=None,
                 transform_check_max_number_error=1e-5,
                 skip_check_transform_type=True, copy_transform_data=True):
        super().__init__(y=y, name=name,
                         transform_check_max_number_error=transform_check_max_number_error,
                         skip_check_transform_type=skip_check_transform_type,
                         copy_transform_data=copy_transform_data)
        self.gamma = gamma
        self.C = c
        self.kernel = kernel
        self.svm = None

    @fit_wrapper
    def fit(self, s: dataframe_type) -> dataframe_type:
        from sklearn.svm import SVC
        s_ = self.pd2csr(s)
        self.svm = SVC(kernel=self.kernel, decision_function_shape="ovo", gamma=self.gamma, C=self.C, probability=True)
        self.svm.fit(s_, self.y)
        return self

    @transform_wrapper
    def transform(self, s: dataframe_type) -> dataframe_type:
        result = pandas.DataFrame(self.svm.predict_proba(s),
                                  columns=[self.id2label.get(i) for i in range(self.num_class)])
        return result

    def _get_params(self) -> dict_type:
        return {"svm": self.svm}

    def _set_params(self, params: dict_type):
        self.svm = params["svm"]


class DecisionTreeClassification(ClassificationPipeObject):
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

    def __init__(self, y: series_type = None, criterion="gini", max_depth=3, min_samples_leaf=16, name=None,
                 transform_check_max_number_error=1e-5,
                 skip_check_transform_type=True, copy_transform_data=True):
        ClassificationPipeObject.__init__(self, y=y, name=name,
                                          transform_check_max_number_error=transform_check_max_number_error,
                                          skip_check_transform_type=skip_check_transform_type,
                                          copy_transform_data=copy_transform_data)
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.tree = None

    @check_dataframe_type
    def fit(self, s: dataframe_type) -> dataframe_type:
        from sklearn.tree import DecisionTreeClassifier
        s_ = self.pd2csr(s)
        self.tree = DecisionTreeClassifier(criterion=self.criterion, max_depth=self.max_depth,
                                           min_samples_leaf=self.min_samples_leaf)
        self.tree.fit(s_, self.y)
        self.input_col_names = s.columns.tolist()
        return self

    def transform(self, s: dataframe_type) -> dataframe_type:
        if self.copy_transform_data:
            s_ = self.pd2csr(s[self.input_col_names])
        else:
            s = self.pd2csr(s[self.input_col_names])
            s_ = s
        result = pandas.DataFrame(self.tree.predict_proba(s_),
                                  columns=[self.id2label.get(i) for i in range(self.num_class)])
        self.output_col_names = result.columns.tolist()
        return result

    def _get_params(self) -> dict_type:
        return {"tree": self.tree}

    def _set_params(self, params: dict_type):
        self.tree = params["tree"]


class RandomForestClassification(ClassificationPipeObject):
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

    def __init__(self, y: series_type = None, n_estimators=128, criterion="gini", max_depth=3, min_samples_leaf=16,
                 name=None,
                 transform_check_max_number_error=1e-5,
                 skip_check_transform_type=True, copy_transform_data=True):
        ClassificationPipeObject.__init__(self, y=y, name=name,
                                          transform_check_max_number_error=transform_check_max_number_error,
                                          skip_check_transform_type=skip_check_transform_type,
                                          copy_transform_data=copy_transform_data)
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.tree = None

    @check_dataframe_type
    def fit(self, s: dataframe_type) -> dataframe_type:
        from sklearn.ensemble import RandomForestClassifier
        s_ = self.pd2csr(s)
        self.tree = RandomForestClassifier(n_estimators=self.n_estimators, criterion=self.criterion,
                                           max_depth=self.max_depth,
                                           min_samples_leaf=self.min_samples_leaf)
        self.tree.fit(s_, self.y)
        self.input_col_names = s.columns.tolist()
        return self

    def transform(self, s: dataframe_type) -> dataframe_type:
        if self.copy_transform_data:
            s_ = self.pd2csr(s[self.input_col_names])
        else:
            s = self.pd2csr(s[self.input_col_names])
            s_ = s
        result = pandas.DataFrame(self.tree.predict_proba(s_),
                                  columns=[self.id2label.get(i) for i in range(self.num_class)])
        self.output_col_names = result.columns.tolist()
        return result

    def _get_params(self) -> dict_type:
        return {"tree": self.tree}

    def _set_params(self, params: dict_type):
        self.tree = params["tree"]


class KNeighborsClassification(ClassificationPipeObject):
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

    def __init__(self, y: series_type = None, n_neighbors=5, name=None,
                 transform_check_max_number_error=1e-5,
                 skip_check_transform_type=True, copy_transform_data=True):
        ClassificationPipeObject.__init__(self, y=y, name=name,
                                          transform_check_max_number_error=transform_check_max_number_error,
                                          skip_check_transform_type=skip_check_transform_type,
                                          copy_transform_data=copy_transform_data)
        self.n_neighbors = n_neighbors
        self.knn = None

    @check_dataframe_type
    def fit(self, s: dataframe_type) -> dataframe_type:
        from sklearn.neighbors import KNeighborsClassifier
        s_ = self.pd2csr(s)
        self.knn = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        self.knn.fit(s_, self.y)
        self.input_col_names = s.columns.tolist()
        return self

    def transform(self, s: dataframe_type) -> dataframe_type:
        if self.copy_transform_data:
            s_ = self.pd2csr(s[self.input_col_names])
        else:
            s = self.pd2csr(s[self.input_col_names])
            s_ = s
        result = pandas.DataFrame(self.knn.predict_proba(s_),
                                  columns=[self.id2label.get(i) for i in range(self.num_class)])
        self.output_col_names = result.columns.tolist()
        return result

    def _get_params(self) -> dict_type:
        return {"knn": self.knn}

    def _set_params(self, params: dict_type):
        self.knn = params["knn"]


class GaussianNBClassification(ClassificationPipeObject):
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

    def __init__(self, y: series_type = None, name=None,
                 transform_check_max_number_error=1e-5,
                 skip_check_transform_type=True, copy_transform_data=True):
        ClassificationPipeObject.__init__(self, y=y, name=name,
                                          transform_check_max_number_error=transform_check_max_number_error,
                                          skip_check_transform_type=skip_check_transform_type,
                                          copy_transform_data=copy_transform_data)
        self.nb = None

    @check_dataframe_type
    def fit(self, s: dataframe_type) -> dataframe_type:
        from sklearn.naive_bayes import GaussianNB
        s_ = self.pd2dense(s)
        self.nb = GaussianNB()
        self.nb.fit(s_, self.y)
        self.input_col_names = s.columns.tolist()
        return self

    def transform(self, s: dataframe_type) -> dataframe_type:
        if self.copy_transform_data:
            s_ = self.pd2dense(s[self.input_col_names])
        else:
            s = self.pd2dense(s[self.input_col_names])
            s_ = s
        result = pandas.DataFrame(self.nb.predict_proba(s_),
                                  columns=[self.id2label.get(i) for i in range(self.num_class)])
        self.output_col_names = result.columns.tolist()
        return result

    def _get_params(self) -> dict_type:
        return {"nb": self.nb}

    def _set_params(self, params: dict_type):
        self.nb = params["nb"]


class MultinomialNBClassification(ClassificationPipeObject):
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

    def __init__(self, y: series_type = None, name=None,
                 transform_check_max_number_error=1e-5,
                 skip_check_transform_type=True, copy_transform_data=True):
        ClassificationPipeObject.__init__(self, y=y, name=name,
                                          transform_check_max_number_error=transform_check_max_number_error,
                                          skip_check_transform_type=skip_check_transform_type,
                                          copy_transform_data=copy_transform_data)
        self.nb = None

    @check_dataframe_type
    def fit(self, s: dataframe_type) -> dataframe_type:
        from sklearn.naive_bayes import MultinomialNB
        s_ = self.pd2csr(s)
        self.nb = MultinomialNB()
        self.nb.fit(s_, self.y)
        self.input_col_names = s.columns.tolist()
        return self

    def transform(self, s: dataframe_type) -> dataframe_type:
        if self.copy_transform_data:
            s_ = self.pd2csr(s[self.input_col_names])
        else:
            s = self.pd2csr(s[self.input_col_names])
            s_ = s
        result = pandas.DataFrame(self.nb.predict_proba(s_),
                                  columns=[self.id2label.get(i) for i in range(self.num_class)])
        self.output_col_names = result.columns.tolist()
        return result

    def _get_params(self) -> dict_type:
        return {"nb": self.nb}

    def _set_params(self, params: dict_type):
        self.nb = params["nb"]


class BernoulliNBClassification(ClassificationPipeObject):
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

    def __init__(self, y: series_type = None, name=None,
                 transform_check_max_number_error=1e-5,
                 skip_check_transform_type=True, copy_transform_data=True):
        ClassificationPipeObject.__init__(self, y=y, name=name,
                                          transform_check_max_number_error=transform_check_max_number_error,
                                          skip_check_transform_type=skip_check_transform_type,
                                          copy_transform_data=copy_transform_data)
        self.nb = None

    @check_dataframe_type
    def fit(self, s: dataframe_type) -> dataframe_type:
        from sklearn.naive_bayes import BernoulliNB
        s_ = self.pd2csr(s)
        self.nb = BernoulliNB()
        self.nb.fit(s_, self.y)
        self.input_col_names = s.columns.tolist()
        return self

    def transform(self, s: dataframe_type) -> dataframe_type:
        if self.copy_transform_data:
            s_ = self.pd2csr(s[self.input_col_names])
        else:
            s = self.pd2csr(s[self.input_col_names])
            s_ = s
        result = pandas.DataFrame(self.nb.predict_proba(s_),
                                  columns=[self.id2label.get(i) for i in range(self.num_class)])
        self.output_col_names = result.columns.tolist()
        return result

    def _get_params(self) -> dict_type:
        return {"nb": self.nb}

    def _set_params(self, params: dict_type):
        self.nb = params["nb"]
