from .base import *
import copy
import scipy.sparse as sp
import numpy as np
import pandas as pd


class ClassificationPipeObject(PipeObject):
    def __init__(self, y: series_type = None, name=None, transform_check_max_number_error=1e-5,
                 skip_check_transform_type=True):
        PipeObject.__init__(self, name, transform_check_max_number_error, skip_check_transform_type)
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

    def fit(self, s):
        return self

    def transform(self, s):
        raise Exception("need to implement")

    @check_dict_type
    def transform_single(self, s: dict_type):
        input_dataframe = pd.DataFrame([s])
        input_dataframe = input_dataframe[self.input_col_names]
        return self.transform(input_dataframe).to_dict("record")[0]

    def get_params(self) -> dict_type:
        params = PipeObject.get_params(self)
        params.update({"id2label": self.id2label, "label2id": self.label2id, "num_class": self.num_class})
        return params

    def set_params(self, params: dict):
        PipeObject.set_params(self, params)
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
                 skip_check_transform_type=True):
        ClassificationPipeObject.__init__(self, y, name, transform_check_max_number_error, skip_check_transform_type)
        self.max_depth = max_depth
        self.num_boost_round = num_boost_round
        self.lgb_model = None

    @check_dataframe_type
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
        self.input_col_names = s.columns.tolist()
        return self

    @check_dataframe_type
    def transform(self, s: dataframe_type) -> dataframe_type:
        s = self.pd2csr(s)
        result = pandas.DataFrame(self.lgb_model.predict(s),
                                  columns=[self.id2label.get(i) for i in range(self.num_class)])
        self.output_col_names = result.columns.tolist()
        return result

    def get_params(self) -> dict_type:
        params = ClassificationPipeObject.get_params(self)
        params.update({"lgb_model": self.lgb_model, "num_class": self.num_class, "id2label": self.id2label})
        return params

    def set_params(self, params: dict_type):
        ClassificationPipeObject.set_params(self, params)
        self.lgb_model = params["lgb_model"]
        self.num_class = params["num_class"]
        self.id2label = params["id2label"]


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
                 skip_check_transform_type=True):
        ClassificationPipeObject.__init__(self, y, name, transform_check_max_number_error, skip_check_transform_type)
        self.multi_class = multi_class
        self.solver = solver
        self.max_iter = max_iter
        self.lr = None

    @check_dataframe_type
    def fit(self, s: dataframe_type) -> dataframe_type:
        from sklearn.linear_model import LogisticRegression
        s_ = self.pd2csr(s)
        self.lr = LogisticRegression(multi_class=self.multi_class, solver=self.solver, max_iter=self.max_iter)
        self.lr.fit(s_, self.y)
        self.input_col_names = s.columns.tolist()
        return self

    def transform(self, s: dataframe_type) -> dataframe_type:
        s = self.pd2csr(s)

        result = pandas.DataFrame(self.lr.predict_proba(s),
                                  columns=[self.id2label.get(i) for i in range(self.num_class)])
        self.output_col_names = result.columns.tolist()
        return result

    def get_params(self) -> dict:
        params = ClassificationPipeObject.get_params(self)
        params.update({"lr": self.lr, "num_class": self.num_class, "id2label": self.id2label})
        return params

    def set_params(self, params: dict):
        ClassificationPipeObject.set_params(self, params)
        self.lr = params["lr"]
        self.num_class = params["num_class"]
        self.id2label = params["id2label"]


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
                 skip_check_transform_type=True):
        ClassificationPipeObject.__init__(self, y, name, transform_check_max_number_error, skip_check_transform_type)
        self.gamma = gamma
        self.C = c
        self.kernel = kernel
        self.svm = None

    @check_dataframe_type
    def fit(self, s: dataframe_type) -> dataframe_type:
        from sklearn.svm import SVC
        s_ = self.pd2csr(s)
        self.svm = SVC(kernel=self.kernel, decision_function_shape="ovo", gamma=self.gamma, C=self.C, probability=True)
        self.svm.fit(s_, self.y)
        self.input_col_names = s.columns.tolist()
        return self

    def transform(self, s: dataframe_type) -> dataframe_type:
        s = self.pd2csr(s)
        result = pandas.DataFrame(self.svm.predict_proba(s),
                                  columns=[self.id2label.get(i) for i in range(self.num_class)])
        self.output_col_names = result.columns.tolist()
        return result

    def get_params(self) -> dict_type:
        params = ClassificationPipeObject.get_params(self)
        params.update({"svm": self.svm, "num_class": self.num_class, "id2label": self.id2label})
        return params

    def set_params(self, params: dict_type):
        ClassificationPipeObject.set_params(self, params)
        self.svm = params["svm"]
        self.num_class = params["num_class"]
        self.id2label = params["id2label"]


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
                 skip_check_transform_type=True):
        ClassificationPipeObject.__init__(self, y, name, transform_check_max_number_error, skip_check_transform_type)
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
        s = self.pd2csr(s)
        result = pandas.DataFrame(self.tree.predict_proba(s),
                                  columns=[self.id2label.get(i) for i in range(self.num_class)])
        self.output_col_names = result.columns.tolist()
        return result

    def get_params(self) -> dict_type:
        params = ClassificationPipeObject.get_params(self)
        params.update({"tree": self.tree, "num_class": self.num_class, "id2label": self.id2label})
        return params

    def set_params(self, params: dict_type):
        ClassificationPipeObject.set_params(self, params)
        self.tree = params["tree"]
        self.num_class = params["num_class"]
        self.id2label = params["id2label"]


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
                 skip_check_transform_type=True):
        ClassificationPipeObject.__init__(self, y, name, transform_check_max_number_error, skip_check_transform_type)
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
        s = self.pd2csr(s)
        result = pandas.DataFrame(self.tree.predict_proba(s),
                                  columns=[self.id2label.get(i) for i in range(self.num_class)])
        self.output_col_names = result.columns.tolist()
        return result

    def get_params(self) -> dict_type:
        params = ClassificationPipeObject.get_params(self)
        params.update({"tree": self.tree, "num_class": self.num_class, "id2label": self.id2label})
        return params

    def set_params(self, params: dict_type):
        ClassificationPipeObject.set_params(self, params)
        self.tree = params["tree"]
        self.num_class = params["num_class"]
        self.id2label = params["id2label"]


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
                 skip_check_transform_type=True):
        ClassificationPipeObject.__init__(self, y, name, transform_check_max_number_error, skip_check_transform_type)
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
        s = self.pd2csr(s)
        result = pandas.DataFrame(self.knn.predict_proba(s),
                                  columns=[self.id2label.get(i) for i in range(self.num_class)])
        self.output_col_names = result.columns.tolist()
        return result

    def get_params(self) -> dict_type:
        params = ClassificationPipeObject.get_params(self)
        params.update({"knn": self.knn, "num_class": self.num_class, "id2label": self.id2label})
        return params

    def set_params(self, params: dict_type):
        ClassificationPipeObject.set_params(self, params)
        self.knn = params["knn"]
        self.num_class = params["num_class"]
        self.id2label = params["id2label"]


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
                 skip_check_transform_type=True):
        ClassificationPipeObject.__init__(self, y, name, transform_check_max_number_error, skip_check_transform_type)
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
        s = self.pd2dense(s)
        result = pandas.DataFrame(self.nb.predict_proba(s),
                                  columns=[self.id2label.get(i) for i in range(self.num_class)])
        self.output_col_names = result.columns.tolist()
        return result

    def get_params(self) -> dict_type:
        params = ClassificationPipeObject.get_params(self)
        params.update({"nb": self.nb, "num_class": self.num_class, "id2label": self.id2label})
        return params

    def set_params(self, params: dict_type):
        ClassificationPipeObject.set_params(self, params)
        self.nb = params["nb"]
        self.num_class = params["num_class"]
        self.id2label = params["id2label"]


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
                 skip_check_transform_type=True):
        ClassificationPipeObject.__init__(self, y, name, transform_check_max_number_error, skip_check_transform_type)
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
        s = self.pd2csr(s)
        result = pandas.DataFrame(self.nb.predict_proba(s),
                                  columns=[self.id2label.get(i) for i in range(self.num_class)])
        self.output_col_names = result.columns.tolist()
        return result

    def get_params(self) -> dict_type:
        params = ClassificationPipeObject.get_params(self)
        params.update({"nb": self.nb, "num_class": self.num_class, "id2label": self.id2label})
        return params

    def set_params(self, params: dict_type):
        ClassificationPipeObject.set_params(self, params)
        self.nb = params["nb"]
        self.num_class = params["num_class"]
        self.id2label = params["id2label"]


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
                 skip_check_transform_type=True):
        ClassificationPipeObject.__init__(self, y, name, transform_check_max_number_error, skip_check_transform_type)
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
        s = self.pd2csr(s)
        result = pandas.DataFrame(self.nb.predict_proba(s),
                                  columns=[self.id2label.get(i) for i in range(self.num_class)])
        self.output_col_names = result.columns.tolist()
        return result

    def get_params(self) -> dict_type:
        params = ClassificationPipeObject.get_params(self)
        params.update({"nb": self.nb, "num_class": self.num_class, "id2label": self.id2label})
        return params

    def set_params(self, params: dict_type):
        ClassificationPipeObject.set_params(self, params)
        self.nb = params["nb"]
        self.num_class = params["num_class"]
        self.id2label = params["id2label"]
