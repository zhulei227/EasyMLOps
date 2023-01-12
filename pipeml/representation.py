from .base import *
from .preprocessing import FillNa
import pandas as pd


class UserDefinedRepresentationObject(PipeObject):
    """
    只需要对单个key-value处理的类型
    """

    def __init__(self, cols="all", y=None, name=None, transform_check_max_number_error=1e-5,
                 skip_check_transform_type=False, skip_check_transform_value=False):
        PipeObject.__init__(self, name=name, transform_check_max_number_error=transform_check_max_number_error,
                            skip_check_transform_type=skip_check_transform_type,
                            skip_check_transform_value=skip_check_transform_value)
        self.cols = cols
        self.y = y

    def _user_defined_function(self, col, x):
        """
        对col的对应值x执行某种自定义操作
        """
        raise Exception("need to implement")

    @staticmethod
    def extract_dict(s: dict_type, keys: list):
        new_s = dict()
        for key in keys:
            new_s[key] = s[key]
        return new_s

    @check_dataframe_type
    def _fit(self, s: dataframe_type) -> dataframe_type:
        """
        补充fit阶段的自定义操作
        """
        return self

    @check_dataframe_type
    def fit(self, s: dataframe_type) -> dataframe_type:
        if str(self.cols) == "all" or self.cols is None or (type(self.cols) == list and len(self.cols) == 0):
            self.cols = []
            for col in s.columns.tolist():
                self.cols.append((col, col))
        else:
            if type(self.cols) == list:
                if type(self.cols[0]) == tuple or type(self.cols[0]) == list:
                    pass
                else:
                    new_cols = []
                    for col in self.cols:
                        new_cols.append((col, col))
                    self.cols = new_cols
            else:
                raise Exception("cols should be None,'all' or a list")
        self.input_col_names = s.columns.tolist()
        self._fit(s)
        return self

    @check_dataframe_type
    def transform(self, s: dataframe_type) -> dataframe_type:
        s_ = s[self.input_col_names]
        for col, new_col in self.cols:
            if col in s_.columns:
                s_[new_col] = s_[col].apply(lambda x: self._user_defined_function(col, x))
        self.output_col_names = s_.columns.tolist()
        return s_

    @check_dict_type
    def transform_single(self, s: dict_type) -> dict_type:
        s_ = self.extract_dict(s, self.input_col_names)
        for col, new_col in self.cols:
            if col in s_.keys():
                s_[new_col] = self._user_defined_function(col, s_[col])
        return self.extract_dict(s_, self.output_col_names)

    def get_params(self) -> dict_type:
        params = PipeObject.get_params(self)
        params.update({"cols": self.cols})
        return params

    def set_params(self, params: dict):
        PipeObject.set_params(self, params)
        self.cols = params["cols"]


class TargetEncoding(UserDefinedRepresentationObject):
    def __init__(self, cols="all", y=None, name=None, error_value=0,
                 transform_check_max_number_error=1e-3):
        UserDefinedRepresentationObject.__init__(self, cols=cols, name=name, y=y,
                                                 transform_check_max_number_error=transform_check_max_number_error,
                                                 skip_check_transform_type=True)
        self.error_value = error_value
        self.target_map_detail = dict()

    @check_dataframe_type
    def _fit(self, s: dataframe_type):
        s_ = s
        s_["y_"] = self.y
        for col, _ in self.cols:
            tmp_ = s_[[col, "y_"]]
            col_map = list(tmp_.groupby([col]).agg({"y_": ["mean"]}).to_dict().values())[0]
            self.target_map_detail[col] = col_map
        del s_["y_"]
        return self

    def _user_defined_function(self, col, x):
        map_detail_ = self.target_map_detail.get(col, dict())
        return map_detail_.get(x, self.error_value)

    def get_params(self) -> dict_type:
        params = UserDefinedRepresentationObject.get_params(self)
        params.update({"target_map_detail": self.target_map_detail, "error_value": self.error_value})
        return params

    def set_params(self, params: dict):
        UserDefinedRepresentationObject.set_params(self, params)
        self.target_map_detail = params["target_map_detail"]
        self.error_value = params["error_value"]


class LabelEncoding(UserDefinedRepresentationObject):
    def __init__(self, cols="all", name=None, error_value=0,
                 transform_check_max_number_error=1e-3):
        UserDefinedRepresentationObject.__init__(self, cols=cols, name=name,
                                                 transform_check_max_number_error=transform_check_max_number_error,
                                                 skip_check_transform_type=True)
        self.error_value = error_value
        self.label_map_detail = dict()

    @check_dataframe_type
    def _fit(self, s: dataframe_type):
        s_ = s
        for col, _ in self.cols:
            col_map = s_[col].value_counts().to_dict()
            c = 1
            for key, value in col_map.items():
                col_map[key] = c
                c += 1
            self.label_map_detail[col] = col_map
        return self

    def _user_defined_function(self, col, x):
        map_detail_ = self.label_map_detail.get(col, dict())
        return map_detail_.get(x, self.error_value)

    def get_params(self) -> dict_type:
        params = UserDefinedRepresentationObject.get_params(self)
        params.update({"label_map_detail": self.label_map_detail, "error_value": self.error_value})
        return params

    def set_params(self, params: dict):
        UserDefinedRepresentationObject.set_params(self, params)
        self.label_map_detail = params["label_map_detail"]
        self.error_value = params["error_value"]


class OneHotEncoding(UserDefinedRepresentationObject):
    def __init__(self, cols="all", name=None, drop_col=True,
                 transform_check_max_number_error=1e-3):
        UserDefinedRepresentationObject.__init__(self, cols=cols, name=name,
                                                 transform_check_max_number_error=transform_check_max_number_error,
                                                 skip_check_transform_type=True)
        self.drop_col = drop_col
        self.one_hot_detail = dict()
        self.fill_na_model = None

    @check_dataframe_type
    def _fit(self, s: dataframe_type):
        s_ = copy.copy(s)
        self.fill_na_model = FillNa(cols=[col for col, _ in self.cols], fill_value="nan")
        s_ = self.fill_na_model.fit(s_).transform(s_)
        for col, _ in self.cols:
            self.one_hot_detail[col] = s_[col].unique().tolist()
        return self

    @check_dataframe_type
    def transform(self, s: dataframe_type) -> dataframe_type:
        s_ = s[self.input_col_names]
        s_ = self.fill_na_model.transform(s_)
        for col, new_col in self.cols:
            if col not in self.one_hot_detail.keys():
                raise Exception("{} not in {}".format(col, self.one_hot_detail.keys()))
            values = self.one_hot_detail.get(col)
            for value in values:
                s_["{}_{}".format(new_col, value)] = (s_[col] == value).astype(int)
        if self.drop_col:
            for col in self.one_hot_detail.keys():
                del s_[col]
        self.output_col_names = s_.columns.tolist()
        return s_

    @check_dict_type
    def transform_single(self, s: dict_type) -> dict_type:
        s_ = self.extract_dict(s, self.input_col_names)
        s_ = self.fill_na_model.transform_single(s_)
        for col, new_col in self.cols:
            if col not in self.one_hot_detail.keys():
                raise Exception("{} not in {}".format(col, self.one_hot_detail.keys()))
            values = self.one_hot_detail.get(col)
            for value in values:
                if s_[col] == value:
                    s_["{}_{}".format(new_col, value)] = 1
                else:
                    s_["{}_{}".format(new_col, value)] = 0
        if self.drop_col:
            for col in self.one_hot_detail.keys():
                del s_[col]
        return self.extract_dict(s_, self.output_col_names)

    def get_params(self) -> dict_type:
        params = UserDefinedRepresentationObject.get_params(self)
        params.update({"fill_na_model": self.fill_na_model.get_params(), "one_hot_detail": self.one_hot_detail,
                       "drop_col": self.drop_col})
        return params

    def set_params(self, params: dict):
        UserDefinedRepresentationObject.set_params(self, params)
        self.one_hot_detail = params["one_hot_detail"]
        self.drop_col = params["drop_col"]
        self.fill_na_model = FillNa()
        self.fill_na_model.set_params(params["fill_na_model"])


class PCADecomposition(PipeObject):
    """
    input type:pandas.dataframe
    input like:
    | 0 | 1 | 2 | 3 |
    |0.2|0.3|0.4|0.5|
    |0.3|0.2|0.6|0.1|
    ----------------------------
    output type:pandas.dataframe
    output like:
    | 0 | 1 |
    |-0.4|0.4|
    |0.2|0.6|
    """

    def __init__(self, n_components=3, name=None, transform_check_max_number_error=1e-5,
                 skip_check_transform_type=True):
        PipeObject.__init__(self, name, transform_check_max_number_error, skip_check_transform_type)
        self.n_components = n_components
        self.pca = None

    @check_dataframe_type
    def fit(self, s: dataframe_type) -> dataframe_type:
        from sklearn.decomposition import PCA
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(s.fillna(0).values)
        self.input_col_names = s.columns.tolist()
        return self

    @check_dataframe_type
    def transform(self, s: dataframe_type) -> dataframe_type:
        result = pandas.DataFrame(self.pca.transform(s[self.input_col_names].fillna(0).values))
        self.output_col_names = result.columns.tolist()
        return result

    @check_dict_type
    def transform_single(self, s: dict_type):
        input_dataframe = pd.DataFrame([s])
        input_dataframe = input_dataframe[self.input_col_names]
        return self.transform(input_dataframe).to_dict("record")[0]

    def get_params(self) -> dict_type:
        params = PipeObject.get_params(self)
        params.update({"pca": self.pca, "n_components": self.n_components})
        return params

    def set_params(self, params: dict):
        PipeObject.set_params(self, params)
        self.pca = params["pca"]
        self.n_components = params["n_components"]


class NMFDecomposition(PipeObject):
    """
    input type:pandas.dataframe
    input like:
    | 0 | 1 | 2 | 3 |
    |0.2|0.3|0.4|0.5|
    |0.3|0.2|0.6|0.1|
    ----------------------------
    output type:pandas.dataframe
    output like:
    | 0 | 1 |
    |0.4|0.4|
    |0.2|0.6|
    """

    def __init__(self, n_components=3, name=None, transform_check_max_number_error=1e-5,
                 skip_check_transform_type=True):
        PipeObject.__init__(self, name, transform_check_max_number_error, skip_check_transform_type)
        self.n_components = n_components
        self.nmf = None

    @check_dataframe_type
    def fit(self, s: dataframe_type) -> dataframe_type:
        from sklearn.decomposition import NMF
        self.nmf = NMF(n_components=self.n_components)
        self.nmf.fit(s.fillna(0).values)
        self.input_col_names = s.columns.tolist()
        return self

    @check_dataframe_type
    def transform(self, s: dataframe_type) -> dataframe_type:
        result = pandas.DataFrame(self.nmf.transform(s[self.input_col_names].fillna(0).values))
        self.output_col_names = result.columns.tolist()
        return result

    @check_dict_type
    def transform_single(self, s: dict_type):
        input_dataframe = pd.DataFrame([s])
        input_dataframe = input_dataframe[self.input_col_names]
        return self.transform(input_dataframe).to_dict("record")[0]

    def get_params(self) -> dict:
        params = PipeObject.get_params(self)
        params.update({"nmf": self.nmf, "n_components": self.n_components})
        return params

    def set_params(self, params: dict):
        PipeObject.set_params(self, params)
        self.nmf = params["nmf"]
        self.n_components = params["n_components"]
