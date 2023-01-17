from ..base import *
from .preprocessing import FillNa
import pandas as pd


class UserDefinedRepresentationObject(PipeObject):
    """
    只需要对单个key-value处理的类型
    """

    def __init__(self, cols="all", y=None, name=None, transform_check_max_number_error=1e-5,
                 skip_check_transform_type=False, skip_check_transform_value=False, copy_transform_data=True):
        super().__init__(name=name, transform_check_max_number_error=transform_check_max_number_error,
                         skip_check_transform_type=skip_check_transform_type,
                         skip_check_transform_value=skip_check_transform_value,
                         copy_transform_data=copy_transform_data)
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

    def before_fit(self, s: dataframe_type) -> dataframe_type:
        s = super().before_fit(s)
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
        return s

    def _transform(self, s: dataframe_type) -> dataframe_type:
        for col, new_col in self.cols:
            if col in s.columns:
                s[new_col] = s[col].apply(lambda x: self._user_defined_function(col, x))
        return s

    def _transform_single(self, s: dict_type) -> dict_type:
        for col, new_col in self.cols:
            if col in s.keys():
                s[new_col] = self._user_defined_function(col, s[col])
        return s

    def _get_params(self) -> dict_type:
        return {"cols": self.cols}

    def _set_params(self, params: dict):
        self.cols = params["cols"]


class TargetEncoding(UserDefinedRepresentationObject):
    """
      input type:pandas.dataframe
      input like:
      | 0 | 1 |
      |good|A|
      |bad|B|
      |good|A|
      |bad|B|

      y like:
      |y|
      |1|
      |0|
      |0|
      |1|
      ----------------------------
      output type:pandas.dataframe
      output like:
      | 0 | 1 |
      |0.5|0.5|
      |0.5|0.5|
      |0.5|0.5|
      |0.5|0.5|
      """
    def __init__(self, cols="all", y=None, name=None, error_value=0,
                 transform_check_max_number_error=1e-3, copy_transform_data=True):
        super().__init__(cols=cols, name=name, y=y,
                         transform_check_max_number_error=transform_check_max_number_error,
                         skip_check_transform_type=True,
                         copy_transform_data=copy_transform_data)
        self.error_value = error_value
        self.target_map_detail = dict()

    def _fit(self, s: dataframe_type):
        assert self.y is not None and len(self.y) == len(s)
        s["y_"] = self.y
        for col, _ in self.cols:
            tmp_ = s[[col, "y_"]]
            col_map = list(tmp_.groupby([col]).agg({"y_": ["mean"]}).to_dict().values())[0]
            self.target_map_detail[col] = col_map
        del s["y_"]
        return self

    def _user_defined_function(self, col, x):
        map_detail_ = self.target_map_detail.get(col, dict())
        return map_detail_.get(x, self.error_value)

    def _get_params(self) -> dict_type:
        return {"target_map_detail": self.target_map_detail, "error_value": self.error_value}

    def _set_params(self, params: dict_type):
        self.target_map_detail = params["target_map_detail"]
        self.error_value = params["error_value"]


class LabelEncoding(UserDefinedRepresentationObject):
    """
      input type:pandas.dataframe
      input like:
      | 0 | 1 |
      |good|A|
      |bad|B|
      |good|A|
      |bad|B|
      ----------------------------
      output type:pandas.dataframe
      output like:
      | 0 | 1 |
      |1|0|
      |0|1|
      |1|0|
      |0|1|
      """
    def __init__(self, cols="all", name=None, error_value=0,
                 transform_check_max_number_error=1e-3, copy_transform_data=True):
        super().__init__(cols=cols, name=name,
                         transform_check_max_number_error=transform_check_max_number_error,
                         skip_check_transform_type=True,
                         copy_transform_data=copy_transform_data)
        self.error_value = error_value
        self.label_map_detail = dict()

    def _fit(self, s: dataframe_type):
        for col, _ in self.cols:
            col_map = s[col].value_counts().to_dict()
            c = 1
            for key, value in col_map.items():
                col_map[key] = c
                c += 1
            self.label_map_detail[col] = col_map
        return self

    def _user_defined_function(self, col, x):
        map_detail_ = self.label_map_detail.get(col, dict())
        return map_detail_.get(x, self.error_value)

    def _get_params(self) -> dict_type:
        return {"label_map_detail": self.label_map_detail, "error_value": self.error_value}

    def _set_params(self, params: dict):
        self.label_map_detail = params["label_map_detail"]
        self.error_value = params["error_value"]


class OneHotEncoding(UserDefinedRepresentationObject):
    """
         input type:pandas.dataframe
         input like:
         | x | y |
         |good|A|
         |bad|B|
         |good|A|
         |bad|B|
         ----------------------------
         output type:pandas.dataframe
         output like:
         | x_good | x_bad | y_A | y_B |
         |   1    |   0   |  1  |  0  |
         |   0    |   1   |  0  |  1  |
         |   1    |   0   |  1  |  0  |
         |   0    |   1   |  0  |  1  |
         """
    def __init__(self, cols="all", name=None, drop_col=True,
                 transform_check_max_number_error=1e-3, copy_transform_data=True):
        super().__init__(cols=cols, name=name,
                         transform_check_max_number_error=transform_check_max_number_error,
                         skip_check_transform_type=True,
                         copy_transform_data=copy_transform_data)
        self.drop_col = drop_col
        self.one_hot_detail = dict()
        self.fill_na_model = None

    def _fit(self, s: dataframe_type):
        self.fill_na_model = FillNa(cols=[col for col, _ in self.cols], fill_category_value="nan", fill_number_value=0)
        s = self.fill_na_model.fit(s).transform(s)
        for col, _ in self.cols:
            self.one_hot_detail[col] = s[col].unique().tolist()
        return self

    def _transform(self, s: dataframe_type) -> dataframe_type:
        s = self.fill_na_model.transform(s)
        for col, new_col in self.cols:
            if col not in self.one_hot_detail.keys():
                raise Exception("{} not in {}".format(col, self.one_hot_detail.keys()))
            values = self.one_hot_detail.get(col)
            for value in values:
                s["{}_{}".format(new_col, value)] = (s[col] == value).astype(int)
        if self.drop_col:
            for col in self.one_hot_detail.keys():
                del s[col]
        return s

    def _transform_single(self, s: dict_type) -> dict_type:
        s = self.fill_na_model.transform_single(s)
        for col, new_col in self.cols:
            if col not in self.one_hot_detail.keys():
                raise Exception("{} not in {}".format(col, self.one_hot_detail.keys()))
            values = self.one_hot_detail.get(col)
            for value in values:
                if s[col] == value:
                    s["{}_{}".format(new_col, value)] = 1
                else:
                    s["{}_{}".format(new_col, value)] = 0
        if self.drop_col:
            for col in self.one_hot_detail.keys():
                del s[col]
        return s

    def _get_params(self) -> dict_type:
        return {"fill_na_model": self.fill_na_model.get_params(), "one_hot_detail": self.one_hot_detail,
                "drop_col": self.drop_col}

    def _set_params(self, params: dict):
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
                 skip_check_transform_type=True, copy_transform_data=True, prefix=None):
        super().__init__(name=name, transform_check_max_number_error=transform_check_max_number_error,
                         skip_check_transform_type=skip_check_transform_type,
                         copy_transform_data=copy_transform_data, prefix=prefix)
        self.n_components = n_components
        self.pca = None

    def _fit(self, s: dataframe_type) -> dataframe_type:
        from sklearn.decomposition import PCA
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(s.fillna(0).values)
        return self

    def _transform(self, s: dataframe_type) -> dataframe_type:
        result = pandas.DataFrame(self.pca.transform(s.fillna(0).values))
        return result

    def _transform_single(self, s: dict_type):
        input_dataframe = pd.DataFrame([s])
        return self._transform(input_dataframe).to_dict("record")[0]

    def _get_params(self) -> dict_type:
        return {"pca": self.pca, "n_components": self.n_components}

    def _set_params(self, params: dict):
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
                 skip_check_transform_type=True, copy_transform_data=True, prefix=None):
        super().__init__(name=name, transform_check_max_number_error=transform_check_max_number_error,
                         skip_check_transform_type=skip_check_transform_type,
                         copy_transform_data=copy_transform_data, prefix=prefix)
        self.n_components = n_components
        self.nmf = None

    def _fit(self, s: dataframe_type) -> dataframe_type:
        from sklearn.decomposition import NMF
        self.nmf = NMF(n_components=self.n_components)
        self.nmf.fit(s.fillna(0).values)
        return self

    def _transform(self, s: dataframe_type) -> dataframe_type:
        result = pandas.DataFrame(self.nmf.transform(s.fillna(0).values))
        return result

    def _transform_single(self, s: dict_type):
        input_dataframe = pd.DataFrame([s])
        return self._transform(input_dataframe).to_dict("record")[0]

    def _get_params(self) -> dict_type:
        return {"nmf": self.nmf, "n_components": self.n_components}

    def _set_params(self, params: dict_type):
        self.nmf = params["nmf"]
        self.n_components = params["n_components"]
