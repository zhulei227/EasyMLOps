from easymlops.table.core import *


class PCADecomposition(TablePipeObjectBase):

    def __init__(self, n_components=3, native_init_params=None, native_fit_params=None,
                 **kwargs):
        """
        :param n_components: 保留的pca主成分数量
        :param native_init_params: sklearn.decomposition.PCA的初始化参数
        :param native_fit_params: sklearn.decomposition.PCA.fit除X以外的参数
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.pca = None
        self.native_init_params = native_init_params
        self.native_fit_params = native_fit_params
        if self.native_init_params is None:
            self.native_init_params = dict()
        if self.native_fit_params is None:
            self.native_fit_params = dict()
        self.native_init_params["n_components"] = n_components
        self.col_types = dict()

    def udf_fit(self, s: dataframe_type, **kwargs):
        from sklearn.decomposition import PCA
        self.pca = PCA(**self.native_init_params)
        # 记录数据类型
        for col in s.columns.tolist():
            self.col_types[col] = self.get_col_type(s[col])
        s = s.fillna(0)
        self.pca.fit(X=s, **self.native_fit_params)
        return self

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        # 转换数据类型
        for col, col_type in self.col_types.items():
            s[col] = s[col].astype(col_type)
        s = s.fillna(0)
        result = pd.DataFrame(
            self.pca.transform(np.clip(s.fillna(0).values, np.finfo(np.float64).min, np.finfo(np.float64).max)),
            index=s.index)
        return result

    def transform_single(self, s: dict_type, **kwargs):
        input_dataframe = pd.DataFrame([s])
        return self.transform(input_dataframe, **kwargs).to_dict("record")[0]

    def udf_get_params(self) -> dict_type:
        params = {"pca": self.pca, "native_init_params": self.native_init_params,
                  "native_fit_params": self.native_fit_params, "col_types": self.col_types}
        return params

    def udf_set_params(self, params: dict):
        self.col_types = params["col_types"]
        self.pca = params["pca"]
        self.native_init_params = params["native_init_params"]
        self.native_fit_params = params["native_fit_params"]


class NMFDecomposition(TablePipeObjectBase):

    def __init__(self, n_components=3, native_init_params=None, native_fit_params=None, **kwargs):
        """
        :param n_components: 保留的nmf主成分数量
        :param native_init_params: sklearn.decomposition.NMF的初始化参数
        :param native_fit_params: sklearn.decomposition.NMF.fit除X以外的参数
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.nmf = None
        self.native_init_params = native_init_params
        self.native_fit_params = native_fit_params
        if self.native_init_params is None:
            self.native_init_params = dict()
        if self.native_fit_params is None:
            self.native_fit_params = dict()
        self.native_init_params["n_components"] = n_components
        self.col_types = dict()

    def udf_fit(self, s: dataframe_type, **kwargs):
        from sklearn.decomposition import NMF
        # 记录数据类型
        for col in s.columns.tolist():
            self.col_types[col] = self.get_col_type(s[col])
        self.nmf = NMF(**self.native_init_params)
        # 将小于0的值，设置为0
        values = s.fillna(0).values
        values = np.where(values < 0, 0, values)
        self.nmf.fit(values, **self.native_fit_params)
        return self

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        # 转换数据类型
        for col, col_type in self.col_types.items():
            s[col] = s[col].astype(col_type)
        # 将小于0的值，设置为0
        values = s.fillna(0).values
        values = np.where(values < 0, 0, values)
        result = pd.DataFrame(self.nmf.transform(values), index=s.index)
        return result

    def transform_single(self, s: dict_type, **kwargs):
        input_dataframe = pd.DataFrame([s])
        return self.transform(input_dataframe, **kwargs).to_dict("record")[0]

    def udf_get_params(self) -> dict_type:
        return {"nmf": self.nmf, "native_init_params": self.native_init_params,
                "native_fit_params": self.native_fit_params, "col_types": self.col_types}

    def udf_set_params(self, params: dict_type):
        self.col_types = params["col_types"]
        self.nmf = params["nmf"]
        self.native_init_params = params["native_init_params"]
        self.native_fit_params = params["native_fit_params"]
