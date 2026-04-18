from easymlops.table.core import *
from sklearn import decomposition
from sklearn import manifold


class Decomposition(TablePipeObjectBase):

    def __init__(self, n_components=3, decomposition_class=None, native_init_params=None, native_fit_params=None,
                 **kwargs):
        """
        :param n_components: 保留的主成分数量
        :param decomposition_class:降维类别
        :param native_init_params: sklearn.decomposition.PCA的初始化参数
        :param native_fit_params: sklearn.decomposition.PCA.fit除X以外的参数
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.decomposition_object = None
        self.decomposition_class = decomposition_class
        self.native_init_params = native_init_params
        self.native_fit_params = native_fit_params
        if self.native_init_params is None:
            self.native_init_params = dict()
        if self.native_fit_params is None:
            self.native_fit_params = dict()
        self.native_init_params["n_components"] = n_components
        self.col_types = dict()

    def udf_fit(self, s: dataframe_type, **kwargs):
        self.decomposition_object = self.decomposition_class(**self.native_init_params)
        # 记录数据类型
        for col in s.columns.tolist():
            self.col_types[col] = self.get_col_type(s[col])
        s = s.fillna(0)
        self.decomposition_object.fit(X=s, **self.native_fit_params)
        return self

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        # 转换数据类型
        for col, col_type in self.col_types.items():
            s[col] = s[col].astype(col_type)
        s = s.fillna(0)
        result = pd.DataFrame(
            self.decomposition_object.transform(
                np.clip(s.fillna(0).values, np.finfo(np.float64).min, np.finfo(np.float64).max)),
            index=s.index)
        return result

    def transform_single(self, s: dict_type, **kwargs):
        input_dataframe = pd.DataFrame([s])
        return self.transform(input_dataframe, **kwargs).to_dict("records")[0]

    def udf_get_params(self) -> dict_type:
        params = {"decomposition_object": self.decomposition_object,
                  "decomposition_class": self.decomposition_class,
                  "native_init_params": self.native_init_params,
                  "native_fit_params": self.native_fit_params, "col_types": self.col_types}
        return params

    def udf_set_params(self, params: dict):
        self.col_types = params["col_types"]
        self.decomposition_object = params["decomposition_object"]
        self.decomposition_class = params["decomposition_class"]
        self.native_init_params = params["native_init_params"]
        self.native_fit_params = params["native_fit_params"]


class PCADecomposition(Decomposition):
    def __init__(self, n_components=3,
                 decomposition_class=decomposition.PCA,
                 native_init_params=None,
                 native_fit_params=None,
                 **kwargs):
        super().__init__(n_components=n_components,
                         decomposition_class=decomposition_class,
                         native_init_params=native_init_params,
                         native_fit_params=native_fit_params, **kwargs)

    def udf_get_params(self):
        return {}


class NMFDecomposition(Decomposition):
    def __init__(self, n_components=3,
                 decomposition_class=decomposition.NMF,
                 native_init_params=None,
                 native_fit_params=None,
                 **kwargs):
        super().__init__(n_components=n_components,
                         decomposition_class=decomposition_class,
                         native_init_params=native_init_params,
                         native_fit_params=native_fit_params, **kwargs)

    def udf_get_params(self):
        return {}


class KernelPCADecomposition(Decomposition):
    def __init__(self, n_components=3,
                 kernel="rbf",
                 decomposition_class=decomposition.KernelPCA,
                 native_init_params=None,
                 native_fit_params=None,
                 **kwargs):
        super().__init__(n_components=n_components,
                         decomposition_class=decomposition_class,
                         native_init_params=native_init_params,
                         native_fit_params=native_fit_params, **kwargs)
        self.native_init_params["kernel"] = kernel

    def udf_get_params(self):
        return {}


class FastICADecomposition(Decomposition):
    def __init__(self, n_components=3,
                 decomposition_class=decomposition.FastICA,
                 native_init_params=None,
                 native_fit_params=None,
                 **kwargs):
        super().__init__(n_components=n_components,
                         decomposition_class=decomposition_class,
                         native_init_params=native_init_params,
                         native_fit_params=native_fit_params, **kwargs)

    def udf_get_params(self):
        return {}


class DictionaryLearningDecomposition(Decomposition):
    def __init__(self, n_components=3,
                 decomposition_class=decomposition.DictionaryLearning,
                 native_init_params=None,
                 native_fit_params=None,
                 **kwargs):
        super().__init__(n_components=n_components,
                         decomposition_class=decomposition_class,
                         native_init_params=native_init_params,
                         native_fit_params=native_fit_params, **kwargs)

    def udf_get_params(self):
        return {}


class MiniBatchDictionaryLearningDecomposition(Decomposition):
    def __init__(self, n_components=3,
                 decomposition_class=decomposition.MiniBatchDictionaryLearning,
                 native_init_params=None,
                 native_fit_params=None,
                 **kwargs):
        super().__init__(n_components=n_components,
                         decomposition_class=decomposition_class,
                         native_init_params=native_init_params,
                         native_fit_params=native_fit_params, **kwargs)

    def udf_get_params(self):
        return {}


class LDADecomposition(Decomposition):
    def __init__(self, n_components=3,
                 decomposition_class=decomposition.LatentDirichletAllocation,
                 native_init_params=None,
                 native_fit_params=None,
                 **kwargs):
        super().__init__(n_components=n_components,
                         decomposition_class=decomposition_class,
                         native_init_params=native_init_params,
                         native_fit_params=native_fit_params, **kwargs)

    def udf_get_params(self):
        return {}


class TSNEDecomposition(Decomposition):
    def __init__(self, n_components=3,
                 decomposition_class=manifold.TSNE,
                 native_init_params=None,
                 native_fit_params=None,
                 **kwargs):
        super().__init__(n_components=n_components,
                         decomposition_class=decomposition_class,
                         native_init_params=native_init_params,
                         native_fit_params=native_fit_params, **kwargs)

    def udf_get_params(self):
        return {}


class MDSDecomposition(Decomposition):
    def __init__(self, n_components=3,
                 decomposition_class=manifold.MDS,
                 native_init_params=None,
                 native_fit_params=None,
                 **kwargs):
        super().__init__(n_components=n_components,
                         decomposition_class=decomposition_class,
                         native_init_params=native_init_params,
                         native_fit_params=native_fit_params, **kwargs)

    def udf_get_params(self):
        return {}


class IsomapDecomposition(Decomposition):
    def __init__(self, n_components=3,
                 decomposition_class=manifold.Isomap,
                 native_init_params=None,
                 native_fit_params=None,
                 **kwargs):
        super().__init__(n_components=n_components,
                         decomposition_class=decomposition_class,
                         native_init_params=native_init_params,
                         native_fit_params=native_fit_params, **kwargs)

    def udf_get_params(self):
        return {}


class SpectralEmbeddingDecomposition(Decomposition):
    def __init__(self, n_components=3,
                 decomposition_class=manifold.SpectralEmbedding,
                 native_init_params=None,
                 native_fit_params=None,
                 **kwargs):
        super().__init__(n_components=n_components,
                         decomposition_class=decomposition_class,
                         native_init_params=native_init_params,
                         native_fit_params=native_fit_params, **kwargs)

    def udf_get_params(self):
        return {}


class LocallyLinearEmbeddingDecomposition(Decomposition):
    def __init__(self, n_components=3,
                 decomposition_class=manifold.LocallyLinearEmbedding,
                 native_init_params=None,
                 native_fit_params=None,
                 **kwargs):
        super().__init__(n_components=n_components,
                         decomposition_class=decomposition_class,
                         native_init_params=native_init_params,
                         native_fit_params=native_fit_params, **kwargs)

    def udf_get_params(self):
        return {}
