from ..base import *
from .preprocessing import Clip
import pandas as pd


class PCADecomposition(PipeObject):
    """
    n_components:保留的pca主成分数量
    native_init_params:sklearn.decomposition.PCA的初始化参数
    native_fit_params:sklearn.decomposition.PCA.fit除X以外的参数
    ----------------------------
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

    def __init__(self, n_components=3, native_init_params=None, native_fit_params=None, clip=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.pca = None
        self.native_init_params = native_init_params
        self.native_fit_params = native_fit_params
        self.clip = clip
        self.clip_model = None
        if self.native_init_params is None:
            self.native_init_params = dict()
        if self.native_fit_params is None:
            self.native_fit_params = dict()
        self.native_init_params["n_components"] = n_components

    def _fit(self, s: dataframe_type) -> dataframe_type:
        from sklearn.decomposition import PCA
        self.pca = PCA(**self.native_init_params)
        s = s.fillna(0)
        if self.clip:
            self.clip_model = Clip(percent_range=(1, 99))
            s = self.clip_model.fit(s).transform(s)
        self.pca.fit(X=s, **self.native_fit_params)
        return self

    def _transform(self, s: dataframe_type) -> dataframe_type:
        s = s.fillna(0)
        if self.clip:
            s = self.clip_model.transform(s)
        result = pandas.DataFrame(
            self.pca.transform(np.clip(s.fillna(0).values, np.finfo(np.float64).min, np.finfo(np.float64).max)))
        return result

    def _transform_single(self, s: dict_type):
        input_dataframe = pd.DataFrame([s])
        return self._transform(input_dataframe).to_dict("record")[0]

    def _get_params(self) -> dict_type:
        params = {"pca": self.pca, "native_init_params": self.native_init_params,
                  "native_fit_params": self.native_fit_params, "clip": self.clip}
        if self.clip:
            params["clip_model"] = self.clip_model.get_params()
        return params

    def _set_params(self, params: dict):
        self.pca = params["pca"]
        self.native_init_params = params["native_init_params"]
        self.native_fit_params = params["native_fit_params"]
        self.clip = params["clip"]
        if self.clip:
            self.clip_model = Clip()
            self.clip_model.set_params(params["clip_model"])
        else:
            self.clip_model = None


class NMFDecomposition(PipeObject):
    """
    n_components:保留的nmf主成分数量
    native_init_params:sklearn.decomposition.NMF的初始化参数
    native_fit_params:sklearn.decomposition.NMF.fit除X以外的参数
    ----------------------------
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

    def __init__(self, n_components=3, native_init_params=None, native_fit_params=None, **kwargs):
        super().__init__(**kwargs)
        self.nmf = None
        self.native_init_params = native_init_params
        self.native_fit_params = native_fit_params
        if self.native_init_params is None:
            self.native_init_params = dict()
        if self.native_fit_params is None:
            self.native_fit_params = dict()
        self.native_init_params["n_components"] = n_components

    def _fit(self, s: dataframe_type) -> dataframe_type:
        from sklearn.decomposition import NMF
        self.nmf = NMF(**self.native_init_params)
        self.nmf.fit(s.fillna(0).values, **self.native_fit_params)
        return self

    def _transform(self, s: dataframe_type) -> dataframe_type:
        result = pandas.DataFrame(self.nmf.transform(s.fillna(0).values))
        return result

    def _transform_single(self, s: dict_type):
        input_dataframe = pd.DataFrame([s])
        return self._transform(input_dataframe).to_dict("record")[0]

    def _get_params(self) -> dict_type:
        return {"nmf": self.nmf, "native_init_params": self.native_init_params,
                "native_fit_params": self.native_fit_params}

    def _set_params(self, params: dict_type):
        self.nmf = params["nmf"]
        self.native_init_params = params["native_init_params"]
        self.native_fit_params = params["native_fit_params"]
