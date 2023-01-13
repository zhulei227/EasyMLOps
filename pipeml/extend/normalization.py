"""
自定义包
"""
from ..base import *
import numpy as np
import scipy.stats as ss


class Normalization(PipeObject):
    """
    input type:pandas.dataframe
    input like:
    | 0 | 1 | 2 | 3 |
    |0.2|0.3|0.4|0.5|
    |0.3|0.2|0.6|0.1|
    -------------------------
    output type:pandas.dataframe
    | 0 | 1 | 2 | 3 |
    | 45| 56| 45| 45|
    | 55| 34| 56| 23|
    """

    def __init__(self, cols=None, name=None, normal_range=100, normal_type="cdf", std_range=10,
                 copy_transform_data=True):
        """
        :param normal_range:
        :param normal_type: cdf,range
        :param std_range:
        """
        PipeObject.__init__(self, name=name, copy_transform_data=copy_transform_data)
        self.normal_range = normal_range
        self.normal_type = normal_type
        self.std_range = std_range
        self.mean_std = dict()
        self.cols = cols

    @check_dataframe_type
    def fit(self, s: dataframe_type) -> dataframe_type:
        self.input_col_names = s.columns.tolist()
        if str(self.cols).lower() in ["none", "all"]:
            self.cols = s.columns.tolist()
        if self.normal_type == "cdf":
            for col in self.cols:
                col_value = s[col]
                mean = np.median(col_value)
                std = np.std(col_value) * self.std_range
                self.mean_std[col] = (mean, std)
        return self

    def transform(self, s: dataframe_type) -> dataframe_type:
        if self.copy_transform_data:
            s_ = s[self.input_col_names]
        else:
            s = s[self.input_col_names]
            s_ = s
        if self.normal_type == "cdf":
            for col in self.cols:
                if col in self.mean_std:
                    s_[col] = np.round(
                        ss.norm.cdf((s_[col] - self.mean_std[col][0]) / self.mean_std[col][1]) * self.normal_range, 2)
        elif self.normal_type == "range":
            for col in self.cols:
                if col in self.mean_std:
                    s_[col] = self.normal_range * s_[col]
        self.output_col_names = s_.columns.tolist()
        return s_

    def transform_single(self, s: dict_type) -> dict_type:
        s_ = copy.copy(s)
        if self.normal_type == "cdf":
            for col in self.cols:
                if col in self.mean_std:
                    s_[col] = np.round(
                        ss.norm.cdf((s_[col] - self.mean_std[col][0]) / self.mean_std[col][1]) * self.normal_range, 2)
        elif self.normal_type == "range":
            for col in self.cols:
                if col in self.mean_std:
                    s_[col] = self.normal_range * s_[col]
        return s_

    def get_params(self) -> dict:
        params = PipeObject.get_params(self)
        params.update({"mean_std": self.mean_std, "normal_range": self.normal_range, "normal_type": self.normal_type,
                       "cols": self.cols})
        return params

    def set_params(self, params: dict):
        PipeObject.set_params(self, params)
        self.mean_std = params["mean_std"]
        self.normal_range = params["normal_range"]
        self.normal_type = params["normal_type"]
        self.cols = params["cols"]


class MapValues(PipeObject):
    def __init__(self, name=None, map_values: dict = None, copy_transform_data=True):
        PipeObject.__init__(self, name=name, copy_transform_data=copy_transform_data)
        self.map_values = map_values if map_values is not None else dict()

    def fit(self, s):
        self.input_col_names = s.columns.tolist()

    def transform(self, s: dataframe_type) -> dataframe_type:
        if self.copy_transform_data:
            s_ = s[self.input_col_names]
        else:
            s = s[self.input_col_names]
            s_ = s
        for col in s_.columns:
            if col in self.map_values:
                s_[col] = np.round(s_[col] / self.map_values[col][0] * self.map_values[col][1], 2)
        self.output_col_names = s_.columns.tolist()
        return s_

    def transform_single(self, s: dict_type) -> dict_type:
        s_ = copy.copy(s)
        for col in s_.keys():
            if col in self.map_values:
                s_[col] = np.round(s_[col] / self.map_values[col][0] * self.map_values[col][1], 2)
        return s_

    def get_params(self) -> dict_type:
        params = PipeObject.get_params(self)
        params.update({"map_values": self.map_values})
        return params

    def set_params(self, params: dict_type):
        PipeObject.set_params(self, params)
        self.map_values = params["map_values"]
