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

    def __init__(self, cols=None, normal_range=100, normal_type="cdf", std_range=10, **kwargs):
        """
        :param normal_range:
        :param normal_type: cdf,range
        :param std_range:
        """
        super().__init__(**kwargs)
        self.normal_range = normal_range
        self.normal_type = normal_type
        self.std_range = std_range
        self.mean_std = dict()
        self.cols = cols

    def _fit(self, s: dataframe_type) -> dataframe_type:
        if str(self.cols).lower() in ["none", "all"]:
            self.cols = s.columns.tolist()
        if self.normal_type == "cdf":
            for col in self.cols:
                col_value = s[col]
                mean = np.median(col_value)
                std = np.std(col_value) * self.std_range
                self.mean_std[col] = (mean, std)
        return self

    def _transform(self, s: dataframe_type) -> dataframe_type:
        if self.normal_type == "cdf":
            for col in self.cols:
                if col in self.mean_std:
                    s[col] = np.round(
                        ss.norm.cdf((s[col] - self.mean_std[col][0]) / self.mean_std[col][1]) * self.normal_range, 2)
        elif self.normal_type == "range":
            for col in self.cols:
                if col in self.mean_std:
                    s[col] = self.normal_range * s[col]
        return s

    def _transform_single(self, s: dict_type) -> dict_type:
        if self.normal_type == "cdf":
            for col in self.cols:
                if col in self.mean_std:
                    s[col] = np.round(
                        ss.norm.cdf((s[col] - self.mean_std[col][0]) / self.mean_std[col][1]) * self.normal_range, 2)
        elif self.normal_type == "range":
            for col in self.cols:
                if col in self.mean_std:
                    s[col] = self.normal_range * s[col]
        return s

    def _get_params(self) -> dict:
        return {"mean_std": self.mean_std, "normal_range": self.normal_range, "normal_type": self.normal_type,
                "cols": self.cols}

    def _set_params(self, params: dict):
        self.mean_std = params["mean_std"]
        self.normal_range = params["normal_range"]
        self.normal_type = params["normal_type"]
        self.cols = params["cols"]


class MapValues(PipeObject):
    def __init__(self, map_values: dict = None, copy_transform_data=True, **kwargs):
        super().__init__(copy_transform_data=copy_transform_data, **kwargs)
        self.map_values = map_values if map_values is not None else dict()

    def _transform(self, s: dataframe_type) -> dataframe_type:
        for col in s.columns:
            if col in self.map_values:
                s[col] = np.round(s[col] / self.map_values[col][0] * self.map_values[col][1], 2)
        return s

    def _transform_single(self, s: dict_type) -> dict_type:
        for col in s.keys():
            if col in self.map_values:
                s[col] = np.round(s[col] / self.map_values[col][0] * self.map_values[col][1], 2)
        return s

    def _get_params(self) -> dict_type:
        return {"map_values": self.map_values}

    def _set_params(self, params: dict_type):
        self.map_values = params["map_values"]
