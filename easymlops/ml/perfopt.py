from ..base import *
import scipy.sparse as sp

"""
性能优化模块
"""


class ReduceMemUsage(PipeObject):
    """
    功能性：减少内存使用量
    """

    def __init__(self, name=None, copy_transform_data=False):
        super().__init__(name=name, copy_transform_data=copy_transform_data)
        self.type_map_detail = dict()

    @staticmethod
    def extract_dict(s: dict_type, keys: list):
        new_s = dict()
        for key in keys:
            new_s[key] = s[key]
        return new_s

    @staticmethod
    def get_type(ser: series_type):
        col_type = ser.dtype
        if col_type != object:
            c_min = ser.min()
            c_max = ser.max()
            if "int" in str(col_type) or "bool" in str(col_type):
                int_types = [np.uint8, np.int8, np.uint16, np.int16, np.uint32, np.int32, np.uint64, np.int64]
                for int_type in int_types:
                    if c_min > np.iinfo(int_type).min and c_max < np.iinfo(int_type).max:
                        return int_type, (np.iinfo(int_type).min, np.iinfo(int_type).max)
            elif "float" in str(col_type) or "double" in str(col_type):
                float_types = [np.float16, np.float32, np.float64]
                for float_type in float_types:
                    if c_min > np.finfo(float_type).min and c_max < np.finfo(float_type).max:
                        return float_type, (np.finfo(float_type).min, np.finfo(float_type).max)
            else:
                return str, None
        else:
            return str, None

    @fit_wrapper
    def fit(self, s: dataframe_type) -> dataframe_type:
        for col in self.input_col_names:
            self.type_map_detail[col] = self.get_type(s[col])
        return self

    @transform_wrapper
    def transform(self, s: dataframe_type) -> dataframe_type:
        for col, ti in self.type_map_detail.items():
            tp, ran = ti
            try:
                if ran is not None:
                    s[col] = tp(np.clip(s[col], ran[0], ran[1]))
                else:
                    s[col] = s[col].astype(str)
            except:
                pass
        return s

    @transform_single_wrapper
    def transform_single(self, s: dict_type) -> dict_type:
        for col, ti in self.type_map_detail.items():
            tp, ran = ti
            try:
                if ran is not None:
                    s[col] = tp(np.clip(s[col], ran[0], ran[1]))
                else:
                    s[col] = str(s[col])
            except:
                pass
        return s

    def _get_params(self) -> dict_type:
        return {"type_map_detail": self.type_map_detail}

    def _set_params(self, params: dict):
        self.type_map_detail = params["type_map_detail"]


class Dense2Sparse(ReduceMemUsage):
    """
    功能性：稠密矩阵转稀疏矩阵
    """

    def __init__(self, name=None, copy_transform_data=False):
        super().__init__(name=name, copy_transform_data=copy_transform_data)

    @transform_wrapper
    def transform(self, s: dataframe_type) -> dataframe_type:
        s = pd.DataFrame.sparse.from_spmatrix(data=sp.csr_matrix(s), columns=self.input_col_names)
        for col, ti in self.type_map_detail.items():
            tp, ran = ti
            try:
                if ran is not None:
                    s[col] = s[col].astype(tp)
                else:
                    s[col] = s[col].astype(str)
            except:
                pass
        return s

    def _get_params(self) -> dict_type:
        return {}