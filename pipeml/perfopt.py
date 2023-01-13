from .base import *
import scipy.sparse as sp

"""
性能优化模块
"""


class ReduceMemUsage(PipeObject):
    """
    功能性：减少内存使用量
    """

    def __init__(self, name=None, copy_transform_data=False):
        PipeObject.__init__(self, name=name, copy_transform_data=copy_transform_data)
        self.type_map_detail = dict()
        self.input_col_names = None
        self.output_col_names = None

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

    def fit(self, s: dataframe_type) -> dataframe_type:
        self.input_col_names = s.columns.tolist()
        for col in self.input_col_names:
            self.type_map_detail[col] = self.get_type(s[col])
        return self

    def transform(self, s: dataframe_type) -> dataframe_type:
        if self.copy_transform_data:
            s_ = s[self.input_col_names]
        else:
            s = s[self.input_col_names]
            s_ = s
        for col, ti in self.type_map_detail.items():
            tp, ran = ti
            try:
                if ran is not None:
                    s_[col] = tp(np.clip(s_[col], ran[0], ran[1]))
                else:
                    s_[col] = s_[col].astype(str)
            except:
                pass
        self.output_col_names = s_.columns.tolist()
        return s_

    def transform_single(self, s: dict_type) -> dict_type:
        s_ = self.extract_dict(s, self.input_col_names)
        for col, ti in self.type_map_detail.items():
            tp, ran = ti
            try:
                if ran is not None:
                    s_[col] = tp(np.clip(s_[col], ran[0], ran[1]))
                else:
                    s_[col] = str(s_[col])
            except:
                pass
        return self.extract_dict(s_, self.output_col_names)

    def get_params(self) -> dict_type:
        params = PipeObject.get_params(self)
        params.update({"type_map_detail": self.type_map_detail, "input_col_names": self.input_col_names,
                       "output_col_names": self.output_col_names})
        return params

    def set_params(self, params: dict):
        PipeObject.set_params(self, params)
        self.type_map_detail = params["type_map_detail"]
        self.input_col_names = params["input_col_names"]
        self.output_col_names = params["output_col_names"]


class Dense2Sparse(ReduceMemUsage):
    """
    功能性：稠密矩阵转稀疏矩阵
    """

    def __init__(self, name=None, copy_transform_data=False):
        ReduceMemUsage.__init__(self, name=name, copy_transform_data=copy_transform_data)

    def transform(self, s: dataframe_type) -> dataframe_type:
        if self.copy_transform_data:
            s_ = s[self.input_col_names]
        else:
            s = s[self.input_col_names]
            s_ = s
        s_ = pd.DataFrame.sparse.from_spmatrix(data=sp.csr_matrix(s_), columns=self.input_col_names)
        for col, ti in self.type_map_detail.items():
            tp, ran = ti
            try:
                if ran is not None:
                    s_[col] = s_[col].astype(tp)
                else:
                    s_[col] = s_[col].astype(str)
            except:
                pass
        self.output_col_names = s_.columns.tolist()
        return s_
