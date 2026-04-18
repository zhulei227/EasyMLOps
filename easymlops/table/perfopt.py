from easymlops.table.core import *
import scipy.sparse as sp


class ReduceMemUsage(TablePipeObjectBase):
    """
    内存优化Pipe。
    
    通过修改数据类型减少内存使用量。
    自动检测并转换整数和浮点数的最小数据类型。
    
    Example:
        >>> pipe = ReduceMemUsage()
    """
    
    def __init__(self, skip_check_transform_type=True, **kwargs):
        """
        初始化内存优化Pipe。
        
        Args:
            skip_check_transform_type: 跳过类型检测
            **kwargs: 其他父类参数
        """
        super().__init__(skip_check_transform_type=skip_check_transform_type, **kwargs)
        self.type_map_detail = dict()

    @staticmethod
    def extract_dict(s: dict_type, keys: list):
        """从字典中提取指定键。"""
        new_s = dict()
        for key in keys:
            new_s[key] = s[key]
        return new_s

    @staticmethod
    def get_type(ser: series_type):
        """
        获取列的最优数据类型。
        
        Args:
            ser: 数据Series
            
        Returns:
            (数据类型, (最小值, 最大值))
        """
        col_type = ser.dtype
        col_type_str = str(col_type).lower()
        if col_type != object:
            c_min = ser.min()
            c_max = ser.max()
            if "int" in col_type_str or "bool" in col_type_str:
                int_types = [np.uint8, np.int8, np.uint16, np.int16, np.uint32, np.int32, np.uint64, np.int64]
                for int_type in int_types:
                    if c_min > np.iinfo(int_type).min and c_max < np.iinfo(int_type).max:
                        return int_type, (np.iinfo(int_type).min, np.iinfo(int_type).max)
                return np.int64, (np.iinfo(np.int64).min, np.iinfo(np.int64).max)
            elif "float" in col_type_str or "double" in col_type_str:
                float_types = [np.float16, np.float32, np.float64]
                for float_type in float_types:
                    if c_min > np.finfo(float_type).min and c_max < np.finfo(float_type).max:
                        return float_type, (np.finfo(float_type).min, np.finfo(float_type).max)
                return np.float64, (np.finfo(np.float64).min, np.finfo(np.float64).max)
            else:
                return str, None
        else:
            return str, None

    def udf_fit(self, s: dataframe_type, **kwargs):
        """
        训练阶段，确定每列的最优数据类型。
        
        Args:
            s: 输入数据
            **kwargs: 其他参数
            
        Returns:
            self
        """
        for col in self.input_col_names:
            self.type_map_detail[col] = self.get_type(s[col])
        return self

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        """批量数据类型转换。"""
        for col, ti in self.type_map_detail.items():
            tp, ran = ti
            try:
                if ran is not None:
                    s[col] = tp(np.clip(s[col], ran[0], ran[1]))
                else:
                    s[col] = s[col].astype(str)
            except Exception as e:
                print(e)
        return s

    def udf_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        """单条数据类型转换。"""
        for col, ti in self.type_map_detail.items():
            tp, ran = ti
            try:
                if ran is not None:
                    s[col] = tp(np.clip(s[col], ran[0], ran[1]))
                else:
                    s[col] = str(s[col])
            except Exception as e:
                print(e)
        return s

    def udf_get_params(self) -> dict_type:
        """获取参数。"""
        return {"type_map_detail": self.type_map_detail}

    def udf_set_params(self, params: dict):
        """设置参数。"""
        self.type_map_detail = params["type_map_detail"]


class Dense2Sparse(ReduceMemUsage):
    """
    稠密转稀疏矩阵Pipe。
    
    通过将稠密矩阵压缩为稀疏矩阵减少内存使用。
    适用于0值较多的数据。
    注意：后续pipe object需要支持csr结构的稀疏矩阵。
    """
    
    def __init__(self, **kwargs):
        """初始化稠密转稀疏Pipe。"""
        super().__init__(**kwargs)

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        """转换为稀疏矩阵。"""
        if not hasattr(s, "sparse"):
            s = pd.DataFrame.sparse.from_spmatrix(data=sp.csr_matrix(s), columns=self.input_col_names, index=s.index)
        for col, ti in self.type_map_detail.items():
            tp, ran = ti
            try:
                if ran is not None:
                    s[col] = s[col].astype(tp)
                else:
                    s[col] = s[col].astype(str)
            except Exception as e:
                print(e)
        return s

    def udf_get_params(self) -> dict_type:
        """获取参数。"""
        return {}
