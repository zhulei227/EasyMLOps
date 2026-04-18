from easymlops.table.core import *


class SQL(TablePipeObjectBase):
    """
    评估Pipe。
    
    使用SQL表达式对数据进行计算评估。
    
    Example:
        >>> eval_pipe = Eval(sql="(a+b)/c as col1, c//d as col2")
    """
    
    def __init__(self, sql="all", **kwargs):
        """
        初始化评估Pipe。
        
        Args:
            sql: SQL表达式，多个表达式用逗号分隔，格式如 "(a+b)/c as col1, c//d as col2"
            **kwargs: 其他父类参数
        """
        super().__init__(**kwargs)
        self.sql = sql
        self.sql_details = []
        for item in self.sql.replace("\r\n", "").replace("\n", "").split(","):
            ql, output_col = item.split(" as ")
            self.sql_details.append((ql, output_col))

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        """
        批量评估计算。
        
        Args:
            s: 输入数据
            **kwargs: 其他参数
            
        Returns:
            计算后的DataFrame
        """
        for ql, output_col in self.sql_details:
            s[output_col] = s.eval(ql)
        return s

    def udf_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        """单条数据评估计算。"""
        s_ = {}
        for k, v in s.items():
            s_[k] = [v]
        return self.udf_transform(pd.DataFrame(s_)).to_dict("records")[0]

    def udf_get_params(self):
        """获取参数。"""
        return {"sql": self.sql, "sql_details": self.sql_details}

    def udf_set_params(self, params: dict):
        """设置参数。"""
        self.sql = params["sql"]
        self.sql_details = params["sql_details"]
