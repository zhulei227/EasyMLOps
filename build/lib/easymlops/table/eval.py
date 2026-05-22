from easymlops.table.core import *


class Eval(TablePipeObjectBase):
    def __init__(self, sql="all", **kwargs):
        """
        describe基本语法:
        (a+b)/c as col1,
        c//d as col2
        """
        super().__init__(**kwargs)
        self.sql = sql
        self.sql_details = []
        for item in self.sql.replace("\r\n", "").replace("\n", "").split(","):
            ql, output_col = item.split(" as ")
            self.sql_details.append((ql, output_col))

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        for ql, output_col in self.sql_details:
            s[output_col] = s.eval(ql)
        return s

    def udf_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        s_ = {}
        for k, v in s.items():
            s_[k] = [v]
        return self.udf_transform(pd.DataFrame(s_)).to_dict("records")[0]

    def udf_get_params(self):
        return {"sql": self.sql, "sql_details": self.sql_details}

    def udf_set_params(self, params: dict):
        self.sql = params["sql"]
        self.sql_details = params["sql_details"]
