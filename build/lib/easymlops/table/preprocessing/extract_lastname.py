from easymlops.table.core import TablePipeObjectBase


class ExtractLastName(TablePipeObjectBase):
    def __init__(self, name_col="Name", new_col="LastName", split_char=",", **kwargs):
        super().__init__(**kwargs)
        self.name_col = name_col
        self.new_col = new_col
        self.split_char = split_char

    def udf_get_params(self):
        return {"name_col": self.name_col, "new_col": self.new_col, "split_char": self.split_char}

    def udf_set_params(self, params):
        self.name_col = params["name_col"]
        self.new_col = params["new_col"]
        self.split_char = params["split_char"]

    def udf_fit(self, s, **kwargs):
        return self

    def udf_transform(self, s, **kwargs):
        s[self.new_col] = s[self.name_col].str.split(self.split_char).str[0]
        return s

    def udf_transform_single(self, s, **kwargs):
        s[self.new_col] = s[self.name_col].split(self.split_char)[0]
        return s
