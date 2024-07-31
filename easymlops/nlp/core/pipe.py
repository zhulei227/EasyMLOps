from easymlops.table.core import TablePipeObjectBase


class NLPPipeObjectBase(TablePipeObjectBase):
    """
    由于NLP任务也可以存放于表格数据中，这里借用pandas dataframe做NLP任务
    """

    def udf_get_params(self):
        return {}
