from easymlops.table.core import TablePipeLine


class NLPPipeline(TablePipeLine):
    """
    借用PipeTable做相关任务
    """

    def udf_get_params(self):
        return {}
