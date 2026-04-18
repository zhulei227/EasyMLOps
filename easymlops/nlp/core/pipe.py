from easymlops.table.core import TablePipeObjectBase


class NLPPipeObjectBase(TablePipeObjectBase):
    """
    NLP Pipe基类。
    
    继承自TablePipeObjectBase，专门用于处理NLP任务。
    允许使用pandas DataFrame进行文本处理。
    
    Example:
        >>> class MyNLPPipe(NLPPipeObjectBase):
        ...     def udf_transform(self, s, **kwargs):
        ...         return s
    """
    
    def udf_get_params(self):
        """获取参数。"""
        return {}
