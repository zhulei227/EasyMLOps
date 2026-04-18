from easymlops.table.core import TablePipeObjectBase
import pandas as pd
import numpy as np

dataframe_type = pd.DataFrame
dict_type = dict


class OCRPipeObjectBase(TablePipeObjectBase):
    """
    OCR Pipe基类。
    
    继承自TablePipeObjectBase，专门用于处理OCR相关任务。
    支持文本检测、文本识别等任务。
    
    Example:
        >>> class MyOCRPipe(OCRPipeObjectBase):
        ...     def udf_transform(self, s, **kwargs):
        ...         return s
    """
    
    def __init__(self, model_name=None, model_path=None, lang="en",
                 gpu=True, batch_size=1, **kwargs):
        """
        初始化OCR Pipe对象。
        
        Args:
            model_name: 模型名称
            model_path: 模型文件路径，如果指定则优先使用
            lang: 语言代码，如 'en', 'ch_sim', 'ja' 等，支持多语言如 ['en', 'ch_sim']
            gpu: 是否使用GPU加速
            batch_size: 批处理大小
            **kwargs: 其他父类参数
        """
        super().__init__(**kwargs)
        self.model_name = model_name
        self.model_path = model_path
        self.lang = lang
        self.gpu = gpu
        self.batch_size = batch_size
        self.model = None

    def udf_get_params(self):
        """获取参数。"""
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "lang": self.lang,
            "gpu": self.gpu,
            "batch_size": self.batch_size
        }

    def udf_set_params(self, params: dict_type):
        """设置参数。"""
        self.model_name = params.get("model_name", self.model_name)
        self.model_path = params.get("model_path", self.model_path)
        self.lang = params.get("lang", self.lang)
        self.gpu = params.get("gpu", self.gpu)
        self.batch_size = params.get("batch_size", self.batch_size)
