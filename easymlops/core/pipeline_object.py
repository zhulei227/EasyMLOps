from .pipe_object import *


class PipeLineBase(PipeObjectBase):
    """
    PipeLine设计为Pipe的子类，目的是为了方便嵌套使用;这个类名称虽为PipeLine的Base，但更多是起一个规范作用
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def auto_test(self, x):
        """
        测试功能

        :param x:
        :return:
        """
        raise Exception("need to implement!")

    def save(self, path):
        """
        保存模型

        :param path:
        :return:
        """
        raise Exception("need to implement!")

    def load(self, path):
        """
        加载模型

        :param path:
        :return:
        """
        raise Exception("need to implement!")

    def pipe(self, module):
        """
        加入pipe模块

        :param module:
        :return:
        """
        raise Exception("need to implement!")
