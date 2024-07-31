import warnings

warnings.filterwarnings("ignore")


class PipeObjectBase(object):
    """
     所有Pipe以及PipeLine的父类
    """

    def __init__(self, name=None, **kwargs):
        """
        :param name: 该pipe模块名称，默认self.__class__
        :param kwargs:
        """
        super().__init__(**kwargs)
        if name is None:
            name = self.__class__
        self.name = name

    def fit(self, x, **kwargs):
        """
        fit:依次调用before_fit,_fit,after_fit

        :param x:
        :param kwargs:
        :return:
        """
        self.udf_fit(self.before_fit(x, **kwargs), **kwargs)
        return self.after_fit(x, **kwargs)

    def before_fit(self, x, **kwargs):
        """
        :param x: 入参x

        :param kwargs:
        :return:
        """
        return x

    def udf_fit(self, x, **kwargs):
        """
        用户自己实现的fit模块

        :param x: 输入x
        :param kwargs:
        :return:
        """
        return self

    def after_fit(self, x=None, **kwargs):
        """
        在udf_fit之后被调用

        :param x: 入参x
        :param kwargs:
        :return:
        """
        return self

    def transform(self, x, **kwargs):
        """
        用于批量数据的transform:依次调用before_transform,udf_transform,after_transform

        :param x:
        :param kwargs:
        :return:
        """
        return self.after_transform(self.udf_transform(self.before_transform(x, **kwargs), **kwargs), **kwargs)

    def before_transform(self, x, **kwargs):
        """
        在udf_transform之前被调用，并将return的结果，作为udf_transform的x

        :param x:
        :param kwargs:
        :return:
        """
        return x

    def udf_transform(self, x, **kwargs):
        """
        用户自定义transform

        :param x:
        :param kwargs:
        :return:
        """
        return x

    def after_transform(self, x, **kwargs):
        """
        在udf_transform之后被调用

        :param x:
        :param kwargs:
        :return:
        """
        return x

    def transform_single(self, x, **kwargs):
        """
        用于单条数据的transform函数:调用顺序before_transform_single,_transform_single,after_transform_single

        :param x:
        :param kwargs:
        :return:
        """
        return self.after_transform_single(
            self.udf_transform_single(self.before_transform_single(x, **kwargs), **kwargs), **kwargs)

    def before_transform_single(self, x, **kwargs):
        """
        在udf_transform_single之前被调用

        :param x:
        :param kwargs:
        :return:
        """
        return x

    def udf_transform_single(self, x, **kwargs):
        """
        用户自定义单条数据预测实现

        :param x:
        :param kwargs:
        :return:
        """
        return x

    def after_transform_single(self, x, **kwargs):
        """
        在udf_transform_single之后被调用

        :param x:
        :param kwargs:
        :return:
        """
        return x

    def get_params(self):
        """
        return 当前pipe需要被保存的所有参数

        :return:
        """
        return self.udf_get_params()

    def udf_get_params(self):
        """
        return 前pipe需要被保存的参数（不需要考虑父类）

        :return:
        """
        return {"name": self.name}

    def set_params(self, params):
        """
        恢复所有参数

        :param params:
        :return:
        """
        self.udf_set_params(params)

    def udf_set_params(self, params):
        """
        恢复参数（不需要考虑父类）  \n

        :param params:
        :return:
        """
        self.name = params["name"]

    def callback(self, callback_func, data, return_callback_result=False, *args, **kwargs):
        """
        回调函数接口  \n

        :param callback_func: 回调函数
        :param data: 回调入参x
        :param return_callback_result: 是否返回回调函数结果
        :param args:
        :param kwargs:
        :return:
        """
        result = callback_func(self, data, *args, **kwargs)
        if return_callback_result:
            return result

    def set_branch_pipe(self, pipe_obj):
        """
        挂载支线的pipe模块，不影响主线的执行任务(支线的输出结果不会并入到主线中)，比如存储监控模块等; \n
        branch_pipe的fit/transform/transform_single运行均在当前挂载对象之后  \n

        :param pipe_obj:
        :return:
        """
        pass

    def get_branch_pipe(self, index):
        """
        获取到对应的模块，注意branch pipe可以有多个，所以通过index索引 \n

        :param index:
        :return:
        """
        raise Exception("need to implement!")

    def remove_branch_pipe(self, index):
        """
        移除指定的branch pipe

        :param index:
        :return:
        """
        raise Exception("need to implement!")

    def set_master_pipe(self, master_pipe):
        """
        当前pipe作为branch时所绑定的master的pipe

        :param master_pipe:
        :return:
        """
        raise Exception("need to implement!")

    def get_master_pipe(self):
        """
        当前pipe作为branch时所绑定的master的pipe

        :return:
        """
        raise Exception("need to implement!")

    def set_parent_pipe(self, parent_pipe=None):
        """
        设置父类pipe模块，有时候需要在内部回溯之前的transform操作

        :param parent_pipe:
        :return:
        """
        raise Exception("need to implement!")

    def get_parent_pipe(self):
        """
        返回父类pipe模块

        :return:
        """
        raise Exception("need to implement!")

    def get_all_parent_pipes(self):
        """
        返回所有父类（从头到尾的顺序）

        :return:
        """
        raise Exception("need to implement!")

    def transform_all_parent(self, x, **kwargs):
        """
        顺序运行当前pipe之前父类的transform

        :param x:
        :param kwargs:
        :return:
        """
        raise Exception("need to implement!")

    def transform_single_all_parent(self, x, **kwargs):
        """
        顺序运行当前pipe之前父类的transform_single

        :param x:
        :param kwargs:
        :return:
        """
        raise Exception("need to implement!")
