import warnings

warnings.filterwarnings("ignore")


class PipeObjectBase(object):
    """
     name:模块名称，如果为空默认为self.__class__
     """

    def __init__(self, name=None, **kwargs):
        super().__init__(**kwargs)
        if name is None:
            name = self.__class__
        self.name = name

    def fit(self, x, **kwargs):
        """
        fit:依次调用before_fit,_fit,after_fit
        """
        self.udf_fit(self.before_fit(x, **kwargs), **kwargs)
        return self.after_fit(x, **kwargs)

    def before_fit(self, x, **kwargs):
        """
        fit前预操作
        """
        return x

    def udf_fit(self, x, **kwargs):
        return self

    def after_fit(self, x=None, **kwargs):
        """
        fit后操作
        """
        return self

    def transform(self, x, **kwargs):
        """
        批量接口:依次调用before_transform,udf_transform,after_transform
        """
        return self.after_transform(self.udf_transform(self.before_transform(x, **kwargs), **kwargs), **kwargs)

    def before_transform(self, x, **kwargs):
        return x

    def udf_transform(self, x, **kwargs):
        return x

    def after_transform(self, x, **kwargs):
        return x

    def transform_single(self, x, **kwargs):
        """
        当条数据接口:调用顺序before_transform_single,_transform_single,after_transform_single
        """
        return self.after_transform_single(
            self.udf_transform_single(self.before_transform_single(x, **kwargs), **kwargs), **kwargs)

    def before_transform_single(self, x, **kwargs):
        return x

    def udf_transform_single(self, x, **kwargs):
        """
        当条数据接口，生产用
        """
        return x

    def after_transform_single(self, x, **kwargs):
        return x

    def get_params(self):
        return self.udf_get_params()

    def udf_get_params(self):
        return {"name": self.name}

    def set_params(self, params):
        self.udf_set_params(params)

    def udf_set_params(self, params):
        self.name = params["name"]

    def callback(self, callback_func, data, return_callback_result=False, *args, **kwargs):
        """
        回调函数接口
        """
        result = callback_func(self, data, *args, **kwargs)
        if return_callback_result:
            return result

    def set_branch_pipe(self, pipe_obj):
        """
        挂载支线的pipe模块，不影响主线的执行任务(支线的输出结果不会并入到主线中)，比如存储监控模块等;
        branch_pipe的fit/transform/transform_single运行均在当前挂载对象之后
        """
        pass

    def get_branch_pipe(self, index):
        """
        获取到对应的模块，注意branch pipe可以有多个，所以通过index索引
        """
        raise Exception("need to implement!")

    def remove_branch_pipe(self, index):
        """
        移除指定的branch pipe
        """
        raise Exception("need to implement!")

    def set_master_pipe(self, master_pipe):
        """
        当前pipe作为branch时所绑定的master的pipe
        """
        raise Exception("need to implement!")

    def get_master_pipe(self):
        """
        当前pipe作为branch时所绑定的master的pipe
        """
        raise Exception("need to implement!")

    def set_parent_pipe(self, parent_pipe=None):
        """
        设置父类pipe模块，有时候需要在内部回溯之前的transform操作
        """
        raise Exception("need to implement!")

    def get_parent_pipe(self):
        """
        返回父类pipe模块
        """
        raise Exception("need to implement!")

    def get_all_parent_pipes(self):
        """
        返回所有父类
        """
        raise Exception("need to implement!")

    def transform_all_parent(self, x, **kwargs):
        """
        顺序运行当前pipe之前父类的transform
        """
        raise Exception("need to implement!")

    def transform_single_all_parent(self, x, **kwargs):
        """
        顺序运行当前pipe之前父类的transform_single
        """
        raise Exception("need to implement!")
