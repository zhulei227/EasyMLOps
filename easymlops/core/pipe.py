import warnings

warnings.filterwarnings("ignore")


class PipeObjectBase(object):
    """
    所有Pipe以及PipeLine的基类。
    
    提供了fit、transform、get_params等核心方法的标准调用流程。
    子类需要实现udf_fit、udf_transform等方法。
    
    Example:
        >>> class MyPipe(PipeObjectBase):
        ...     def udf_fit(self, x, **kwargs):
        ...         return self
        ...     def udf_transform(self, x, **kwargs):
        ...         return x
    """
    
    def __init__(self, name=None, **kwargs):
        """
        初始化Pipe对象。
        
        Args:
            name: 该pipe模块名称，默认使用self.__class__
            **kwargs: 其他参数
        """
        super().__init__(**kwargs)
        if name is None:
            name = self.__class__
        self.name = name

    def fit(self, x, **kwargs):
        """
        训练阶段的标准入口方法。
        
        依次调用: before_fit -> udf_fit -> after_fit
        
        Args:
            x: 输入数据
            **kwargs: 其他参数
            
        Returns:
            self
        """
        self.udf_fit(self.before_fit(x, **kwargs), **kwargs)
        return self.after_fit(x, **kwargs)

    def before_fit(self, x, **kwargs):
        """
        在udf_fit之前被调用，用于数据预处理。
        
        Args:
            x: 输入数据
            **kwargs: 其他参数
            
        Returns:
            处理后的数据
        """
        return x

    def udf_fit(self, x, **kwargs):
        """
        用户自定义的fit实现。
        
        Args:
            x: 输入数据
            **kwargs: 其他参数
            
        Returns:
            self
        """
        return self

    def after_fit(self, x=None, **kwargs):
        """
        在udf_fit之后被调用，用于后处理。
        
        Args:
            x: 输入数据
            **kwargs: 其他参数
            
        Returns:
            self
        """
        return self

    def transform(self, x, **kwargs):
        """
        批量数据转换的标准入口方法。
        
        依次调用: before_transform -> udf_transform -> after_transform
        
        Args:
            x: 输入数据
            **kwargs: 其他参数
            
        Returns:
            转换后的数据
        """
        return self.after_transform(self.udf_transform(self.before_transform(x, **kwargs), **kwargs), **kwargs)

    def before_transform(self, x, **kwargs):
        """
        在udf_transform之前被调用，用于数据预处理。
        
        Args:
            x: 输入数据
            **kwargs: 其他参数
            
        Returns:
            处理后的数据
        """
        return x

    def udf_transform(self, x, **kwargs):
        """
        用户自定义的transform实现。
        
        Args:
            x: 输入数据
            **kwargs: 其他参数
            
        Returns:
            转换后的数据
        """
        return x

    def after_transform(self, x, **kwargs):
        """
        在udf_transform之后被调用，用于后处理。
        
        Args:
            x: 输入数据
            **kwargs: 其他参数
            
        Returns:
            处理后的数据
        """
        return x

    def transform_single(self, x, **kwargs):
        """
        单条数据转换的标准入口方法。
        
        依次调用: before_transform_single -> udf_transform_single -> after_transform_single
        
        Args:
            x: 输入数据（单条）
            **kwargs: 其他参数
            
        Returns:
            转换后的数据
        """
        return self.after_transform_single(
            self.udf_transform_single(self.before_transform_single(x, **kwargs), **kwargs), **kwargs)

    def before_transform_single(self, x, **kwargs):
        """
        在udf_transform_single之前被调用。
        
        Args:
            x: 输入数据（单条）
            **kwargs: 其他参数
            
        Returns:
            处理后的数据
        """
        return x

    def udf_transform_single(self, x, **kwargs):
        """
        用户自定义的单条数据预测实现。
        
        Args:
            x: 输入数据（单条）
            **kwargs: 其他参数
            
        Returns:
            转换后的数据
        """
        return x

    def after_transform_single(self, x, **kwargs):
        """
        在udf_transform_single之后被调用。
        
        Args:
            x: 输入数据（单条）
            **kwargs: 其他参数
            
        Returns:
            处理后的数据
        """
        return x

    def get_params(self):
        """
        获取当前pipe需要被保存的所有参数。
        
        Returns:
            dict: 参数字典
        """
        return self.udf_get_params()

    def udf_get_params(self):
        """
        获取当前pipe需要保存的参数（不需要考虑父类）。
        
        Returns:
            dict: 参数字典
        """
        return {"name": self.name}

    def set_params(self, params):
        """
        恢复所有参数。
        
        Args:
            params: 参数字典
            
        Returns:
            self
        """
        self.udf_set_params(params)

    def udf_set_params(self, params):
        """
        恢复参数（不需要考虑父类）。
        
        Args:
            params: 参数字典
        """
        self.name = params["name"]

    def callback(self, callback_func, data, return_callback_result=False, *args, **kwargs):
        """
        回调函数接口。
        
        Args:
            callback_func: 回调函数
            data: 回调入参x
            return_callback_result: 是否返回回调函数结果
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            回调函数结果（如果return_callback_result为True）
        """
        result = callback_func(self, data, *args, **kwargs)
        if return_callback_result:
            return result

    def set_branch_pipe(self, pipe_obj):
        """
        挂载支线的pipe模块。
        
        不影响主线的执行任务（支线的输出结果不会并入到主线中），比如存储监控模块等。
        branch_pipe的fit/transform/transform_single运行均在当前挂载对象之后。
        
        Args:
            pipe_obj: 支线pipe对象
        """
        pass

    def get_branch_pipe(self, index):
        """
        获取对应的支线模块。
        
        注意branch pipe可以有多个，所以通过index索引。
        
        Args:
            index: 支线索引
            
        Returns:
            支线pipe对象
        """
        raise Exception("need to implement!")

    def remove_branch_pipe(self, index):
        """
        移除指定的branch pipe。
        
        Args:
            index: 支线索引
        """
        raise Exception("need to implement!")

    def set_master_pipe(self, master_pipe):
        """
        设置当前pipe作为branch时所绑定的master pipe。
        
        Args:
            master_pipe: 主线pipe对象
        """
        raise Exception("need to implement!")

    def get_master_pipe(self):
        """
        获取当前pipe作为branch时所绑定的master pipe。
        
        Returns:
            主线pipe对象
        """
        raise Exception("need to implement!")

    def set_parent_pipe(self, parent_pipe=None):
        """
        设置父类pipe模块。
        
        有时候需要在内部回溯之前的transform操作。
        
        Args:
            parent_pipe: 父pipe对象
        """
        raise Exception("need to implement!")

    def get_parent_pipe(self):
        """
        获取父类pipe模块。
        
        Returns:
            父pipe对象
        """
        raise Exception("need to implement!")

    def get_all_parent_pipes(self):
        """
        获取所有父类（从头到尾的顺序）。
        
        Returns:
            父pipe对象列表
        """
        raise Exception("need to implement!")

    def transform_all_parent(self, x, **kwargs):
        """
        顺序运行当前pipe之前父类的transform。
        
        Args:
            x: 输入数据
            **kwargs: 其他参数
            
        Returns:
            转换后的数据
        """
        raise Exception("need to implement!")

    def transform_single_all_parent(self, x, **kwargs):
        """
        顺序运行当前pipe之前父类的transform_single。
        
        Args:
            x: 输入数据（单条）
            **kwargs: 其他参数
            
        Returns:
            转换后的数据
        """
        raise Exception("need to implement!")
