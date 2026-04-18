from easymlops.table.core import *
from easymlops.table.utils import PandasUtils


class Parallel(TablePipeObjectBase):
    """
    并行模块。
    
    接受相同的数据，并行执行多个pipe object，然后将结果按列合并输出。
    列名相同的后续结果会覆盖前面的结果。
    
    Example:
        >>> parallel = Parallel(pipe_objects=[pipe1, pipe2, pipe3])
    """
    
    def __init__(self, pipe_objects=None, drop_input_data=True, skip_check_transform_type=True, **kwargs):
        """
        初始化并行模块。
        
        Args:
            pipe_objects: 要并行执行的pipe对象列表
            drop_input_data: 是否删除输入数据
            skip_check_transform_type: 跳过类型检测
            **kwargs: 其他父类参数
        """
        super().__init__(skip_check_transform_type=skip_check_transform_type, **kwargs)
        self.pipe_objects = pipe_objects
        for obj in self.pipe_objects:
            obj.set_parent_pipe(self.get_parent_pipe())
        self.drop_input_data = drop_input_data

    def __getitem__(self, target_pipe_model):
        """通过索引或名称获取子pipe对象。"""
        return self._match_index_pipe(match_index=target_pipe_model)

    def _match_index_pipe(self, match_index) -> TablePipeObjectBase:
        """
        根据索引匹配pipe对象。
        
        Args:
            match_index: 索引，可以是int、tuple、list或str
            
        Returns:
            匹配的pipe对象
        """
        if type(match_index) == int:
            return self.pipe_objects[match_index]
        if type(match_index) == tuple:
            match_index = list(match_index)
        if type(match_index) == list:
            pipes_ = []
            pipes_.extend(self.pipe_objects)
            current_pipe = None
            while len(match_index) > 0:
                index = match_index.pop(0)
                current_pipe = pipes_[index]
                pipes_ = []
                if issubclass(current_pipe.__class__, TablePipeLine):
                    pipes_.extend(current_pipe.models)
                if issubclass(current_pipe.__class__, easymlops.table.ensemble.Parallel):
                    pipes_.extend(current_pipe.pipe_objects)
            return current_pipe
        if type(match_index) == str:
            pipes_ = []
            pipes_.extend(self.pipe_objects)
            while len(pipes_) > 0:
                ipipe = pipes_.pop(0)
                if ipipe.name == match_index:
                    return ipipe
                if issubclass(ipipe.__class__, easymlops.table.ensemble.Parallel):
                    pipes_.extend(ipipe.pipe_objects)
                if issubclass(ipipe.__class__, TablePipeLine):
                    pipes_.extend(ipipe.models)
        raise Exception(f"can't match the index:{match_index}")

    def udf_fit(self, s: dataframe_type, **kwargs):
        """并行训练所有子pipe对象。"""
        for obj in self.pipe_objects:
            s_ = copy.deepcopy(s)
            obj.fit(s_, **kwargs)
        return self

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        """并行转换数据并合并结果。"""
        dfs = []
        for index in range(len(self.pipe_objects)):
            s_ = copy.deepcopy(s)
            df = self.pipe_objects[index].transform(s_, **kwargs)
            df.index = s.index
            dfs.append(df)
        return PandasUtils.concat_duplicate_columns(dfs)

    def udf_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        """单条数据并行转换。"""
        result = dict()
        for index in range(len(self.pipe_objects)):
            s_ = copy.deepcopy(s)
            result.update(self.pipe_objects[index].transform_single(s_, **kwargs))
        return result

    def transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        """批量数据转换。"""
        output = self.after_transform(self.udf_transform(self.before_transform(s, **kwargs), **kwargs), **kwargs)
        if self.drop_input_data:
            return output
        else:
            return PandasUtils.concat_duplicate_columns([s, output])

    def transform_single(self, s: dict_type, **kwargs) -> dict_type:
        """单条数据转换。"""
        output = self.after_transform_single(
            self.udf_transform_single(self.before_transform_single(s, **kwargs), **kwargs), **kwargs)
        if self.drop_input_data:
            return output
        else:
            s.update(output)
            return s

    def udf_get_params(self):
        """获取参数。"""
        return {"pipe_objects_params": [obj.get_params() for obj in self.pipe_objects],
                "drop_input_data": self.drop_input_data}

    def udf_set_params(self, params: dict_type):
        """设置参数。"""
        self.drop_input_data = params["drop_input_data"]
        pipe_objects_params = params["pipe_objects_params"]
        for i in range(len(pipe_objects_params)):
            pipe_params = pipe_objects_params[i]
            obj = self.pipe_objects[i]
            obj.set_params(pipe_params)
