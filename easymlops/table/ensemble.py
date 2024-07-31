from easymlops.table.core import *
from easymlops.table.utils import PandasUtils


class Parallel(TablePipeObjectBase):
    """
    并行模块:接受相同的数据，每个pipe object运行后合并(按col名覆盖)输出
    """

    def __init__(self, pipe_objects=None, drop_input_data=True, skip_check_transform_type=True, **kwargs):
        """

        :param pipe_objects:
        :param drop_input_data:
        :param skip_check_transform_type:
        :param kwargs:
        """
        super().__init__(skip_check_transform_type=skip_check_transform_type, **kwargs)
        self.pipe_objects = pipe_objects
        for obj in self.pipe_objects:
            # 设置parent
            obj.set_parent_pipe(self.get_parent_pipe())
        self.drop_input_data = drop_input_data

    def __getitem__(self, target_pipe_model):
        return self._match_index_pipe(match_index=target_pipe_model)

    def _match_index_pipe(self, match_index) -> TablePipeObjectBase:
        if type(match_index) == int:
            return self.pipe_objects[match_index]
        if type(match_index) == tuple:
            match_index = list(match_index)
        if type(match_index) == list:
            # 逐层获取
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
        for obj in self.pipe_objects:
            s_ = copy.deepcopy(s)  # 强制copy
            obj.fit(s_, **kwargs)
        return self

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        dfs = []
        for index in range(len(self.pipe_objects)):
            s_ = copy.deepcopy(s)
            df = self.pipe_objects[index].transform(s_, **kwargs)
            df.index = s.index
            dfs.append(df)
        return PandasUtils.concat_duplicate_columns(dfs)

    def udf_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        result = dict()
        for index in range(len(self.pipe_objects)):
            s_ = copy.deepcopy(s)
            result.update(self.pipe_objects[index].transform_single(s_, **kwargs))
        return result

    def transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        output = self.after_transform(self.udf_transform(self.before_transform(s, **kwargs), **kwargs), **kwargs)
        if self.drop_input_data:
            return output
        else:
            return PandasUtils.concat_duplicate_columns([s, output])

    def transform_single(self, s: dict_type, **kwargs) -> dict_type:
        output = self.after_transform_single(
            self.udf_transform_single(self.before_transform_single(s, **kwargs), **kwargs), **kwargs)
        if self.drop_input_data:
            return output
        else:
            s.update(output)
            return s

    def udf_get_params(self):
        return {"pipe_objects_params": [obj.get_params() for obj in self.pipe_objects],
                "drop_input_data": self.drop_input_data}

    def udf_set_params(self, params: dict_type):
        self.drop_input_data = params["drop_input_data"]
        pipe_objects_params = params["pipe_objects_params"]
        for i in range(len(pipe_objects_params)):
            pipe_params = pipe_objects_params[i]
            obj = self.pipe_objects[i]
            obj.set_params(pipe_params)
