from ..base import *


class Parallel(PipeObject):
    """
    并行模块:接受相同的数据，每个pipe object运行后合并(按col名覆盖)输出
    """

    def __init__(self, pipe_objects=None, name=None, drop_input_data=True):
        super().__init__(name=name,skip_check_transform_type=True)
        self.pipe_objects = pipe_objects
        self.drop_input_data = drop_input_data

    def _fit(self, s: dataframe_type):
        for obj in self.pipe_objects:
            s_ = copy.copy(s)  # 强制copy
            obj.fit(s_)
        return self

    def _transform(self, s: dataframe_type) -> dataframe_type:
        s_ = copy.copy(s)
        data = self.pipe_objects[0].transform(s_)
        for obj in self.pipe_objects[1:]:
            s_ = copy.copy(s)
            data_ = obj.transform(s_)
            for col in data_.columns:
                data[col] = data_[col]
        return data

    def _transform_single(self, s: dict_type) -> dict_type:
        s_ = copy.copy(s)
        data = self.pipe_objects[0].transform_single(s_)
        for obj in self.pipe_objects[1:]:
            s_ = copy.copy(s)
            data_ = obj.transform_single(s_)
            data.update(data_)
        return data

    def transform(self, s: dataframe_type) -> dataframe_type:
        output = self.after_transform(self._transform(self.before_transform(s)))
        if self.drop_input_data:
            return output
        else:
            for col in output.columns.tolist():
                s[col] = output[col]
            return s

    def transform_single(self, s: dict_type) -> dict_type:
        output = self.after_transform_single(self._transform_single(self.before_transform_single(s)))
        if self.drop_input_data:
            return output
        else:
            s.update(output)
            return s

    def _get_params(self):
        return {"pipe_objects_params": [obj.get_params() for obj in self.pipe_objects],
                "pipe_objects": self.pipe_objects, "drop_input_data": self.drop_input_data}

    def _set_params(self, params: dict_type):
        self.drop_input_data = params["drop_input_data"]
        self.pipe_objects = params["pipe_objects"]
        pipe_objects_params = params["pipe_objects_params"]
        for i in range(len(pipe_objects_params)):
            pipe_params = pipe_objects_params[i]
            obj = self.pipe_objects[i]
            obj.set_params(pipe_params)
