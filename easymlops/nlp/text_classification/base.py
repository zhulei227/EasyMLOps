from easymlops.table.core import *
import pandas as pd
from easymlops.table.utils import PandasUtils


class ClassificationBase(TablePipeObjectBase):
    """
    dnn文本分类模型Base类
    """

    def __init__(self, y: series_type = None, col="all", skip_check_transform_type=True, drop_input_data=True,
                 native_init_params=None, native_fit_params=None,
                 **kwargs):
        """
        :param y:
        :param col: 用于模型训练的col,只支持对单个col进行文本分类
        :param skip_check_transform_type: 跳过类型检测
        :param drop_input_data: 删掉输入数据，默认True，不然输出为x1,x2,..,xn,y
        :param native_init_params: 底层分类模型的init入参，调用格式为BaseModel(**native_init_params)
        :param native_fit_params:底层分类模型的fit入参，调用格式为BaseModel.fit(x,y,**native_fit_params)
        :param kwargs:
        """
        super().__init__(skip_check_transform_type=skip_check_transform_type, **kwargs)
        self.col = col
        self.drop_input_data = drop_input_data
        self.y = copy.deepcopy(y)
        self.id2label = {}
        self.label2id = {}
        self.num_class = None
        if self.y is not None:
            for idx, label in enumerate(self.y.value_counts().index):
                self.id2label[idx] = label
                self.label2id[label] = idx
            self.y = self.y.apply(lambda x: self.label2id.get(x))
            self.num_class = len(self.id2label)
        # 底层模型自带参数
        self.native_init_params = copy.deepcopy(native_init_params)
        self.native_fit_params = copy.deepcopy(native_fit_params)
        if self.native_init_params is None:
            self.native_init_params = dict()
        if self.native_fit_params is None:
            self.native_fit_params = dict()

    def before_fit(self, s: dataframe_type, **kwargs) -> dataframe_type:
        s = super().before_fit(s, **kwargs)
        assert self.y is not None
        return s

    def before_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        s = super().before_transform(s, **kwargs)
        return s

    def transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        s_ = self.after_transform(self.udf_transform(self.before_transform(s, **kwargs), **kwargs), **kwargs)
        if not self.drop_input_data:
            s_ = PandasUtils.concat_duplicate_columns([s, s_])
        return s_

    def before_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        s = super().before_transform_single(s, **kwargs)
        return s

    def transform_single(self, s: dict_type, **kwargs) -> dict_type:
        s_ = copy.deepcopy(s)
        s_ = self.after_transform_single(
            self.udf_transform_single(self.before_transform_single(s_, **kwargs), **kwargs), **kwargs)
        if not self.drop_input_data:
            s.update(s_)
            return s
        else:
            return s_

    def udf_fit(self, s, **kwargs):
        return self

    def udf_transform(self, s, **kwargs):
        return s

    def udf_transform_single(self, s: dict_type, **kwargs):
        input_dataframe = pd.DataFrame([s])
        return self.udf_transform(input_dataframe, **kwargs).to_dict("record")[0]

    def udf_get_params(self) -> dict_type:
        return {"id2label": self.id2label, "label2id": self.label2id, "num_class": self.num_class, "col": self.col,
                "drop_input_data": self.drop_input_data}

    def udf_set_params(self, params: dict):
        self.id2label = params["id2label"]
        self.label2id = params["label2id"]
        self.num_class = params["num_class"]
        self.col = params["col"]
        self.drop_input_data = params["drop_input_data"]
