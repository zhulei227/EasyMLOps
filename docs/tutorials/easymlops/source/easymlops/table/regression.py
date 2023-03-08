from easymlops.table.core import *
import pandas as pd
from easymlops.table.utils import PandasUtils


class RegressionBase(TablePipeObjectBase):
    def __init__(self, y: series_type = None, cols="all", pred_name="pred", skip_check_transform_type=True,
                 drop_input_data=True, support_sparse_input=False,
                 native_init_params=None, native_fit_params=None,
                 **kwargs):
        super().__init__(skip_check_transform_type=skip_check_transform_type, **kwargs)
        if cols is None or type(cols) == str:
            self.cols = []
        self.drop_input_data = drop_input_data
        self.y = copy.deepcopy(y)
        self.pred_name = pred_name
        self.support_sparse_input = support_sparse_input
        # 底层模型自带参数
        self.native_init_params = native_init_params
        self.native_fit_params = native_fit_params
        if self.native_init_params is None:
            self.native_init_params = dict()
        if self.native_fit_params is None:
            self.native_fit_params = dict()

    def before_fit(self, s: dataframe_type, **kwargs) -> dataframe_type:
        s = super().before_fit(s, **kwargs)
        assert self.y is not None
        if len(self.cols) == 0:
            self.cols = s.columns.tolist()
        assert type(self.cols) == list
        if self.check_list_same(s.columns.tolist(), self.cols):
            return s
        else:
            return s[self.cols]

    def before_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        s = super().before_transform(s, **kwargs)
        if self.check_list_same(s.columns.tolist(), self.cols):
            return s
        else:
            return s[self.cols]

    def transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        s_ = self.after_transform(self.udf_transform(self.before_transform(s, **kwargs), **kwargs), **kwargs)
        if not self.drop_input_data:
            s_ = PandasUtils.concat_duplicate_columns([s, s_])
        return s_

    def before_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        s = super().before_transform_single(s, **kwargs)
        return self.extract_dict(s, self.cols)

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
        input_dataframe = input_dataframe[self.cols]
        return self.udf_transform(input_dataframe, **kwargs).to_dict("record")[0]

    def udf_get_params(self) -> dict_type:
        return {"pred_name": self.pred_name, "cols": self.cols,
                "drop_input_data": self.drop_input_data, "native_init_params": self.native_init_params,
                "native_fit_params": self.native_fit_params, "support_sparse_input": self.support_sparse_input}

    def udf_set_params(self, params: dict):
        self.pred_name = params["pred_name"]
        self.cols = params["cols"]
        self.drop_input_data = params["drop_input_data"]
        self.native_init_params = params["native_init_params"]
        self.native_fit_params = params["native_fit_params"]
        self.support_sparse_input = params["support_sparse_input"]


class LGBMRegression(RegressionBase):
    """
    input type:pandas.dataframe
    input like:
    | 0 | 1 | 2 | 3 |
    |0.2|0.3|0.4|0.5|
    |0.3|0.2|0.6|0.1|
    label type:pandas:series
    label like:
    |1|
    |0|
    |0|
    -------------------------
    output type:pandas.dataframe
    output like:
    |pred|
    |0.5 |
    |0.5 |
    """

    def __init__(self, y=None, support_sparse_input=False, verbose=-1, objective="regression",
                 use_faster_predictor=True, dataset_params=None, **kwargs):
        super().__init__(y=y, support_sparse_input=support_sparse_input, **kwargs)
        self.native_init_params.update({
            'objective': objective,
            'verbose': verbose
        })
        self.lgb_model = None
        self.use_faster_predictor = use_faster_predictor
        self.lgb_model_faster_predictor_params = None
        self.lgb_model_faster_predictor = None
        self.dataset_params = dataset_params
        if self.dataset_params is None:
            self.dataset_params = dict()

    def udf_fit(self, s: dataframe_type, **kwargs):
        import lightgbm as lgb
        if self.support_sparse_input:
            s_ = PandasUtils.pd2csr(s)
        else:
            s_ = s
        self.lgb_model = lgb.train(params=self.native_init_params,
                                   train_set=lgb.Dataset(data=s_, label=self.y, feature_name=list(s.columns),
                                                         **self.dataset_params),
                                   **self.native_fit_params)
        if self.use_faster_predictor:
            from easymlops.table.utils import FasterLgbSinglePredictor
            self.lgb_model_faster_predictor_params = self.lgb_model.dump_model()
            self.lgb_model_faster_predictor = FasterLgbSinglePredictor(
                model=self.lgb_model_faster_predictor_params, cache_num=10)
        return self

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        if self.support_sparse_input:
            s_ = PandasUtils.pd2csr(s)
        else:
            s_ = s
        result = pd.DataFrame(self.lgb_model.predict(s_), columns=[self.pred_name], index=s.index)
        return result

    def udf_transform_single(self, s: dict_type, **kwargs):
        if self.use_faster_predictor:
            return {self.pred_name: self.lgb_model_faster_predictor.predict(s).get("score")}
        else:
            input_dataframe = pd.DataFrame([s])
            input_dataframe = input_dataframe[self.cols]
            return self.udf_transform(input_dataframe, **kwargs).to_dict("record")[0]

    def udf_get_params(self) -> dict_type:
        params = {"lgb_model": self.lgb_model, "use_faster_predictor": self.use_faster_predictor}
        if self.use_faster_predictor:
            params["lgb_model_faster_predictor_params"] = self.lgb_model_faster_predictor_params
        return params

    def udf_set_params(self, params: dict_type):
        self.lgb_model = params["lgb_model"]
        self.use_faster_predictor = params["use_faster_predictor"]
        if self.use_faster_predictor:
            from easymlops.table.utils import FasterLgbSinglePredictor
            self.lgb_model_faster_predictor_params = params["lgb_model_faster_predictor_params"]
            self.lgb_model_faster_predictor = FasterLgbSinglePredictor(
                model=self.lgb_model_faster_predictor_params, cache_num=10)

    def get_contrib(self, s: dict_type) -> dict_type:
        """
        获取sabbas特征重要性
        """
        assert self.use_faster_predictor
        return self.lgb_model_faster_predictor.predict(s).get("contrib")
