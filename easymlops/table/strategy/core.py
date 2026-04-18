from easymlops.table.core import *
import pandas as pd
from easymlops.table.utils import PandasUtils
from .faster_lgbm_predictor_single import FasterLgbSinglePredictor


class LGBMRegression4Ruler(TablePipeObjectBase):

    def __init__(self, y: series_type = None, cols=None, predict_prefix="ruler", skip_check_transform_type=True,
                 drop_input_data=True, support_sparse_input=False,
                 native_init_params=None, native_fit_params=None, verbose=-1,
                 objective="regression", use_faster_predictor=True, dataset_params=None, **kwargs):
        """
         :param y:
        :param cols: 用于模型训练的cols
        :param predict_prefix: 模型输出的预测名称，prefix
        :param skip_check_transform_type: 跳过类型检测
        :param drop_input_data: 删掉输入数据，默认True，不然输出为x1,x2,..,xn
        :param support_sparse_input: 是否支持稀疏矩阵，如果输入数据中有稀疏数据，需要设置为True
        :param native_init_params: 底层分类模型的init入参，调用格式为BaseModel(**native_init_params)
        :param native_fit_params: 底层分类模型的fit入参，调用格式为BaseModel.fit(x,y,**native_fit_params)
        :param verbose: 默认-1，不显示训练日志
        :param objective: 默认regression
        :param use_faster_predictor: transform_single中是否使用预测加速
        :param dataset_params: 透传DataSet用,lgb.Dataset(data=xx, label=xx, feature_name=xx,**self.dataset_params)
        :param kwargs:
        """
        super().__init__(skip_check_transform_type=skip_check_transform_type, **kwargs)
        if cols is None or type(cols) == str:
            self.cols = []
        else:
            self.cols = cols
        self.drop_input_data = drop_input_data
        self.y = copy.deepcopy(y)
        self.predict_prefix = predict_prefix
        self.support_sparse_input = support_sparse_input
        # 底层模型自带参数
        self.native_init_params = copy.deepcopy(native_init_params)
        self.native_fit_params = copy.deepcopy(native_fit_params)
        if self.native_init_params is None or len(self.native_init_params) == 0:
            self.native_init_params = dict(max_depth=3)
        if self.native_fit_params is None or len(self.native_fit_params) == 0:
            self.native_fit_params = dict(num_boost_round=1)
        else:
            self.native_fit_params.update(dict(num_boost_round=1))

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

    def udf_fit(self, s: dataframe_type, **kwargs):
        import lightgbm as lgb
        if self.support_sparse_input:
            s_ = PandasUtils.pd2csr(s)
        else:
            s_ = s
        # if self.dataset_params.get("feature_name") is None:
        #     self.dataset_params["feature_name"] = self.cols
        self.lgb_model = lgb.train(params=self.native_init_params,
                                   train_set=lgb.Dataset(data=s_, label=self.y, **self.dataset_params),
                                   **self.native_fit_params)
        self.lgb_model_faster_predictor_params = self.lgb_model.dump_model()
        self.lgb_model_faster_predictor = FasterLgbSinglePredictor(model=self.lgb_model_faster_predictor_params,
                                                                   cache_num=10)
        return self

    def before_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        s = super().before_transform(s, **kwargs)
        if self.check_list_same(s.columns.tolist(), self.cols):
            return s
        else:
            return s[self.cols]

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        if self.support_sparse_input:
            s_ = PandasUtils.pd2csr(s)
        else:
            s_ = s
        result = pd.DataFrame(self.lgb_model.predict(s_),
                              columns=[self.predict_prefix + "_score"], index=s.index)
        result[self.predict_prefix + "_leaf_index"] = self.lgb_model.predict(s_, pred_leaf=True)
        result[self.predict_prefix + "_leaf_describe"] = result[self.predict_prefix + "_leaf_index"]. \
            apply(lambda x: self.simple_describe(self.lgb_model_faster_predictor.leaf_map_describe[0].get(x, "")))
        return result

    def transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        s_ = self.after_transform(self.udf_transform(self.before_transform(s, **kwargs), **kwargs), **kwargs)
        if not self.drop_input_data:
            s_ = PandasUtils.concat_duplicate_columns([s, s_])
        return s_

    def before_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        s = super().before_transform_single(s, **kwargs)
        return self.extract_dict(s, self.cols)

    def udf_transform_single(self, s: dict_type, **kwargs):
        if self.use_faster_predictor:
            faster_predict = self.lgb_model_faster_predictor.predict(s)
            return {self.predict_prefix + "_score": faster_predict.get("score"),
                    self.predict_prefix + "_leaf_index": faster_predict.get("leaf_index")[0],
                    self.predict_prefix + "_leaf_describe":
                        self.simple_describe(self.lgb_model_faster_predictor.leaf_map_describe[0].get(
                            faster_predict.get("leaf_index")[0], ""))}
        else:
            input_dataframe = pd.DataFrame([s])
            input_dataframe = input_dataframe[self.cols]
            return self.udf_transform(input_dataframe, **kwargs).to_dict("records")[0]

    def transform_single(self, s: dict_type, **kwargs) -> dict_type:
        s_ = copy.deepcopy(s)
        s_ = self.after_transform_single(
            self.udf_transform_single(self.before_transform_single(s_, **kwargs), **kwargs), **kwargs)
        if not self.drop_input_data:
            s.update(s_)
            return s
        else:
            return s_

    def udf_get_params(self) -> dict_type:
        params = {"predict_prefix": self.predict_prefix, "cols": self.cols,
                  "drop_input_data": self.drop_input_data, "support_sparse_input": self.support_sparse_input,
                  "lgb_model": self.lgb_model, "use_faster_predictor": self.use_faster_predictor}
        if self.use_faster_predictor:
            params["lgb_model_faster_predictor_params"] = self.lgb_model_faster_predictor_params
        return params

    def udf_set_params(self, params: dict_type):
        self.predict_prefix = params["predict_prefix"]
        self.cols = params["cols"]
        self.drop_input_data = params["drop_input_data"]
        self.support_sparse_input = params["support_sparse_input"]
        self.lgb_model = params["lgb_model"]
        self.use_faster_predictor = params["use_faster_predictor"]
        self.lgb_model_faster_predictor_params = params["lgb_model_faster_predictor_params"]
        self.lgb_model_faster_predictor = FasterLgbSinglePredictor(model=self.lgb_model_faster_predictor_params,
                                                                   cache_num=10)

    @staticmethod
    def simple_describe(describe):
        # 对describe进行简化
        if "|" in describe or "<" in describe or ">" in describe:
            describe_map = {}
            items = describe.replace("not <=", ">") \
                .replace("not >=", "<") \
                .replace("not <", ">=") \
                .replace("not >", "<=") \
                .replace(">=", ">") \
                .replace("<=", "<") \
                .replace(" ", "") \
                .split("|")
            for item in items:
                if "<" in item:
                    factor, thresh = item.split("<")
                    if factor not in describe_map:
                        describe_map[factor] = {}
                    thresh = round(float(thresh), 3)
                    if "<" not in describe_map[factor]:
                        describe_map[factor]["<"] = thresh
                    else:
                        describe_map[factor]["<"] = min(thresh, describe_map[factor]["<"])
                else:  # >
                    factor, thresh = item.split(">")
                    if factor not in describe_map:
                        describe_map[factor] = {}
                    thresh = round(float(thresh), 3)
                    if ">" not in describe_map[factor]:
                        describe_map[factor][">"] = thresh
                    else:
                        describe_map[factor][">"] = max(thresh, describe_map[factor][">"])
            # 重构describe
            new_describe = []
            for factor in describe_map.keys():
                arr = []
                if ">" in describe_map[factor]:
                    arr.append(str(describe_map[factor][">"]))
                    arr.append("<")
                arr.append(factor)
                if "<" in describe_map[factor]:
                    arr.append("<")
                    arr.append(str(describe_map[factor]["<"]))
                new_describe.append("".join(arr))
            return "|".join(new_describe)
        else:
            return describe
