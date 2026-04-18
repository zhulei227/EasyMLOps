from easymlops.table.core import *
from easymlops.table.encoding import TargetEncoding
from easymlops.table.utils import PandasUtils


class LGBMRegressionLayers(TablePipeObjectBase):
    def __init__(self, y: series_type = None, pred_name="pred", cols_layer=None,
                 dataset_params_layer=None,
                 embedding_target_encoding_params_layer=None,
                 skip_check_transform_type=True,
                 drop_input_data=True, support_sparse_input=False,
                 native_init_params=None, native_fit_params=None,
                 use_faster_predictor=True,
                 verbose=-1, objective="regression",
                 **kwargs):
        """
        :param y:
        :param pred_name:预测名称
        :param cols_layer: 用于模型训练的cols_layer
        :param dataset_params_layer: 每层模型训练时的dataset_params
        :embedding_target_encoding_params_layer:嵌入target encoding
        :param skip_check_transform_type: 跳过类型检测
        :param drop_input_data: 删掉输入数据，默认True，不然输出为x1,x2,..,xn,one_var_ruler1,one_var_ruler2....
        :param support_sparse_input: 是否支持稀疏矩阵，如果输入数据中有稀疏数据，需要设置为True
        :param native_init_params: 底层分类模型的init入参，调用格式为BaseModel(**native_init_params)
        :param native_fit_params: 底层分类模型的fit入参，调用格式为BaseModel.fit(x,y,**native_fit_params)
        :param verbose: 默认-1，不显示训练日志
        :param objective: 默认regression
        :param use_faster_predictor: transform_single中是否使用预测加速
        :param kwargs:
        """
        super().__init__(skip_check_transform_type=skip_check_transform_type, **kwargs)
        self.drop_input_data = drop_input_data
        self.y = copy.deepcopy(y)
        self.pred_name = pred_name
        self.support_sparse_input = support_sparse_input
        # 底层模型自带参数
        self.native_init_params = copy.deepcopy(native_init_params)
        if self.native_init_params is None or len(self.native_init_params) == 0:
            self.native_init_params = dict(max_depth=3)
        self.objective = objective
        self.native_init_params.update({
            'objective': objective,
            'verbose': verbose
        })
        self.native_fit_params = copy.deepcopy(native_fit_params)
        if self.native_fit_params is None or len(self.native_fit_params) == 0:
            self.native_fit_params = dict()

        self.cols_layer = cols_layer
        self.dataset_params_layer = dataset_params_layer
        self.embedding_target_encoding_params_layer = embedding_target_encoding_params_layer
        self.use_faster_predictor = use_faster_predictor
        self.target_encoding_models = []
        self.lgb_regression_models = []

    def udf_fit(self, s: dataframe_type, **kwargs):
        from easymlops.table.regression import LGBMRegression
        # 1.定义残差
        if self.objective in ["poisson", "tweedie"]:
            s["diff__"] = np.ones_like(self.y.values)
        else:
            s["diff__"] = np.zeros_like(self.y.values)
        # 2.逐步训练
        for idx, cols in enumerate(self.cols_layer):
            if self.dataset_params_layer is not None:
                dataset_params = self.dataset_params_layer[idx]
            else:
                dataset_params = {}
            if self.objective in ["poisson", "tweedie"]:
                diff = self.y / s["diff__"]
            else:
                diff = self.y - s["diff__"]
            dat = copy.deepcopy(s[cols])
            if self.embedding_target_encoding_params_layer is not None:
                embedding_target_encoding_params = self.embedding_target_encoding_params_layer[idx]
                if embedding_target_encoding_params is not None and len(embedding_target_encoding_params) > 0:
                    target_encoding_model = TargetEncoding(y=diff, **embedding_target_encoding_params)
                    target_encoding_model.fit(dat)
                    dat = target_encoding_model.transform(dat)
                    self.target_encoding_models.append(target_encoding_model)
                else:
                    self.target_encoding_models.append(None)
            lgb_regression_model = LGBMRegression(y=diff,
                                                  cols=cols, pred_name="pred",
                                                  skip_check_transform_type=self.skip_check_transform_type,
                                                  drop_input_data=self.drop_input_data,
                                                  support_sparse_input=self.support_sparse_input,
                                                  native_init_params=self.native_init_params,
                                                  native_fit_params=self.native_fit_params,
                                                  verbose=-1,
                                                  objective=self.objective,
                                                  use_faster_predictor=self.use_faster_predictor,
                                                  dataset_params=dataset_params)
            lgb_regression_model.fit(copy.deepcopy(dat))
            if self.objective in ["poisson", "tweedie"]:
                s["diff__"] *= lgb_regression_model.transform(dat)["pred"]
            else:
                s["diff__"] += lgb_regression_model.transform(dat)["pred"]
            self.lgb_regression_models.append(lgb_regression_model)
        # 3.清除不需要的数据
        del s["diff__"]

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        if self.objective in ["poisson", "tweedie"]:
            pred_value = 1
        else:
            pred_value = 0
        for idx, model in enumerate(self.lgb_regression_models):
            dat = copy.deepcopy(s)
            # target encoding
            if self.embedding_target_encoding_params_layer is not None:
                target_encoding_model = self.target_encoding_models[idx]
                if target_encoding_model is not None:
                    dat = target_encoding_model.transform(dat)
            if self.objective in ["poisson", "tweedie"]:
                pred_value *= model.transform(dat)["pred"].values
            else:
                pred_value += model.transform(dat)["pred"].values
        result = pd.DataFrame(pred_value, columns=[self.pred_name], index=s.index)
        return result

    def udf_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        if self.objective in ["poisson", "tweedie"]:
            pred_value = 1
        else:
            pred_value = 0
        for idx, model in enumerate(self.lgb_regression_models):
            dat = copy.deepcopy(s)
            # target encoding
            if self.embedding_target_encoding_params_layer is not None:
                target_encoding_model = self.target_encoding_models[idx]
                if target_encoding_model is not None:
                    dat = target_encoding_model.transform_single(dat)
            if self.objective in ["poisson", "tweedie"]:
                pred_value *= model.transform_single(dat)["pred"]
            else:
                pred_value += model.transform_single(dat)["pred"]
        return {self.pred_name: pred_value}

    def udf_get_params(self) -> dict_type:
        lgbm_regression_params = []
        for model in self.lgb_regression_models:
            lgbm_regression_params.append(model.get_params())
        target_encoding_params = []
        if self.embedding_target_encoding_params_layer is not None:
            for target_model in self.target_encoding_models:
                if target_model is not None:
                    target_encoding_params.append(target_model.get_params())
                else:
                    target_encoding_params.append(None)

        params = {"cols_layer": self.cols_layer,
                  "pred_name": self.pred_name,
                  "drop_input_data": self.drop_input_data,
                  "support_sparse_input": self.support_sparse_input,
                  "native_init_params": self.native_init_params,
                  "native_fit_params": self.native_fit_params,
                  "use_faster_predictor": self.use_faster_predictor,
                  "objective": self.objective,
                  "lgbm_regression_params": lgbm_regression_params,
                  "embedding_target_encoding_params_layer": self.embedding_target_encoding_params_layer,
                  "target_encoding_params": target_encoding_params}
        return params

    def udf_set_params(self, params: dict):
        from easymlops.table.regression import LGBMRegression
        self.lgb_regression_models = []
        for model_params in params["lgbm_regression_params"]:
            model = LGBMRegression()
            model.set_params(model_params)
            self.lgb_regression_models.append(model)
        self.cols_layer = params["cols_layer"]
        self.pred_name = params["pred_name"]
        self.drop_input_data = params["drop_input_data"]
        self.support_sparse_input = params["support_sparse_input"]
        self.native_init_params = params["native_init_params"]
        self.native_fit_params = params["native_fit_params"]
        self.use_faster_predictor = params["use_faster_predictor"]
        self.objective = params["objective"]
        self.embedding_target_encoding_params_layer = params["embedding_target_encoding_params_layer"]
        if self.embedding_target_encoding_params_layer is not None:
            self.target_encoding_models = []
            for model_params in params["target_encoding_params"]:
                if model_params is not None:
                    target_model = TargetEncoding()
                    target_model.set_params(model_params)
                    self.target_encoding_models.append(target_model)
                else:
                    self.target_encoding_models.append(None)


class AutoResidualRegression(LGBMRegressionLayers):
    def __init__(self, y: series_type = None, pred_name="pred", layers=3,
                 skip_check_transform_type=True,
                 drop_input_data=True, support_sparse_input=False,
                 native_init_params=None, native_fit_params=None,
                 use_faster_predictor=True,
                 shuffle_cols=True,
                 verbose=-1, objective="regression",
                 **kwargs):
        super().__init__(y=y, pred_name=pred_name, cols_layer=None, dataset_params_layer=None,
                         embedding_target_encoding_params_layer=None,
                         skip_check_transform_value=skip_check_transform_type,
                         drop_input_data=drop_input_data, support_sparse_input=support_sparse_input,
                         native_init_params=native_init_params, native_fit_params=native_fit_params,
                         use_faster_predictor=use_faster_predictor, verbose=verbose, objective=objective,
                         **kwargs)
        self.layers = layers
        self.shuffle_cols = shuffle_cols

    def udf_fit(self, s: dataframe_type, **kwargs):
        cols_all = s.columns.tolist()
        assert self.layers < len(cols_all)
        if self.shuffle_cols:
            np.random.shuffle(cols_all)
        self.cols_layer = self.split_array_equally(cols_all, self.layers)
        self.embedding_target_encoding_params_layer = []
        for cols in self.cols_layer:
            record_cate_cols = {"cols": []}
            for col in cols:
                if "int" not in str(s[col].dtype).lower() and "float" not in str(s[col].dtype).lower():
                    record_cate_cols["cols"].append(col)
            self.embedding_target_encoding_params_layer.append(record_cate_cols)
        return super().udf_fit(s)

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        output = super().udf_transform(s)
        if self.drop_input_data is False:
            output = PandasUtils.concat_duplicate_columns([s, output])
        return output

    def udf_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        output = super().udf_transform_single(s)
        if self.drop_input_data is False:
            s.update(output)
            output = s
        return output

    def udf_get_params(self) -> dict_type:
        return {"layers": self.layers}

    def udf_set_params(self, params: dict):
        self.layers = params["layers"]

    @staticmethod
    def split_array_equally(arr, k):
        n = len(arr)
        group_size = n // k
        remainder = n % k
        result = []
        start = 0
        for i in range(k):
            # 前 'remainder' 组每组多一个元素
            if i < remainder:
                end = start + group_size + 1
            else:
                end = start + group_size
            result.append(arr[start:end])
            start = end
        return result
