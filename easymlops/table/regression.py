from easymlops.table.core import *
import pandas as pd
from easymlops.table.utils import PandasUtils


class RegressionBase(TablePipeObjectBase):
    """
    回归模型基类。
    
    提供回归模型的通用框架，支持多种底层回归算法。
    
    Example:
        >>> class MyRegressor(RegressionBase):
        ...     def udf_fit(self, s, **kwargs):
        ...         self.model.fit(s, self.y, **self.native_fit_params)
        ...         return self
    """
    
    def __init__(self, y: series_type = None, cols="all", pred_name="pred", skip_check_transform_type=True,
                 drop_input_data=True, support_sparse_input=False,
                 native_init_params=None, native_fit_params=None,
                 **kwargs):
        """
        初始化回归模型。
        
        Args:
            y: 目标变量
            cols: 用于模型训练的列
            pred_name: 模型输出的预测列名，默认"pred"
            skip_check_transform_type: 跳过类型检测
            drop_input_data: 是否删除输入数据，默认True
            support_sparse_input: 是否支持稀疏矩阵输入
            native_init_params: 底层回归模型的init参数
            native_fit_params: 底层回归模型的fit参数
            **kwargs: 其他父类参数
        """
        super().__init__(skip_check_transform_type=skip_check_transform_type, **kwargs)
        if cols is None or type(cols) == str:
            self.cols = []
        else:
            self.cols = cols
        self.drop_input_data = drop_input_data
        self.y = copy.deepcopy(y)
        self.pred_name = pred_name
        self.support_sparse_input = support_sparse_input
        self.native_init_params = copy.deepcopy(native_init_params)
        self.native_fit_params = copy.deepcopy(native_fit_params)
        if self.native_init_params is None:
            self.native_init_params = dict()
        if self.native_fit_params is None:
            self.native_fit_params = dict()

    def before_fit(self, s: dataframe_type, **kwargs) -> dataframe_type:
        """fit前处理，选择用于训练的列。"""
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
        """transform前处理，选择用于预测的列。"""
        s = super().before_transform(s, **kwargs)
        if self.check_list_same(s.columns.tolist(), self.cols):
            return s
        else:
            return s[self.cols]

    def transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        """批量数据转换。"""
        s_ = self.after_transform(self.udf_transform(self.before_transform(s, **kwargs), **kwargs), **kwargs)
        if not self.drop_input_data:
            s_ = PandasUtils.concat_duplicate_columns([s, s_])
        return s_

    def before_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        """单条数据转换前处理。"""
        s = super().before_transform_single(s, **kwargs)
        return self.extract_dict(s, self.cols)

    def transform_single(self, s: dict_type, **kwargs) -> dict_type:
        """单条数据转换。"""
        s_ = copy.deepcopy(s)
        s_ = self.after_transform_single(
            self.udf_transform_single(self.before_transform_single(s_, **kwargs), **kwargs), **kwargs)
        if not self.drop_input_data:
            s.update(s_)
            return s
        else:
            return s_

    def udf_fit(self, s, **kwargs):
        """用户自定义的fit实现。"""
        return self

    def udf_transform(self, s, **kwargs):
        """用户自定义的transform实现。"""
        return s

    def udf_transform_single(self, s: dict_type, **kwargs):
        """用户自定义的单条数据预测实现。"""
        input_dataframe = pd.DataFrame([s])
        input_dataframe = input_dataframe[self.cols]
        return self.udf_transform(input_dataframe, **kwargs).to_dict("records")[0]

    def udf_get_params(self) -> dict_type:
        """获取参数。"""
        return {"pred_name": self.pred_name, "cols": self.cols,
                "drop_input_data": self.drop_input_data, "support_sparse_input": self.support_sparse_input}

    def udf_set_params(self, params: dict):
        """设置参数。"""
        self.pred_name = params["pred_name"]
        self.cols = params["cols"]
        self.drop_input_data = params["drop_input_data"]
        self.support_sparse_input = params["support_sparse_input"]


class LGBMRegression(RegressionBase):
    """
    LightGBM回归模型。
    
    使用LightGBM实现的回归模型，支持快速预测和稀疏矩阵输入。
    
    Example:
        >>> model = LGBMRegression(y=y, cols=feature_cols)
    """
    
    def __init__(self, y=None, support_sparse_input=False, verbose=-1, objective="regression",
                 use_faster_predictor=True, dataset_params=None, **kwargs):
        """
        初始化LightGBM回归模型。
        
        Args:
            y: 目标变量
            support_sparse_input: 是否支持稀疏数据
            verbose: 日志显示级别，默认-1不显示
            objective: 目标函数，默认regression
            use_faster_predictor: 是否使用预测加速
            dataset_params: 透传DataSet参数
            **kwargs: 其他父类参数
        """
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
        """
        训练LightGBM回归模型。
        
        Args:
            s: 输入数据
            **kwargs: 其他参数
            
        Returns:
            self
        """
        import lightgbm as lgb
        if self.support_sparse_input:
            s_ = PandasUtils.pd2csr(s)
        else:
            s_ = s
        self.lgb_model = lgb.train(params=self.native_init_params,
                                   train_set=lgb.Dataset(data=s_, label=self.y, **self.dataset_params),
                                   **self.native_fit_params)
        if self.use_faster_predictor:
            from easymlops.table.utils import FasterLgbSinglePredictor
            self.lgb_model_faster_predictor_params = self.lgb_model.dump_model()
            self.lgb_model_faster_predictor = FasterLgbSinglePredictor(
                model=self.lgb_model_faster_predictor_params, cache_num=10)
        return self

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        """
        批量预测。
        
        Args:
            s: 输入数据
            **kwargs: 其他参数
            
        Returns:
            预测结果DataFrame
        """
        if self.support_sparse_input:
            s_ = PandasUtils.pd2csr(s)
        else:
            s_ = s
        result = pd.DataFrame(self.lgb_model.predict(s_), columns=[self.pred_name], index=s.index)
        return result

    def udf_transform_single(self, s: dict_type, **kwargs):
        """单条数据预测。"""
        if self.use_faster_predictor:
            return {self.pred_name: self.lgb_model_faster_predictor.predict(s).get("score")}
        else:
            input_dataframe = pd.DataFrame([s])
            input_dataframe = input_dataframe[self.cols]
            return self.udf_transform(input_dataframe, **kwargs).to_dict("records")[0]

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


class LogisticRegression(RegressionBase):
    """
    逻辑回归模型。
    
    使用sklearn实现的逻辑回归分类器。
    """
    
    def __init__(self, y=None, **kwargs):
        """初始化逻辑回归模型。"""
        super().__init__(y=y, **kwargs)
        self.lr = None

    def udf_fit(self, s: dataframe_type, **kwargs):
        """训练逻辑回归模型。"""
        if self.support_sparse_input:
            s_ = PandasUtils.pd2csr(s)
        else:
            s_ = s
        from sklearn.linear_model import LogisticRegression
        self.lr = LogisticRegression(**self.native_init_params)
        self.lr.fit(s_, self.y, **self.native_fit_params)
        return self

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        """批量预测。"""
        if self.support_sparse_input:
            s_ = PandasUtils.pd2csr(s)
        else:
            s_ = s
        result = pd.DataFrame(self.lr.predict(s_),
                              columns=[self.pred_name], index=s.index)
        return result

    def udf_get_params(self) -> dict:
        """获取参数。"""
        return {"lr": self.lr}

    def udf_set_params(self, params: dict):
        """设置参数。"""
        self.lr = params["lr"]


class LinearRegression(RegressionBase):
    """
    线性回归模型。
    
    使用sklearn实现的线性回归器。
    """
    
    def __init__(self, y=None, **kwargs):
        """初始化线性回归模型。"""
        super().__init__(y=y, **kwargs)
        self.lr = None

    def udf_fit(self, s: dataframe_type, **kwargs):
        """训练线性回归模型。"""
        if self.support_sparse_input:
            s_ = PandasUtils.pd2csr(s)
        else:
            s_ = s
        from sklearn.linear_model import LinearRegression
        self.lr = LinearRegression(**self.native_init_params)
        self.lr.fit(s_, self.y, **self.native_fit_params)
        return self

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        """批量预测。"""
        if self.support_sparse_input:
            s_ = PandasUtils.pd2csr(s)
        else:
            s_ = s
        result = pd.DataFrame(self.lr.predict(s_),
                              columns=[self.pred_name], index=s.index)
        return result

    def udf_get_params(self) -> dict:
        """获取参数。"""
        return {"lr": self.lr}

    def udf_set_params(self, params: dict):
        self.lr = params["lr"]


class RidgeRegression(RegressionBase):
    """
    岭回归模型。
    
    使用sklearn实现的岭回归器，带L2正则化。
    """
    
    def __init__(self, y=None, **kwargs):
        """初始化岭回归模型。"""
        super().__init__(y=y, **kwargs)
        self.lr = None

    def udf_fit(self, s: dataframe_type, **kwargs):
        """训练岭回归模型。"""
        if self.support_sparse_input:
            s_ = PandasUtils.pd2csr(s)
        else:
            s_ = s
        from sklearn.linear_model import Ridge
        self.lr = Ridge(**self.native_init_params)
        self.lr.fit(s_, self.y, **self.native_fit_params)
        return self

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        """批量预测。"""
        if self.support_sparse_input:
            s_ = PandasUtils.pd2csr(s)
        else:
            s_ = s
        result = pd.DataFrame(self.lr.predict(s_),
                              columns=[self.pred_name], index=s.index)
        return result

    def udf_get_params(self) -> dict:
        """获取参数。"""
        return {"lr": self.lr}

    def udf_set_params(self, params: dict):
        """设置参数。"""
        self.lr = params["lr"]


class RidgeCVRegression(RegressionBase):
    """
    交叉验证岭回归模型。
    
    使用sklearn实现的RidgeCV，自动选择最优正则化参数。
    """
    
    def __init__(self, y=None, **kwargs):
        """初始化交叉验证岭回归模型。"""
        super().__init__(y=y, **kwargs)
        self.lr = None

    def udf_fit(self, s: dataframe_type, **kwargs):
        """训练交叉验证岭回归模型。"""
        if self.support_sparse_input:
            s_ = PandasUtils.pd2csr(s)
        else:
            s_ = s
        from sklearn.linear_model import RidgeCV
        self.lr = RidgeCV(**self.native_init_params)
        self.lr.fit(s_, self.y, **self.native_fit_params)
        return self

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        """批量预测。"""
        if self.support_sparse_input:
            s_ = PandasUtils.pd2csr(s)
        else:
            s_ = s
        result = pd.DataFrame(self.lr.predict(s_),
                              columns=[self.pred_name], index=s.index)
        return result

    def udf_get_params(self) -> dict:
        return {"lr": self.lr}

    def udf_set_params(self, params: dict):
        self.lr = params["lr"]


class SVMRegression(RegressionBase):
    """
    支持向量机回归模型。
    
    使用sklearn或thundersvm实现的SVR回归器。
    """
    
    def __init__(self, y=None, use_gpu=True, **kwargs):
        """
        初始化SVM回归模型。
        
        Args:
            y: 目标变量
            use_gpu: 是否使用GPU加速
            **kwargs: 其他父类参数
        """
        super().__init__(y=y, **kwargs)
        self.svr = None
        self.use_gpu = use_gpu

    def udf_fit(self, s: dataframe_type, **kwargs):
        """训练SVM回归模型。"""
        if self.support_sparse_input:
            s_ = PandasUtils.pd2csr(s)
        else:
            s_ = s
        if self.use_gpu:
            from thundersvm import SVR
        else:
            from sklearn.svm import SVR
        self.svr = SVR(**self.native_init_params)
        self.svr.fit(s_, self.y, **self.native_fit_params)
        return self

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        """批量预测。"""
        if self.support_sparse_input:
            s_ = PandasUtils.pd2csr(s)
        else:
            s_ = s
        result = pd.DataFrame(self.svr.predict(s_),
                              columns=[self.pred_name], index=s.index)
        return result

    def udf_get_params(self) -> dict:
        """获取参数。"""
        return {"svr": self.svr}

    def udf_set_params(self, params: dict):
        """设置参数。"""
        self.svr = params["svr"]
