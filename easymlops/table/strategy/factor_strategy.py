import itertools
from .core import *


class FactorStrategy(TablePipeObjectBase):
    """
    构造单因子/多因子输出模型，并输出路径
    """

    def __init__(self, y: series_type = None, cols=None,
                 ruler_prefix="ruler", ruler_feature_num=1, ruler_feature_num_down_expand=True,
                 keep_ruler_rate=1.0, keep_max_ruler_num=10240,
                 skip_check_transform_type=True,
                 drop_input_data=True, support_sparse_input=False,
                 native_init_params=None, native_fit_params=None,
                 use_faster_predictor=True, dataset_params=None,
                 verbose=-1, objective="regression", **kwargs):
        """
        :param y:
        :param cols: 用于模型训练的cols
        :param ruler_prefix: 规则集前缀
        :param ruler_feature_num:构建规则所用的特征数量
        :param ruler_feature_num_down_expand:True,向下扩展入模特征,ruler_feature_num=2的时候，
                                                 同时考虑ruler_feature_num=1和ruler_feature_num的情况
        :param keep_ruler_rate:在ruler_feature_num>=2的时候，设置为<1，随机筛选构建规则的比例
        :param keep_max_ruler_num:保留最多的规则数
        :param skip_check_transform_type: 跳过类型检测
        :param drop_input_data: 删掉输入数据，默认True，不然输出为x1,x2,..,xn,one_var_ruler1,one_var_ruler2....
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
        self.ruler_prefix = ruler_prefix
        self.ruler_feature_num = ruler_feature_num
        self.ruler_feature_num_down_expand = ruler_feature_num_down_expand
        self.keep_ruler_rate_global = keep_ruler_rate
        self.keep_ruler_rate = {ruler_feature_num: keep_ruler_rate}
        self.keep_max_ruler_num = keep_max_ruler_num
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
            self.native_fit_params = dict(num_boost_round=1)
        else:
            self.native_fit_params.update(dict(num_boost_round=1))
        self.dataset_params = copy.deepcopy(dataset_params)
        if self.dataset_params is None:
            self.dataset_params = dict()
        self.use_faster_predictor = use_faster_predictor
        # 记录规则名称+规则集+规则所对应的模型tuple(tree_index,ruler_cols,ruler_model)
        self.ruler_totals = []
        self.record_ruler_detail_df = None
        self.used_tree_index = []

    @staticmethod
    def factorial_loop(n):
        result = 1
        for i in range(1, n + 1):
            result *= i
        return result

    def show_detail(self):
        return self.record_ruler_detail_df

    def before_fit(self, s: dataframe_type, **kwargs) -> dataframe_type:
        s = super().before_fit(s, **kwargs)
        assert self.y is not None
        if len(self.cols) == 0:
            self.cols = s.columns.tolist()
        assert type(self.cols) == list
        if self.check_list_same(s.columns.tolist(), self.cols):
            s_ = s
        else:
            s_ = s[self.cols]

        # 扩充keep_ruler_rate
        ruler_feature_num_ = self.ruler_feature_num
        while ruler_feature_num_ > 0:
            combine_num = self.factorial_loop(len(self.cols)) / self.factorial_loop(
                len(self.cols) - ruler_feature_num_) / self.factorial_loop(ruler_feature_num_)
            self.keep_ruler_rate[ruler_feature_num_] = min(self.keep_ruler_rate_global,
                                                           self.keep_max_ruler_num / combine_num)
            ruler_feature_num_ -= 1
            if not self.ruler_feature_num_down_expand:
                break
        return s_

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
        s_ = s.reset_index()
        del s_["index"]
        s_["target__"] = self.y.values
        trn_data = s_
        record_ruler_detail = []
        all_idx = 0  # 全局idx
        for ruler_feature_num, keep_ruler_rate in self.keep_ruler_rate.items():
            combinations_ = list(itertools.combinations(self.cols, ruler_feature_num))
            combinations = []
            for item in combinations_:
                tmp_ = []
                for item_ in item:
                    tmp_.append(item_)
                combinations.append(tmp_)
            enu_combinations = list(enumerate(combinations))
            if kwargs.get("show_process", False):
                print("fit:")
                enu_combinations = tqdm(enu_combinations)
            for _, input_cols in enu_combinations:
                if np.random.random() <= keep_ruler_rate:
                    model = LGBMRegression4Ruler(y=trn_data["target__"], cols=input_cols,
                                                 support_sparse_input=self.support_sparse_input,
                                                 verbose=-1, objective=self.objective,
                                                 use_faster_predictor=self.use_faster_predictor,
                                                 dataset_params=self.dataset_params,
                                                 native_init_params=self.native_init_params,
                                                 native_fit_params=self.native_fit_params)
                    model.fit(trn_data[input_cols])
                    # 2.记录模型
                    self.ruler_totals.append([all_idx, input_cols, model])
                    self.used_tree_index.append(all_idx)
                    # 3.预测
                    trn_predict = model.transform(trn_data[input_cols])
                    # 4.校验效果(整体)
                    trn_data["predict__"] = trn_predict[model.predict_prefix + "_score"]
                    # 4.校验效果(细节)
                    trn_data["leaf_index__"] = trn_predict[model.predict_prefix + "_leaf_index"]

                    trn_data["leaf_describe__"] = trn_predict[model.predict_prefix + "_leaf_describe"]

                    trn_cnt_data = trn_data.groupby(["leaf_index__", "leaf_describe__"]). \
                        agg({"target__": ["mean", "count"]}).reset_index()

                    trn_cnt_data.columns = ["leaf_index__", "leaf_describe__", "mean", "count"]
                    trn_cnt_data = trn_cnt_data.reset_index()
                    del trn_cnt_data["index"]

                    trn_cnt_data["tree_index"] = all_idx
                    trn_cnt_data["factor"] = ",".join(input_cols)

                    trn_cnt_data = trn_cnt_data[["factor", "tree_index", "leaf_index__", "leaf_describe__"]]

                    trn_cnt_data.columns = ["factor", "tree_index", "leaf_index", "leaf_describe"]
                    record_ruler_detail.append(trn_cnt_data)
                    all_idx += 1
        # 整合所有规则
        record_ruler_detail_df = pd.concat(record_ruler_detail)
        record_ruler_detail_df = record_ruler_detail_df.reset_index()
        del record_ruler_detail_df["index"]
        self.record_ruler_detail_df = record_ruler_detail_df
        return self

    def udf_transform(self, s, **kwargs):
        values = {}
        if kwargs.get("show_process", False):
            print("transform:")
            indexes = tqdm(range(len(self.ruler_totals)))
        else:
            indexes = range(len(self.ruler_totals))
        for index_ in indexes:
            tree_index, input_cols, model = self.ruler_totals[index_]
            if tree_index not in self.used_tree_index:
                continue
            leaf_indexes = model.transform(s[input_cols])["ruler_leaf_index"].values
            values[tree_index] = leaf_indexes
        result = pd.DataFrame(values, index=s.index)
        for col in values.keys():
            result[col] = result[col].astype(int)
        return result

    def udf_transform_single(self, s: dict_type, **kwargs):
        values = {}
        if kwargs.get("show_process", False):
            print("transform:")
            indexes = tqdm(range(len(self.ruler_totals)))
        else:
            indexes = range(len(self.ruler_totals))
        for index_ in indexes:
            tree_index, input_cols, model = self.ruler_totals[index_]
            if tree_index not in self.used_tree_index:
                continue
            s_ = {}
            for col in input_cols:
                s_[col] = s[col]
            leaf_indexes = int(model.transform_single(s_)["ruler_leaf_index"])
            values[tree_index] = leaf_indexes
        return values

    def update_used_tree_index(self, used_tree_index: list):
        self.used_tree_index = used_tree_index
        self.output_col_names = used_tree_index

    def udf_get_params(self) -> dict_type:
        ruler_totals_ = []
        for tree_index, input_cols, model in self.ruler_totals:
            ruler_totals_.append([tree_index, input_cols, model.get_params()])
        params = {"cols": self.cols,
                  "drop_input_data": self.drop_input_data,
                  "ruler_prefix": self.ruler_prefix,
                  "ruler_feature_num": self.ruler_feature_num,
                  "ruler_feature_num_down_expand": self.ruler_feature_num_down_expand,
                  "keep_ruler_rate_global": self.keep_ruler_rate_global,
                  "keep_ruler_rate": self.keep_ruler_rate,
                  "keep_max_ruler_num": self.keep_max_ruler_num,
                  "support_sparse_input": self.support_sparse_input,
                  "native_init_params": self.native_init_params,
                  "native_fit_params": self.native_fit_params,
                  "use_faster_predictor": self.use_faster_predictor,
                  "ruler_totals": ruler_totals_,
                  "record_ruler_detail_df": self.record_ruler_detail_df,
                  "used_tree_index": self.used_tree_index}
        return params

    def udf_set_params(self, params: dict):
        ruler_totals_ = []
        for tree_index, input_cols, model_params in params["ruler_totals"]:
            model = LGBMRegression4Ruler()
            model.set_params(model_params)
            ruler_totals_.append([tree_index, input_cols, model])

        self.cols = params["cols"]
        self.drop_input_data = params["drop_input_data"]
        self.ruler_prefix = params["ruler_prefix"]
        self.ruler_feature_num = params["ruler_feature_num"]
        self.ruler_feature_num_down_expand = params["ruler_feature_num_down_expand"]
        self.keep_ruler_rate = params["keep_ruler_rate"]
        self.keep_ruler_rate_global = params["keep_ruler_rate_global"]
        self.keep_max_ruler_num = params["keep_max_ruler_num"]
        self.support_sparse_input = params["support_sparse_input"]
        self.native_init_params = params["native_init_params"]
        self.native_fit_params = params["native_fit_params"]
        self.use_faster_predictor = params["use_faster_predictor"]
        self.ruler_totals = ruler_totals_
        self.record_ruler_detail_df = params["record_ruler_detail_df"]
        # 兼容版本
        if params.get("used_tree_index") is None or len(params["used_tree_index"]) == 0:
            print(self.__class__.__name__ + " 缺失 used_tree_index,使用全局tree_index替代")
            self.used_tree_index = []
            for tree_index, _, _ in self.ruler_totals:
                self.used_tree_index.append(tree_index)
        else:
            self.used_tree_index = params["used_tree_index"]
