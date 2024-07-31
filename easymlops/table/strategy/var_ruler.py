import toad
from easymlops.table.core import *
import pandas as pd
from easymlops.table.utils import PandasUtils
import itertools
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
            apply(lambda x: self.lgb_model_faster_predictor.leaf_map_describe[0].get(x, ""))
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
                        self.lgb_model_faster_predictor.leaf_map_describe[0].get(faster_predict.get("leaf_index")[0],
                                                                                 "")}
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


class VarRuler(TablePipeObjectBase):
    """
    单变量/多变量规则，默认使用LGBMRegression构造规则
    """

    def __init__(self, y: series_type = None, cols=None,
                 ruler_prefix="ruler", ruler_feature_num=1, keep_ruler_rate=1.0, keep_max_ruler_num=10240,
                 skip_check_transform_type=True,
                 drop_input_data=True, support_sparse_input=False,
                 native_init_params=None, native_fit_params=None,
                 use_faster_predictor=True, dataset_params=None,
                 verbose=-1, objective="regression",
                 val_split_size=0.1, val_split_type="last", val_shuffle=False, n_bins=10,
                 filter_iv=0.01, filter_psi=0.1, filter_diff=0.1, filter_ratio=0.01, filter_mean=0,
                 filter_udf_func=None,
                 output_type="average",
                 **kwargs):
        """
        :param y:
        :param cols: 用于模型训练的cols
        :param ruler_prefix: 规则集前缀
        :param ruler_feature_num:构建规则所用的特征数量
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
        :param val_split_size:验证集切割比例
        :param val_split_type:验证集切割方式，last表示切割最后一部分,random表示随机切割
        :param val_shuffle:是否做shuffle
        :param n_bins:计算iv时，对y做分箱
        :param filter_iv:训练集分箱后的iv>filter_iv则保留该规则
        :param filter_psi:训练集分箱和验证集分箱后的psi<filter_psi则保留规则
        :param filter_diff:叶子节点绝对提升,diff>filter_diff
        :param filter_ratio:叶子节点样本占比,ratio>filter_ratio
        :param filter_mean:叶子节点均值,mean>filter_mean
        :param filter_udf_func:自定义过滤规则,List[(指标名称,自定义计算函数,阈值,方向(True,默认小于)]
        :param output_type:输出方式,average平均,weighted_average加权平均
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
        self.keep_ruler_rate = keep_ruler_rate
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
        # 验证集切割方式
        self.val_split_size = val_split_size
        self.val_split_type = val_split_type
        self.val_shuffle = val_shuffle
        # 分箱
        self.n_bins = n_bins
        # 过滤规则
        self.filter_iv = filter_iv
        self.filter_psi = filter_psi
        self.filter_diff = filter_diff
        self.filter_ratio = filter_ratio
        self.filter_mean = filter_mean
        # 记录规则名称+规则集+规则所对应的模型tuple(ruler_name,ruler_cols,ruler_model)
        self.ruler_totals = []
        self.used_rulers = {}  # 记录满足要求的规则名称+规则id
        self.record_perfer_df = None  # 统计规则效果
        # 规则聚合方式
        self.output_type = output_type

    @staticmethod
    def factorial_loop(n):
        result = 1
        for i in range(1, n + 1):
            result *= i
        return result

    def show_detail(self):
        return self.record_perfer_df

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
        # 更新keep_rate
        combine_num = self.factorial_loop(len(self.cols)) / self.factorial_loop(
            len(self.cols) - self.ruler_feature_num) / self.factorial_loop(self.ruler_feature_num)
        self.keep_ruler_rate = min(self.keep_ruler_rate, self.keep_max_ruler_num / combine_num)
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
        # 1.切分trn,val
        s_ = s.copy(deep=True).reset_index()
        del s_["index"]
        s_["target__"] = self.y.values
        # 构建分箱，便于后续计算
        c = toad.transform.Combiner()
        c.fit(s_[["target__"]], method="quantile", n_bins=self.n_bins)
        s_["target__bin"] = c.transform(s_[["target__"]]).values
        split_len = int(len(s_) * (1 - self.val_split_size))
        idx_ = s_.index.tolist()
        if self.val_split_type == "random":
            np.random.shuffle(idx_)
        else:
            if self.val_shuffle:
                # 验证集再做一次随机打乱
                split_len_ = int(split_len - len(s_) * self.val_split_size)
                idx_trn_ = idx_[:split_len_]
                idx_val_ = idx_[split_len_:]
                np.random.shuffle(idx_val_)
                # 为了保持分布改变不变，对训练样本做了扩充
                idx_ = idx_trn_ + idx_val_[:len(idx_val_) // 2] \
                       + idx_val_[:len(idx_val_) // 2] + idx_val_[len(idx_val_) // 2:]
                split_len = len(idx_trn_) + len(idx_val_)  # 更新
        trn_data = s_.iloc[idx_[:split_len]]
        val_data = s_.iloc[idx_[split_len:]]
        trn_data = trn_data.reset_index()
        val_data = val_data.reset_index()
        del trn_data["index"]
        del val_data["index"]
        record_perfer_total = []
        record_perfer_detail = []
        combinations_ = list(itertools.combinations(self.cols, self.ruler_feature_num))
        combinations = []
        for item in combinations_:
            tmp_ = []
            for item_ in item:
                tmp_.append(item_)
            combinations.append(tmp_)
        for idx, input_cols in enumerate(combinations):
            if np.random.random() <= self.keep_ruler_rate:
                model = LGBMRegression4Ruler(y=trn_data["target__"], cols=input_cols,
                                             support_sparse_input=self.support_sparse_input,
                                             verbose=-1, objective=self.objective,
                                             use_faster_predictor=self.use_faster_predictor,
                                             dataset_params=self.dataset_params,
                                             native_init_params=self.native_init_params,
                                             native_fit_params=self.native_fit_params)
                model.fit(trn_data[input_cols])
                # 2.记录模型
                self.ruler_totals.append([self.ruler_prefix + f"_{idx}", input_cols, model])
                # 3.预测
                trn_predict = model.transform(trn_data[input_cols])
                val_predict = model.transform(val_data[input_cols])
                # 4.校验效果(整体)
                trn_data["predict__"] = trn_predict[model.predict_prefix + "_score"]
                val_data["predict__"] = val_predict[model.predict_prefix + "_score"]
                # 3.1 trn vs val psi
                trn_val_psi = toad.metrics.PSI(trn_data["predict__"], val_data["predict__"])
                # 3.2 trn iv
                c1 = toad.quality(trn_data[["predict__", "target__bin"]], target="target__bin", indicators=["iv"])
                trn_iv = c1["iv"].tolist()[0]
                # 3.2 val iv
                c2 = toad.quality(val_data[["predict__", "target__bin"]], target="target__bin", indicators=["iv"])
                val_iv = c2["iv"].tolist()[0]
                record_perfer_total.append([self.ruler_prefix + f"_{idx}", "|".join(input_cols),
                                            trn_iv, val_iv, trn_val_psi])
                # 4.校验效果(细节)
                trn_data["leaf_index__"] = trn_predict[model.predict_prefix + "_leaf_index"]
                val_data["leaf_index__"] = val_predict[model.predict_prefix + "_leaf_index"]

                trn_data["leaf_describe__"] = trn_predict[model.predict_prefix + "_leaf_describe"]
                val_data["leaf_describe__"] = val_predict[model.predict_prefix + "_leaf_describe"]

                trn_target_mean = trn_data["target__"].mean()
                trn_num = len(trn_data)

                val_target_mean = val_data["target__"].mean()
                val_num = len(val_data)

                trn_cnt_data = trn_data.groupby(["leaf_index__", "leaf_describe__"]). \
                    agg({"target__": ["mean", "count"]}).reset_index()

                val_cnt_data = val_data.groupby(["leaf_index__", "leaf_describe__"]). \
                    agg({"target__": ["mean", "count"]}).reset_index()

                trn_cnt_data.columns = ["leaf_index__", "leaf_describe__", "mean", "count"]
                val_cnt_data.columns = ["leaf_index__", "leaf_describe__", "mean", "count"]
                trn_cnt_data = trn_cnt_data.reset_index()
                val_cnt_data = val_cnt_data.reset_index()
                del trn_cnt_data["index"]
                del val_cnt_data["index"]

                trn_cnt_data["diff"] = trn_cnt_data["mean"] - trn_target_mean
                trn_cnt_data["ratio"] = trn_cnt_data["count"] / trn_num

                val_cnt_data["diff"] = val_cnt_data["mean"] - val_target_mean
                val_cnt_data["ratio"] = val_cnt_data["count"] / val_num
                # 补充计算自定义指标：TODO
                trn_cnt_data["ruler_name"] = self.ruler_prefix + f"_{idx}"
                val_cnt_data["ruler_name"] = self.ruler_prefix + f"_{idx}"

                trn_cnt_data = trn_cnt_data[["ruler_name", "leaf_index__", "leaf_describe__",
                                             "mean", "count", "diff", "ratio"]]
                val_cnt_data = val_cnt_data[["ruler_name", "leaf_index__", "leaf_describe__",
                                             "mean", "count", "diff", "ratio"]]
                trn_cnt_data.columns = ["ruler_name", "leaf_index", "leaf_describe",
                                        "trn_mean", "trn_count", "trn_diff", "trn_ratio"]
                val_cnt_data.columns = ["ruler_name", "leaf_index", "leaf_describe",
                                        "val_mean", "val_count", "val_diff", "val_ratio"]
                perfer_detail_df = pd.merge(trn_cnt_data, val_cnt_data,
                                            on=["ruler_name", "leaf_index", "leaf_describe"], how="left").fillna(0)
                record_perfer_detail.append(perfer_detail_df)
        # 5.筛选
        record_perfer_total_df = pd.DataFrame(data=record_perfer_total,
                                              columns=["ruler_name", "input_cols", "trn_iv", "val_iv", "psi"])
        record_perfer_detail_df = pd.concat(record_perfer_detail)
        record_perfer_total_df = record_perfer_total_df.reset_index()
        record_perfer_detail_df = record_perfer_detail_df.reset_index()
        del record_perfer_total_df["index"]
        del record_perfer_detail_df["index"]
        record_perfer_df = pd.merge(record_perfer_total_df, record_perfer_detail_df, on=["ruler_name"], how="inner")
        flag_index = (record_perfer_df["trn_iv"] > self.filter_iv) \
                     & (record_perfer_df["val_iv"] > self.filter_iv) \
                     & (record_perfer_df["psi"] < self.filter_psi) \
                     & (record_perfer_df["trn_diff"] > self.filter_diff) \
                     & (record_perfer_df["val_diff"] > self.filter_diff) \
                     & (record_perfer_df["trn_ratio"] > self.filter_ratio) \
                     & (record_perfer_df["val_ratio"] > self.filter_ratio) \
                     & (record_perfer_df["trn_mean"] > self.filter_mean) \
                     & (record_perfer_df["val_mean"] > self.filter_mean)
        record_perfer_df["hint"] = np.where(flag_index, 1, 0)
        self.record_perfer_df = record_perfer_df
        for item in record_perfer_df[record_perfer_df["hint"] == 1].to_dict("records"):
            ruler_name = item["ruler_name"]
            leaf_index = item["leaf_index"]
            weight = item["val_diff"]
            if ruler_name not in self.used_rulers:
                self.used_rulers[ruler_name] = {}
            if leaf_index not in self.used_rulers[ruler_name]:
                self.used_rulers[ruler_name][leaf_index] = weight
        return self

    def udf_transform(self, s, **kwargs):
        values = np.zeros(len(s))
        for ruler_name, input_cols, model in self.ruler_totals:
            if ruler_name in self.used_rulers:
                if self.output_type == "average":
                    values = values + model.transform(s[input_cols])["ruler_leaf_index"] \
                        .apply(lambda x: 1.0 if x in self.used_rulers[ruler_name] else 0).values
                elif self.output_type == "weighted_average":
                    values = values + model.transform(s[input_cols])["ruler_leaf_index"] \
                        .apply(lambda x: self.used_rulers[ruler_name].get(x, 0.0)).values
                else:
                    raise Exception("output_type should be average or weighted_average")
        values = values / (len(self.used_rulers) + 1e-3)
        result = pd.DataFrame(values, columns=[self.ruler_prefix + "_hint_rate"], index=s.index)
        return result

    def udf_transform_single(self, s: dict_type, **kwargs):
        values = 0.0
        for ruler_name, input_cols, model in self.ruler_totals:
            if ruler_name in self.used_rulers:
                s_ = {}
                for col in input_cols:
                    s_[col] = s[col]
                if self.output_type == "average":
                    if int(model.transform_single(s_)["ruler_leaf_index"]) in self.used_rulers[ruler_name]:
                        values = values + 1.0
                elif self.output_type == "weighted_average":
                    values += self.used_rulers[ruler_name].get(int(model.transform_single(s_)["ruler_leaf_index"]), 0.0)
                else:
                    raise Exception("output_type should be average or weighted_average")
        values = values / (len(self.used_rulers) + 1e-3)
        return {self.ruler_prefix + "_hint_rate": values}

    def udf_get_params(self) -> dict_type:
        ruler_totals_ = []
        for ruler_name, input_cols, model in self.ruler_totals:
            ruler_totals_.append([ruler_name, input_cols, model.get_params()])
        return {"cols": self.cols,
                "drop_input_data": self.drop_input_data,
                "ruler_prefix": self.ruler_prefix,
                "ruler_feature_num": self.ruler_feature_num,
                "keep_ruler_rate": self.keep_ruler_rate,
                "keep_max_ruler_num": self.keep_max_ruler_num,
                "support_sparse_input": self.support_sparse_input,
                "native_init_params": self.native_init_params,
                "native_fit_params": self.native_fit_params,
                "use_faster_predictor": self.use_faster_predictor,
                "val_split_size": self.val_split_size,
                "val_split_type": self.val_split_type,
                "val_shuffle": self.val_shuffle,
                "n_bins": self.n_bins,
                "filter_iv": self.filter_iv,
                "filter_psi": self.filter_psi,
                "filter_diff": self.filter_diff,
                "filter_ratio": self.filter_ratio,
                "filter_mean": self.filter_mean,
                "ruler_totals": ruler_totals_,
                "used_rulers": self.used_rulers,
                "record_perfer_df": self.record_perfer_df,
                "output_type": self.output_type}

    def udf_set_params(self, params: dict):
        ruler_totals_ = []
        for ruler_name, input_cols, model_params in params["ruler_totals"]:
            model = LGBMRegression4Ruler()
            model.set_params(model_params)
            ruler_totals_.append([ruler_name, input_cols, model])

        self.cols = params["cols"]
        self.drop_input_data = params["drop_input_data"]
        self.ruler_prefix = params["ruler_prefix"]
        self.ruler_feature_num = params["ruler_feature_num"]
        self.keep_ruler_rate = params["keep_ruler_rate"]
        self.keep_max_ruler_num = params["keep_max_ruler_num"]
        self.support_sparse_input = params["support_sparse_input"]
        self.native_init_params = params["native_init_params"]
        self.native_fit_params = params["native_fit_params"]
        self.use_faster_predictor = params["use_faster_predictor"]
        self.val_split_size = params["val_split_size"]
        self.val_split_type = params["val_split_type"]
        self.val_shuffle = params["val_shuffle"]
        self.n_bins = params["n_bins"]
        self.filter_iv = params["filter_iv"]
        self.filter_psi = params["filter_psi"]
        self.filter_diff = params["filter_diff"]
        self.filter_ratio = params["filter_ratio"]
        self.filter_mean = params["filter_mean"]
        self.ruler_totals = ruler_totals_
        self.used_rulers = params["used_rulers"]
        self.record_perfer_df = params["record_perfer_df"]
        self.output_type = params["output_type"]
