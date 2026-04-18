import os

import pandas as pd
import copy
from easymlops.table.preprocessing import FixInput, FillNa
from easymlops.table.encoding import LabelEncoding
from easymlops.table.classification import LGBMClassification
from easymlops.table.regression import LGBMRegression
from easymlops.table.strategy import AutoResidualRegression
from easymlops.table.strategy import HillClimbingStackingRegression
from easymlops import TablePipeLine
from tqdm.notebook import tqdm
from easymlops.table.utils import EvalFunction
from .tools import LLMToolManager
from .sessions import LLMSessionManager
import datetime


class AutoMLTab(object):
    def __init__(self, trn_path=None, val_path=None, label="label", task_type="regression", llm_models=[],
                 llm_weights=[],
                 load_self_tool=True,
                 describe_feature_function=None,
                 eval_metric="mae", eval_metric_direct="minimize",
                 early_stop_steps=10, max_iter=5, max_time=24 * 60 * 60,
                 keep_history_message_len=5,
                 llm_max_iter_each_time=3,
                 suffix_pipeline_prefix=None):
        """
        :param trn_path:训练集路径
        :param val_path:验证集路径
        :param label:标签列名
        :param task_type:任务类型
        :param llm_models:llm模型组,需要实现一个invoke接口,可以调用一系列模型
        :param llm_weights:llm模型组权重
        :param load_self_tool:是否加载自带工具
        :param describe_feature_function:对feature的describe
        :param eval_metric:评估函数(y_true,y_pred)
        :param eval_metric_direct:minimize或maximize,越小越好或者越大越好
        :param early_stop_steps:如果evaluate值经过一定的迭代次数后依然没有降低，则终止
        :param max_iter:最大执行批次
        :param max_time:最大执行时间
        :param keep_history_message_len:默认保留最近5次的对话历史
        :param llm_max_iter_each_time:每次请求llm,最多重复次数
        :param suffix_pipeline_prefix:最后一个pipeline要不要加前缀,stacking的时候使用
        """
        self.trn_path = trn_path
        self.val_path = val_path
        self.label = label
        self.task_type = task_type
        # 由于llm_model通常在序列化时候会有问题,先转class+param成结构存储,fit前重构为object,fit后释放
        self.llm_models_struct = self.llm_models_o2c(llm_models)
        self.llm_tool_manager = LLMToolManager(llm_models=[], llm_weights=llm_weights)
        self.llm_session_manager = LLMSessionManager()
        self.describe_feature_function = describe_feature_function
        self.eval_metric = eval_metric
        self.eval_metric_direct = eval_metric_direct
        self.early_stop_steps = early_stop_steps
        self.max_iter = max_iter
        self.max_time = max_time
        self.keep_history_message_len = keep_history_message_len
        self.llm_max_iter_each_time = llm_max_iter_each_time
        if load_self_tool:
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            for filename in os.listdir(f"{root_dir}/automl/config/"):
                self.llm_tool_manager.load_yaml_config(f"{root_dir}/automl/config/" + filename)
        self.message_queue = []
        self.all_chat_info = []
        self.all_evaluate_values = []
        self.tools_count = {}
        self.prefix_pipeline = TablePipeLine() - FixInput() - FillNa()
        self.main_pipeline = TablePipeLine()
        self.suffix_pipeline = TablePipeLine()
        self.combine_pipeline = None
        self.suffix_pipeline_prefix = suffix_pipeline_prefix

    @staticmethod
    def llm_models_o2c(llm_models):
        llm_models_struct = []
        for llm_model in llm_models:
            llm_models_struct.append((llm_model.__class__, llm_model.model))
        return llm_models_struct

    @staticmethod
    def llm_model_c2o(llm_models_struct):
        llm_models = []
        for clz, model in llm_models_struct:
            llm_models.append(clz(model=model))
        return llm_models

    def fit_suffix_pipeline(self, x_, y_):
        x = copy.deepcopy(x_)
        y = copy.deepcopy(y_)
        if self.task_type == "regression":
            self.suffix_pipeline = TablePipeLine()
            cat_cols = []
            for col in x.columns:
                if x[col].dtype == object:
                    cat_cols.append(col)
            self.suffix_pipeline.pipe(AutoResidualRegression(y=y, drop_input_data=False,
                                                             pred_name=f"auto_residual_regression__predict__"
                                                                       f"{self.suffix_pipeline_prefix}"))
            self.suffix_pipeline.pipe(LabelEncoding(cols=cat_cols))
            self.suffix_pipeline.pipe(LGBMRegression(y=y, prefix=self.suffix_pipeline_prefix))
            self.suffix_pipeline.fit(x)
        else:  # classificaion
            self.suffix_pipeline = TablePipeLine()
            cat_cols = []
            for col in x.columns:
                if x[col].dtype == object:
                    cat_cols.append(col)
            self.suffix_pipeline.pipe(LabelEncoding(cols=cat_cols))
            self.suffix_pipeline.pipe(LGBMClassification(y=y, prefix=self.suffix_pipeline_prefix))
            self.suffix_pipeline.fit(x)

    def evaluate_calculate(self, x_, y_):
        x = copy.deepcopy(x_)
        y = copy.deepcopy(y_)
        eval_metric_func = self.eval_metric
        if type(eval_metric_func) == str:
            eval_metric_func = EvalFunction.get(eval_metric_func)
        if self.task_type == "regression":
            try:
                output_col_name = self.suffix_pipeline[-1].pred_name
                if self.suffix_pipeline[-1].prefix is not None:
                    output_col_name = f"{self.suffix_pipeline[-1].prefix}_{output_col_name}"
                return eval_metric_func(y.values, self.suffix_pipeline.transform(self.main_pipeline.transform(x))[
                    output_col_name].values)
            except:
                return eval_metric_func(y.values, y.values.mean())

    def fit(self):
        self.llm_tool_manager.llm_models = self.llm_model_c2o(self.llm_models_struct)
        # 1.读取数据
        current_trn_feature = pd.read_csv(self.trn_path)
        current_val_feature = pd.read_csv(self.val_path)

        self.prefix_pipeline.fit(current_trn_feature)
        current_trn_feature = self.prefix_pipeline.transform(current_trn_feature)
        current_val_feature = self.prefix_pipeline.transform(current_val_feature)

        init_trn_label = current_trn_feature[self.label]
        init_val_label = current_val_feature[self.label]

        del current_trn_feature[self.label]
        del current_val_feature[self.label]

        init_trn_feature = copy.deepcopy(current_trn_feature)
        init_val_feature = copy.deepcopy(current_val_feature)

        state = {"tools_count": {},
                 "trn_evaluate_value": [],
                 "val_evaluate_value": [],
                 "history_messages": [],
                 "main_pipeline": copy.deepcopy(self.main_pipeline)}

        # self.fit_suffix_pipeline(current_trn_feature, init_trn_label)
        # 初始
        trn_evaluate_value = self.evaluate_calculate(init_trn_feature, init_trn_label)
        val_evaluate_value = self.evaluate_calculate(init_val_feature, init_val_label)

        state["trn_evaluate_value"].append(copy.deepcopy(trn_evaluate_value))
        state["val_evaluate_value"].append(copy.deepcopy(val_evaluate_value))
        pid = self.llm_session_manager.add_state(state)
        start_time = datetime.datetime.now()
        # 2.循环调用llm做优化
        for iter_num in tqdm(list(range(self.max_iter))):
            end_time = datetime.datetime.now()
            if (end_time - start_time).total_seconds() > self.max_time:
                continue
            # 0.early_stopping
            if len(state["val_evaluate_value"]) > self.early_stop_steps \
                    and state["val_evaluate_value"][-1] >= state["val_evaluate_value"][-self.early_stop_steps]:
                continue
            # 0.当前状态无可用工具
            if len(self.llm_tool_manager.extract_can_use_tools(state["tools_count"])) == 0:
                continue
            # 1.获取当前可用的tools
            call_status, tools, message, response, call_models, msg \
                = self.llm_tool_manager.call_simple_tools(state["tools_count"],
                                                          trn_feature=current_trn_feature,
                                                          history_messages=state["history_messages"],
                                                          max_iter=self.llm_max_iter_each_time)
            state["operator"] = "call_simple_tools"
            state["call_status"] = call_status
            state["tools"] = tools
            state["history_messages"].append(message)
            state["history_messages"].append(response)
            state["call_models"] = call_models
            state["msg"] = msg

            call_simple_tools_id = self.llm_session_manager.add_state(state)
            # 2.操作tools
            call_tools_list = []
            for tool in tools:
                # 免受中途太多tool对话的影响
                history_messages = self.llm_session_manager.get_state(call_simple_tools_id)["history_messages"]
                call_status, tool_param, message, response, call_models, msg \
                    = self.llm_tool_manager.call_tool_param(tool_name=tool,
                                                            trn_feature=current_trn_feature,
                                                            trn_label=init_trn_label,
                                                            history_messages=history_messages,
                                                            max_iter=self.llm_max_iter_each_time)
                state["operator"] = "call_tool_param"
                state["tool"] = tool
                state["call_status"] = call_status
                state["tool_param"] = tool_param
                history_messages.append(message)
                history_messages.append(response)
                state["history_messages"] = history_messages
                state["call_models"] = call_models
                state["msg"] = msg
                call_tool_param_id = self.llm_session_manager.add_state(state)

                call_status, pipe_, current_trn_feature_after_call_tool, tools_count, msg \
                    = self.llm_tool_manager.call_tool(tool={"action": tool, "action_params": tool_param},
                                                      tools_count=state["tools_count"], trn_feature=current_trn_feature,
                                                      trn_label=init_trn_label)

                if call_status:
                    self.main_pipeline.pipe(pipe_)
                    current_trn_feature = current_trn_feature_after_call_tool
                else:
                    pass  # do nothing

                state["operator"] = "call_tool"
                state["tool"] = tool
                state["call_status"] = call_status
                state["main_pipeline"] = copy.deepcopy(self.main_pipeline)
                state["tools_count"] = tools_count
                state["msg"] = msg

                call_tool_id = self.llm_session_manager.add_state(state)

                call_tools_list.append((call_tool_param_id, call_tool_id))

                end_time = datetime.datetime.now()
                if (end_time - start_time).total_seconds() > self.max_time:
                    continue

            # 当前批tools运行后,对suffix_pipeline做一次训练
            self.fit_suffix_pipeline(current_trn_feature, init_trn_label)
            # 评估
            trn_evaluate_value = self.evaluate_calculate(init_trn_feature, init_trn_label)
            val_evaluate_value = self.evaluate_calculate(init_val_feature, init_val_label)

            if self.eval_metric_direct == "minimize":
                if val_evaluate_value <= self.llm_session_manager.get_state(pid)["val_evaluate_value"][-1]:
                    # 更新
                    state["trn_evaluate_value"].append(trn_evaluate_value)
                    state["val_evaluate_value"].append(val_evaluate_value)
                    state["history_messages"] = self.llm_session_manager.get_state(call_simple_tools_id)[
                        "history_messages"]
                    pid = self.llm_session_manager.register_state(pid, state)
                else:  # 恢复
                    self.main_pipeline = self.llm_session_manager.get_state(pid)["main_pipeline"]
                    current_trn_feature = self.main_pipeline.transform(copy.deepcopy(init_trn_feature))
                    self.fit_suffix_pipeline(current_trn_feature, init_trn_label)
            else:
                if val_evaluate_value >= self.llm_session_manager.get_state(pid)["val_evaluate_value"][-1]:
                    # 更新
                    state["trn_evaluate_value"].append(trn_evaluate_value)
                    state["val_evaluate_value"].append(val_evaluate_value)
                    state["history_messages"] = self.llm_session_manager.get_state(call_simple_tools_id)[
                        "history_messages"]
                    pid = self.llm_session_manager.register_state(pid, state)
                else:  # 恢复
                    self.main_pipeline = self.llm_session_manager.get_state(pid)["main_pipeline"]
                    current_trn_feature = self.main_pipeline.transform(copy.deepcopy(init_trn_feature))
                    self.fit_suffix_pipeline(current_trn_feature, init_trn_label)

        # fit完后重新构建pipeline
        self.combine_pipeline = TablePipeLine()
        self.combine_pipeline.pipe(self.prefix_pipeline).pipe(self.main_pipeline).pipe(self.suffix_pipeline)
        # 释放llm_models
        self.llm_tool_manager.llm_models = []

    def transform(self, x):
        return self.combine_pipeline.transform(x)

    def transform_single(self, x):
        return self.combine_pipeline.transform_single(x)

    def save(self, path):
        self.combine_pipeline.save(path)

    def load(self, path):
        self.combine_pipeline = TablePipeLine()
        self.combine_pipeline.load(path)

    def __getitem__(self, index):
        return self.combine_pipeline[index]


class AutoML(object):
    def __init__(self, trn_path=None, val_path=None, label="label", task_type="regression", llm_models=[],
                 llm_weights=[],
                 load_self_tool=True,
                 describe_feature_function=None,
                 eval_metric="mae", eval_metric_direct="minimize",
                 early_stop_steps=10, max_iter=5, max_time=24 * 60 * 60,
                 keep_history_message_len=5, llm_max_iter_each_time=3,
                 parallel=5, agg_type="lightgbm_stacking", backend="threading"):
        """
        :param trn_path:训练集路径
        :param val_path:验证集路径
        :param label:标签列名
        :param task_type:任务类型
        :param llm_models:llm模型组,需要实现一个invoke接口,可以调用一系列模型
        :param llm_weights:llm模型组权重
        :param load_self_tool:是否加载自带工具
        :param describe_feature_function:对feature的describe
        :param eval_metric:评估函数(y_true,y_pred)
        :param eval_metric_direct:方向minimize或maximize
        :param early_stop_steps:如果evaluate值经过一定的迭代次数后依然没有降低，则终止
        :param max_iter:5
        :param max_time:最大运行时间
        :param keep_history_message_len:默认保留最近5次的对话历史
        :param llm_max_iter_each_time:每次请求llm,最多重复次数
        :param parallel:并行模型
        :param agg_type:聚合方式:mean/median/lightgbm_stacking/sum/hill_climb_stacking
        """
        assert agg_type in ["mean", "median", "lightgbm_stacking", "sum", "hill_climb_stacking"]
        self.models = []
        self.parallel = parallel
        self.agg_type = agg_type
        self.agg_model = None
        self.backend = backend
        self.task_type = task_type
        self.trn_path = trn_path
        self.val_path = val_path
        self.label = label
        self.eval_metric = eval_metric
        self.eval_metric_direct = eval_metric_direct
        if task_type == "regression":
            for idx in range(self.parallel):
                self.models.append(AutoMLTab(trn_path=trn_path, val_path=val_path, label=label, task_type=task_type,
                                             llm_models=llm_models, llm_weights=llm_weights,
                                             load_self_tool=load_self_tool,
                                             describe_feature_function=describe_feature_function,
                                             eval_metric=eval_metric, eval_metric_direct=eval_metric_direct,
                                             early_stop_steps=early_stop_steps,
                                             max_iter=max_iter, max_time=max_time,
                                             keep_history_message_len=keep_history_message_len,
                                             llm_max_iter_each_time=llm_max_iter_each_time,
                                             suffix_pipeline_prefix=f"p{idx}_"))

    def __getitem__(self, index):
        return self.models[index]

    def parallel_fit(self, idx):
        self.models[idx].fit()
        return self.models[idx]

    def fit(self):
        from joblib import Parallel, delayed
        trn_data = pd.read_csv(self.trn_path)
        self.models = Parallel(n_jobs=self.parallel, backend=self.backend)(
            delayed(self.parallel_fit)(idx) for idx in list(range(self.parallel)))
        output_df = pd.concat([self.models[idx].transform(trn_data)
                               for idx in list(range(self.parallel))], axis=1)
        # agg方式
        if self.task_type == "regression":
            if self.agg_type == "mean":
                from easymlops.table.preprocessing import Mean
                self.agg_model = TablePipeLine() - Mean(output_col_name="pred")
                self.agg_model.fit(output_df)
            elif self.agg_type == "median":
                from easymlops.table.preprocessing import Median
                self.agg_model = TablePipeLine() - Median(output_col_name="pred")
                self.agg_model.fit(output_df)
            elif self.agg_type == "sum":
                from easymlops.table.preprocessing import Sum
                self.agg_model = TablePipeLine() - Sum(output_col_name="pred")
                self.agg_model.fit(output_df)
            elif self.agg_type == "lightgbm_stacking":
                from easymlops.table.ensemble import Parallel
                self.agg_model = TablePipeLine() \
                                 - Parallel([AutoResidualRegression(y=trn_data[self.label], pred_name="se1"),
                                             AutoResidualRegression(y=trn_data[self.label], pred_name="se2"),
                                             AutoResidualRegression(y=trn_data[self.label], pred_name="se3"),
                                             AutoResidualRegression(y=trn_data[self.label], pred_name="se4"),
                                             AutoResidualRegression(y=trn_data[self.label], pred_name="se5"),
                                             ], drop_input_data=False) - LGBMRegression(y=trn_data[self.label])
                self.agg_model.fit(output_df)
            elif self.agg_type == "hill_climb_stacking":
                from easymlops.table.ensemble import Parallel
                self.agg_model = TablePipeLine() \
                                 - Parallel([AutoResidualRegression(y=trn_data[self.label], pred_name="se1"),
                                             AutoResidualRegression(y=trn_data[self.label], pred_name="se2"),
                                             AutoResidualRegression(y=trn_data[self.label], pred_name="se3"),
                                             AutoResidualRegression(y=trn_data[self.label], pred_name="se4"),
                                             AutoResidualRegression(y=trn_data[self.label], pred_name="se5"),
                                             ], drop_input_data=False) \
                                 - HillClimbingStackingRegression(y=trn_data[self.label],
                                                                  eval_function=self.eval_metric,
                                                                  eval_function_direct=self.eval_metric_direct)

                self.agg_model.fit(output_df)

    def transform(self, x):
        output_df = pd.concat([self.models[idx].transform(x)
                               for idx in list(range(len(self.models)))], axis=1)
        return self.agg_model.transform(output_df)

    def save(self, path):
        # 1.创建目录
        if not os.path.exists(path):
            os.makedirs(path)
        # 2.保存models
        models_path = (path + "/models").replace("//", "/")
        os.makedirs(models_path)
        for idx in range(len(self.models)):
            self.models[idx].save(models_path + f"/{idx}.pkl")
        # 3.保存agg_model
        agg_model_path = (path + "/agg_model.pkl").replace("//", "/")
        self.agg_model.save(agg_model_path)

    def load(self, path):
        models_path = (path + "/models/").replace("//", "/")
        agg_model_path = (path + "/agg_model.pkl").replace("//", "/")
        # 1.load models
        self.models = []
        for filename in sorted(os.listdir(models_path)):
            automl_tab = AutoMLTab()
            automl_tab.load(models_path + filename)
            self.models.append(automl_tab)
        # 2.load agg model
        self.agg_model = TablePipeLine()
        self.agg_model.load(agg_model_path)
