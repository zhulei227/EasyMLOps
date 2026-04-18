import os
import pandas as pd
import copy
from easymlops.table.preprocessing import FixInput, FillNa, Normalizer, Bins, DropCols
from easymlops.table.encoding import LabelEncoding, TargetEncoding, WOEEncoding
from easymlops.table.regression import LGBMRegression
from easymlops.table.classification import LGBMClassification, RandomForestClassification
from easymlops.table.regression import RidgeCVRegression, LinearRegression, LGBMRegression, LogisticRegression, \
    RidgeRegression
from easymlops import TablePipeLine
from tqdm import tqdm
import numpy as np


class StateNode(object):
    def __init__(self):
        pass


class AutoMLTab(object):
    def __init__(self, trn_path=None, val_path=None, label="label", task_type="regression", llm_models=[],
                 llm_weights=[],
                 load_self_tool=True,
                 describe_feature_function=None,
                 evaluate_function="mae", early_stop_steps=10, max_iter=5):
        """
        :param trn_path:训练集路径
        :param val_path:验证集路径
        :param label:标签列名
        :param task_type:任务类型
        :param llm_models:llm模型组,需要实现一个invoke接口,可以调用一系列模型
        :param llm_weights:llm模型组权重
        :param load_self_tool:是否加载自带工具
        :param describe_feature_function:对feature的describe
        :param evaluate_function:评估函数(x,y),需要使结果越小越好
        :param early_stop_steps:如果evaluate值经过一定的迭代次数后依然没有降低，则终止
        :param max_iter:5
        """
        self.trn_path = trn_path
        self.val_path = val_path
        self.label = label
        self.task_type = task_type
        self.llm_models = llm_models
        self.llm_weights = llm_weights
        if len(self.llm_models) == 1:
            self.llm_weights = np.asarray([1.])
        else:
            self.llm_weights = np.asarray(self.llm_weights)
            self.llm_weights = self.llm_weights / self.llm_weights.sum()
        self.describe_feature_function = describe_feature_function
        self.evaluate_function = evaluate_function
        self.early_stop_steps = early_stop_steps
        self.max_iter = max_iter
        self.tool_map_param = {}
        if load_self_tool:
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            for filename in os.listdir(f"{root_dir}/automl/config/"):
                self.load_yaml_config(f"{root_dir}/automl/config/" + filename)
        self.message_queue = []
        self.all_chat_info = []
        self.all_evaluate_values = []
        self.tools_count = {}
        self.prefix_pipeline = TablePipeLine() - FixInput() - FillNa()
        self.main_pipeline = TablePipeLine()
        self.suffix_pipeline = None

    def load_yaml_config(self, path):
        import yaml
        tool_map_param = yaml.load(open(path, encoding="utf8"), Loader=yaml.FullLoader)
        # 校验tool重名
        common_tool_names = tool_map_param.keys() & self.tool_map_param.keys()
        if len(common_tool_names) > 0:
            raise Exception(f"load {path} error,tools has exists:{common_tool_names}")
        self.tool_map_param.update(tool_map_param)

    def extract_tools_describe(self, canot_use_tools):
        tool_names = []
        describes = []
        params = []
        for tool_name, detail in self.tool_map_param.items():
            if tool_name not in canot_use_tools:
                tool_names.append(tool_name)
                describes.append(detail["describe"].format(detail["parameters_template"]))
                params.append(detail["parameters"])
        describes = pd.DataFrame({"tool": tool_names, "describe": describes, "parameters": params}).to_dict("records")
        np.random.shuffle(describes)
        return describes

    def extract_tools_name(self, canot_use_tools):
        tools = list(self.tool_map_param.keys() - set(canot_use_tools))
        np.random.shuffle(tools)
        return "[" + ",".join(tools) + "]"

    @staticmethod
    def format_prompt(tools_describe, tools_name, feature_describe, trn_evaluate_value,
                      val_evaluate_value):
        message = f"""
        You are ai engineer, you have access to the following tools(in json format):

        {tools_describe}
        
        Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do,and return you thought
        Action: the action to take, should be one of {tools_name}
        Action Params: the params to the action

        Begin!

        Question: my current results are as follows:
                  1. Feature representation (in JSON format):
                  {feature_describe}
                  2. Training set error: {trn_evaluate_value}, validation set error: {val_evaluate_value}
                  You need to further help me optimize the feature expression to make the validation set error lower and lower
        Thought: """
        return message

    @staticmethod
    def extract_feature_describe(x):
        import toad
        data_info = toad.detect(x)
        data_info["col"] = list(x.columns)
        data_info = data_info[["col", "type", "size", "missing", "unique"]]
        data_info["unique_rate"] = data_info["unique"] / data_info["size"]
        data_info["type"] = data_info["type"].apply(
            lambda x: "数值" if "int" in str(x) or "float" in str(x) else "离散")
        data_info = data_info[["col", "type", "missing", "unique_rate"]]
        data_info.columns = ["特征", "类型", "缺失率", "唯一值占比"]
        return data_info.to_dict("records")

    def fit_suffix_pipeline(self, x_, y_):
        x = copy.deepcopy(x_)
        y = copy.deepcopy(y_)
        if self.task_type == "regression":
            self.suffix_pipeline = TablePipeLine()
            cat_cols = []
            for col in x.columns:
                if x[col].dtype == object:
                    cat_cols.append(col)
            self.suffix_pipeline.pipe(LabelEncoding(cols=cat_cols))
            self.suffix_pipeline.pipe(LGBMRegression(y=y))
            self.suffix_pipeline.fit(x)
        else:  # classificaion
            self.suffix_pipeline = TablePipeLine()
            cat_cols = []
            for col in x.columns:
                if x[col].dtype == object:
                    cat_cols.append(col)
            self.suffix_pipeline.pipe(LabelEncoding(cols=cat_cols))
            self.suffix_pipeline.pipe(LGBMClassification(y=y))
            self.suffix_pipeline.fit(x)

    def evaluate_calculate(self, x_, y_):
        x = copy.deepcopy(x_)
        y = copy.deepcopy(y_)
        if self.task_type == "regression":
            if self.evaluate_function == "mae":
                try:
                    return np.mean(
                        np.abs(y.values - self.suffix_pipeline.transform(self.main_pipeline.transform(x))["pred"]))
                except:
                    return np.mean(np.abs(y.values - y.values.mean()))

    def call_tool(self, tool, trn_feature, trn_label):
        if tool["action"] in self.tool_map_param:
            if tool["action"] not in self.tools_count:
                self.tools_count[tool["action"]] = 0
            if self.tools_count[tool["action"]] < self.tool_map_param[tool["action"]]["can_use_time"]:
                tool_params = self.tool_map_param[tool["action"]]
                try:
                    clz = eval(tool_params['class'])
                    if tool_params.get("require_label") is True:
                        tool["action_params"].update({"y": trn_label})
                    if "action_params" in tool:
                        pipe_ = clz(**tool["action_params"])
                    else:
                        pipe_ = clz()
                    trn_feature = pipe_.fit(copy.deepcopy(trn_feature)).transform(copy.deepcopy(trn_feature))
                    print("use tool success:", tool)
                    self.tools_count[tool["action"]] += 1
                    return True, pipe_, trn_feature
                except:
                    print("use tool fail:", tool)
        return False, None, trn_feature

    def extract_canot_use_tools(self):
        canot_use_tools = []
        for tool in self.tools_count:
            if self.tools_count[tool] == self.tool_map_param[tool]["can_use_time"]:
                canot_use_tools.append(tool)
        return canot_use_tools

    def fit(self):
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

        # 2.循环调用llm做优化
        for _ in tqdm(list(range(self.max_iter))):
            canot_use_tools = self.extract_canot_use_tools()
            trn_evaluate_value = self.evaluate_calculate(init_trn_feature, init_trn_label),
            val_evaluate_value = self.evaluate_calculate(init_val_feature, init_val_label),
            self.all_evaluate_values.append({"trn": trn_evaluate_value, "val": val_evaluate_value})
            message = self.format_prompt(tools_describe=self.extract_tools_describe(canot_use_tools),
                                         tools_name=self.extract_tools_name(canot_use_tools),
                                         feature_describe=self.extract_feature_describe(current_trn_feature),
                                         trn_evaluate_value=trn_evaluate_value,
                                         val_evaluate_value=val_evaluate_value)
            # 如果下面代码块有运用工具才加入queue
            start_len = len(self.main_pipeline.models)
            self.all_chat_info.append([])
            self.all_chat_info[-1].append({"role": "user", "content": message})
            response = self.choice_llm_model().chat(self.message_queue + [{"role": "user", "content": message}])
            self.all_chat_info[-1].append({"role": "assistant", "content": response})
            tools = self.extract_tools(response)
            for tool in tools:
                call_tool_status, pipe_, current_trn_feature = self.call_tool(tool, current_trn_feature, init_trn_label)
                if call_tool_status:
                    self.main_pipeline.pipe(pipe_)
            if len(self.main_pipeline.models) > start_len:
                self.message_queue += [{"role": "user", "content": message}, {"role": "assistant", "content": response}]
            # 不管是否成功,对suffix_pipeline做一次训练
            self.fit_suffix_pipeline(current_trn_feature, init_trn_label)

    def choice_llm_model(self):
        return np.random.choice(a=self.llm_models, p=self.llm_weights)

    @staticmethod
    def extract_tools(llm_response):
        tools = []
        tool = {}
        for line in llm_response.split("\n"):
            if "Action:" in line:
                tool_name = line.replace("Action:", "").replace(" ", "")
                if "," in tool_name:  # 多个连续工具只取第一个
                    tool_name = tool_name.split(",")[0]
                if len(tool) != 0:
                    tools.append(tool)
                tool["action"] = tool_name
            if "Action Params:" in line:
                try:
                    tool_params = eval(line.replace("Action Params:", "").replace(" ", ""))
                    if "action" in tool:
                        tool["action_params"] = tool_params
                except:
                    pass
            if "action" in tool and "action_params" in tool:
                tools.append(tool)
                tool = {}
        return tools

    def transform(self, x):
        return self.suffix_pipeline.transform(self.main_pipeline.transform(self.prefix_pipeline.transform(x)))
