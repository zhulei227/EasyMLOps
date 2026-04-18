"""
tool管理基础工具
"""
import pandas as pd
import numpy as np
import copy
from easymlops.table.preprocessing import FixInput, FillNa, Normalizer, Bins, DropCols, Tanh, Sigmoid, Swish, Relu
from easymlops.table.encoding import LabelEncoding, TargetEncoding, WOEEncoding
from easymlops.table.regression import LGBMRegression
from easymlops.table.classification import LGBMClassification, RandomForestClassification
from easymlops.table.regression import RidgeCVRegression, LinearRegression, LGBMRegression, LogisticRegression, \
    RidgeRegression
from easymlops.nlp.representation import Word2VecModel, FastTextModel, Doc2VecModel
from easymlops.table.decomposition import FastICADecomposition, PCADecomposition, KernelPCADecomposition, \
    LocallyLinearEmbeddingDecomposition, NMFDecomposition
from easymlops.table.preprocessing import Add, Subtract, Multiply, Divide, DateDayDiff
from easymlops.table.preprocessing import DateMonthInfo, DateHourInfo, DateMinuteInfo, DateTotalMinuteInfo
from easymlops.table.sqls import SQL
from easymlops.table.preprocessing import CrossCategoryWithNumber,CrossNumberWithNumber
from easymlops.table.strategy import AutoResidualRegression


class LLMToolManager(object):
    def __init__(self, llm_models, llm_weights):
        # 权值归一化
        if len(llm_weights) == 1:
            llm_weights = np.asarray([1.])
        else:
            llm_weights = np.asarray(llm_weights)
            llm_weights = llm_weights / llm_weights.sum()

        self.llm_models = llm_models
        self.llm_weights = llm_weights
        self.tool_map_param = {}

    def load_yaml_config(self, path):
        import yaml
        tool_map_param = yaml.load(open(path, encoding="utf8"), Loader=yaml.FullLoader)
        # 校验tool重名
        common_tool_names = tool_map_param.keys() & self.tool_map_param.keys()
        if len(common_tool_names) > 0:
            raise Exception(f"load {path} error,tools has exists:{common_tool_names}")
        self.tool_map_param.update(tool_map_param)

    @DeprecationWarning
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

    def extract_tool_describe(self, tool_name):
        """
        获取单个tool的详细描述信息
        """
        detail = self.tool_map_param[tool_name]
        describe = detail["describe"].format(detail["parameters_template"])
        parameters = detail["parameters"]
        return {"tool": tool_name, "describe": describe, "parameters": parameters}

    def extract_tools_simple_describe(self, tools_count):
        """
        获取所有tool的简单描述信息
        """
        canot_use_tools = self.extract_canot_use_tools(tools_count=tools_count)
        tool_names = []
        simple_describes = []
        for tool_name, detail in self.tool_map_param.items():
            if tool_name not in canot_use_tools:
                tool_names.append(tool_name)
                simple_describes.append(detail["simple_describe"])
        simple_describes = pd.DataFrame({"tool": tool_names, "simple_describe": simple_describes}).to_dict("records")
        np.random.shuffle(simple_describes)
        return simple_describes

    def extract_tools_name(self, tools_count):
        """
        抽取所有tool名称
        """
        canot_use_tools = self.extract_canot_use_tools(tools_count=tools_count)
        tools = list(self.tool_map_param.keys() - set(canot_use_tools))
        np.random.shuffle(tools)
        return "[" + ",".join(tools) + "]"

    def call_simple_tools(self, tools_count, trn_feature, history_messages, max_iter=3):
        """
        目的是拿到到tool列表:[tool1,tool2,...]
        """
        # 获取必要信息
        simple_tools_describe = self.extract_tools_simple_describe(tools_count=tools_count)
        tools_name = self.extract_tools_name(tools_count=tools_count)
        feature_describe = self.extract_feature_describe(copy.deepcopy(trn_feature))
        # 定义状态
        call_status = False
        current_max_iter = max_iter
        tools = []
        message = ""
        response = ""
        msg = ""
        call_models = []
        while current_max_iter >= 0:
            message = f"""
                   You are ai engineer, you have access to the following tools(in json format):

                   {simple_tools_describe}

                   Use the following format:

                   Question: accept the input question from user, and you must thought how to action
                   Thought: you should always think about what to do,and return you action
                   Action: the action to take, should be some of {tools_name},return like [tool1,tool2,tool3,...]

                   Begin!

                   Question: my current feature representation (in JSON format):
                             {feature_describe}
                             you should  help me chose the tools above,and format like ```Action: [tool1,tool2,tool3,...]```
                   Thought: """
            llm_model = self.choice_llm_model()
            call_models.append(llm_model.model)
            response = llm_model.chat(
                copy.deepcopy(history_messages) + [{"role": "user", "content": message}])
            # 尝试提取tools
            for line in str(response).replace("\n```json:", "").replace("\n```json", "") \
                    .replace("```json:", "").replace("```json", "").replace("`", "").split("\n"):
                if "Action:" in line:
                    for tool_name in line.replace("Action:", "") \
                            .replace(" ", "").replace("[", "").replace("]", "").split(","):
                        if tool_name in self.tool_map_param and tool_name not in tools:
                            tools.append(tool_name)
            # 如果拿到了有效tool则终止
            if len(tools) > 0:
                call_status = True
                msg = "success"
                break
            else:
                current_max_iter -= 1
                msg = "failed"
        # print("call tools", tools)
        return call_status, tools, {"role": "user", "content": message}, \
            {"role": "assistant", "content": response}, call_models, msg

    def extract_tool_parameters_template(self, tool_name):
        return self.tool_map_param.get(tool_name, {}).get("parameters_template", {})

    def call_tool_param(self, tool_name, trn_feature, trn_label, history_messages,
                        max_iter=3):
        """
        目的是拿到单个tool对应的parameters
        """
        # 1.获取必要信息
        tool_describe = self.extract_tool_describe(tool_name)
        feature_describe = self.extract_feature_describe(trn_feature)
        tool_parameters_template = self.extract_tool_parameters_template(tool_name)
        # 2.定义状态信息
        call_status = False
        tiny_trn_feature = copy.deepcopy(trn_feature[:100])
        tiny_trn_label = copy.deepcopy(trn_label[:100])
        current_max_iter = max_iter
        tool_param = {}
        message = ""
        response = ""
        msg = ""
        call_models = []
        while current_max_iter >= 0:
            message = f"""
                   you chose the tool:{tool_name},the describe as follow:

                   {tool_describe}

                   now the feature representation  as follow:
                    {feature_describe}

                   You are ai engineer,you should  help me build the tool parameters format like ```Parameters: {tool_parameters_template}```
                   Parameters: """
            llm_model = self.choice_llm_model()
            call_models.append(llm_model.model)
            response = llm_model.chat(
                copy.deepcopy(history_messages) + [{"role": "user", "content": message}])
            # 尝试提取Parameters
            for line in str(response).replace("\n```json:", "").replace("\n```json", "") \
                    .replace("```json:", "").replace("```json", "").replace("`", "").split("\n"):
                if "Parameters:" in line:
                    try:
                        tool_param = eval(line.replace("Parameters:", "").replace(" ", ""))
                        break
                    except:
                        pass
            # check
            try:
                tool_param_ = copy.deepcopy(tool_param)
                clz = eval(self.tool_map_param[tool_name]['class'])
                if self.tool_map_param[tool_name].get("require_label") is True:
                    tool_param_.update({"y": tiny_trn_label})
                action_param = self.tool_map_param[tool_name].get("default_parameters", {})
                action_param.update(tool_param_)
                pipe_ = clz(**action_param)
                pipe_.fit(copy.deepcopy(tiny_trn_feature)).transform(copy.deepcopy(tiny_trn_feature))
                # print(f"call {tool_name} tool params success", tool_param)
                call_status = True
                msg = "success"
                break
            except Exception as e:
                # print("======================================================================")
                # print(f"call {tool_name} tool params fail,exception:{e}")
                # print("======================================================================")
                # print(f"{tool_name}message:{message}")
                # print("======================================================================")
                # print(f"{tool_name} response:{response}")
                # print("======================================================================")
                # print(f"{tool_name}history messages:{history_messages}")
                # print("======================================================================")
                current_max_iter -= 1
                msg = f"{e}"
        return call_status, tool_param, {"role": "user", "content": message}, \
            {"role": "assistant", "content": response}, call_models, msg

    @DeprecationWarning
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

    def choice_llm_model(self):
        """
        按概率随机选择一个llm调用
        """
        return np.random.choice(a=self.llm_models, p=self.llm_weights)

    @staticmethod
    def extract_feature_describe(x):
        import toad
        if len(x) > 0 and len(x.columns):
            data_info = toad.detect(x)
            data_info["col"] = list(x.columns)
            data_info["sample"] = data_info["1%_or_top4"].astype(str) \
                                  + ";" + data_info["10%_or_top5"].astype(str) \
                                  + ";" + data_info["50%_or_bottom5"].astype(str) \
                                  + ";" + data_info["90%_or_bottom3"].astype(str) \
                                  + ";" + data_info["99%_or_bottom2"].astype(str)
            data_info = data_info[["col", "type", "size", "missing", "unique", "sample"]]
            data_info["unique_rate"] = data_info["unique"] / data_info["size"]
            data_info["type_describe"] = data_info["type"].apply(
                lambda x: "数值型" if "int" in str(x).lower() or "float" in str(x).lower() else "离散型")
            data_info = data_info[["col", "type", "type_describe", "missing", "unique", "unique_rate", "sample"]]
            data_info.columns = ["col", "type", "type_describe", "missing", "unique", "unique_rate", "sample"]
            return data_info.to_dict("records")
        else:
            return []

    def call_tool(self, tool, tools_count, trn_feature, trn_label):
        """
        tool:action+parameters
        tools_count:当前状态下tool被调用过的次数
        """
        msg = ""
        if tool["action"] in self.tool_map_param:
            if tool["action"] not in tools_count:
                tools_count[tool["action"]] = 0
            if tools_count[tool["action"]] < self.tool_map_param[tool["action"]]["can_use_time"]:
                tool_params = self.tool_map_param[tool["action"]]
                try:
                    clz = eval(tool_params['class'])
                    if tool_params.get("require_label") is True:
                        tool["action_params"].update({"y": trn_label})
                    if "action_params" in tool:
                        action_param = self.tool_map_param[tool["action"]].get("default_parameters", {})
                        action_param.update(tool["action_params"])
                        pipe_ = clz(**action_param)
                    else:
                        pipe_ = clz()
                    trn_feature = pipe_.fit(copy.deepcopy(trn_feature)).transform(copy.deepcopy(trn_feature))
                    if len(trn_feature.columns) > 0 and len(trn_feature) > 0:
                        tools_count[tool["action"]] += 1
                        msg = "success"
                        return True, pipe_, trn_feature, tools_count, msg
                    else:
                        msg = "data or col be deleted"
                        return False, None, trn_feature, tools_count, msg
                except Exception as e:
                    msg = f"{e}"
        return False, None, trn_feature, tools_count, msg

    def extract_canot_use_tools(self, tools_count):
        canot_use_tools = []
        for tool in tools_count:
            if tools_count[tool] == self.tool_map_param[tool]["can_use_time"]:
                canot_use_tools.append(tool)
        return canot_use_tools

    def extract_can_use_tools(self, tools_count):
        return list(self.tool_map_param.keys() - set(self.extract_canot_use_tools(tools_count)))

    @DeprecationWarning
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
