# AutoML

## 8.1 AutoMLTab

```python
from easymlops.automl import AutoMLTab

# 自动机器学习
automl = AutoMLTab(
    trn_path="train.csv",        # 训练集路径
    val_path="val.csv",          # 验证集路径
    label="target",               # 标签列名
    task_type="regression",       # 任务类型: regression/classification
    eval_metric="mae",            # 评估指标
    eval_metric_direct="minimize", # 评估方向: minimize/maximize
    max_iter=5,                   # 最大迭代次数
    max_time=24 * 60 * 60,        # 最大运行时间(秒)
    early_stop_steps=10,          # 早停步数
    llm_models=[],                # LLM模型列表
    llm_weights=[],               # LLM权重
    load_self_tool=True           # 是否加载自带工具
)

# 运行AutoML
automl.run()

# 预测
predictions = automl.predict(x_test)

# 获取最佳模型
best_model = automl.get_best_model()

# 获取评估历史
eval_history = automl.get_evaluation_history()
```

---

## 8.2 LLM 模型集成

```python
from easymlops.automl.llms import OllamaLLM, SparkLLM, ZhiPuLLM, KimiLLM

# Ollama
ollama = OllamaLLM(
    model_name="llama2",
    base_url="http://localhost:11434"
)

# Spark (讯飞)
spark = SparkLLM(
    app_id="xxx",
    api_key="xxx",
    api_secret="xxx"
)

# 智谱
zhipu = ZhiPuLLM(
    api_key="xxx"
)

# Kimi (月之暗面)
kimi = KimiLLM(
    api_key="xxx"
)

# 在AutoML中使用
automl = AutoMLTab(
    trn_path="train.csv",
    label="target",
    llm_models=[ollama, zhipu],
    llm_weights=[0.5, 0.5]
)
```

---

## 8.3 工具管理

```python
from easymlops.automl.tools import LLMToolManager

# 创建工具管理器
tool_manager = LLMToolManager(llm_models=[], llm_weights=[])

# 加载YAML配置
tool_manager.load_yaml_config("config/tab.yaml")

# 添加自定义工具
def custom_tool(param1, param2):
    """自定义工具"""
    return {"result": f"{param1} - {param2}"}

tool_manager.add_tool("custom_tool", custom_tool)

# 列出所有工具
tools = tool_manager.list_tools()
print(tools)

# 使用工具
result = tool_manager.invoke("tool_name", param1="value1", param2="value2")
```

---

## 8.4 会话管理

```python
from easymlops.automl.sessions import LLMSessionManager

# 创建会话管理器
session_manager = LLMSessionManager()

# 创建新会话
session_id = session_manager.create_session(
    system_prompt="你是一个机器学习助手"
)

# 添加用户消息
session_manager.add_message(
    session_id,
    role="user",
    content="如何提高模型准确率?"
)

# 获取回复
response = session_manager.get_response(session_id, llm_model)

# 获取历史
history = session_manager.get_history(session_id)

# 清除历史
session_manager.clear_history(session_id)
```

---

## 8.5 AutoML 配置

### 特征工程配置

```python
automl = AutoMLTab(
    trn_path="train.csv",
    label="target",
    # 特征描述函数
    describe_feature_function=lambda df: df.describe()
)
```

### 回调配置

```python
from easymlops.automl import AutoMLTab

def on_iteration_end(iteration, metrics):
    print(f"Iteration {iteration}: {metrics}")

def on_error(error):
    print(f"Error: {error}")

automl = AutoMLTab(
    trn_path="train.csv",
    label="target",
    callbacks={
        "on_iteration_end": on_iteration_end,
        "on_error": on_error
    }
)
```
