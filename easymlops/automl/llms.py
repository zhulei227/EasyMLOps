"""
llm交互基础工具,只需要实现chat接口
"""
import requests
import uuid


class LLM(object):
    def __init__(self):
        pass

    def chat(self, message, keep_history_message_len):
        pass


class OllamaLLM(LLM):
    def __init__(self, model="qwen2:1.5b"):
        super().__init__()
        self.model = model

    def chat(self, messages, keep_history_message_len=100):
        import ollama
        try:
            response = ollama.chat(model=self.model,
                                   messages=messages[-keep_history_message_len:],
                                   stream=False)
            return response["message"]["content"]
        except Exception as e:
            print(f"call ollma {self.model} chat error:{e}")
            return None


class SparkLLM(LLM):
    def __init__(self, url="https://spark-api-open.xf-yun.com/v1/chat/completions",
                 api_password="SecMMcUZNlbZEVnsuqodLZ:uVOWzputlaOpuxNmVJVVL", model="lite"):
        """
        :param model:lite,x1
        """
        super().__init__()
        self.url = url
        self.headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_password}"}
        self.model = model
        self.session = requests.session()

    def chat(self, messages, keep_history_message_len=100):
        try:
            response = self.session.post(self.url, json={"model": self.model,
                                                         "messages": messages[-keep_history_message_len:],
                                                         "stream": False,
                                                         "request_id": str(uuid.uuid4())},
                                         headers=self.headers).json()
            assistant_content = response.get("choices", [{"message": {"content": None}}])[0].get(
                "message").get("content")
            return assistant_content
        except Exception as e:
            print(f"call spark {self.model} chat error:{e}")
            return None


class ZhiPuLLM(LLM):
    def __init__(self, url="https://open.bigmodel.cn/api/paas/v4/chat/completions",
                 api_password="cc13655e7b7d240e2a06b4c81312cfddd1.4ZctwJ2EH0cDDQTIW", model="glm-4-9b"):
        """
        :param model:lite,x1
        """
        super().__init__()
        self.url = url
        self.headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_password}"}
        self.model = model
        self.session = requests.session()

    def chat(self, messages, keep_history_message_len=100):
        try:
            response = self.session.post(self.url, json={"model": self.model,
                                                         "messages": messages[-keep_history_message_len:],
                                                         "stream": False,
                                                         "request_id": str(uuid.uuid4())},
                                         headers=self.headers).json()
            assistant_content = response.get("choices", [{"message": {"content": None}}])[0].get(
                "message").get("content")
            return assistant_content
        except Exception as e:
            print(f"call zhipu {self.model} chat error:{e}")
            return None


class KimiLLM(LLM):
    def __init__(self, api_key="sk-GoZH40ma2riU6jetCEa2ANNdQuNyXYRMAHIik3RszkXFdKAQHUU",
                 base_url="https://api.moonshot.cn/v1", model="moonshot-v1-8k"):
        """
        :param model:lite,x1
        """
        super().__init__()
        from openai import OpenAI
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.model = model

    def chat(self, messages, keep_history_message_len=100):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages[-keep_history_message_len:]
            )
            assistant_content = response.choices[0].message.content
            return assistant_content
        except Exception as e:
            print(f"call kimi {self.model} chat error:{e}")
            return None

# print(SparkLLM(model="lite").chat(messages=[{"role": "user", "content": "你是谁?"}]))

# print(OllamaLLM().chat(messages=[{"role": "user", "content": "你是谁?"}]))

# print(KimiLLM().chat(messages=[{"role": "user", "content": "你是谁?"}]))
