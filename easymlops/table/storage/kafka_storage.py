import time
import datetime
from easymlops.table.core import *
import threading
from func_timeout import func_set_timeout
import json


class KafkaStorage(TablePipeObjectBase, threading.Thread):
    """
    存储模块，主要用于在transform_single阶段存储数据,其余过程均透传
    """

    def __init__(self, cols=None, bootstrap_servers="127.0.0.1:9092", topic_name="topic_name",
                 write_batch_size=256, **kwargs):
        super().__init__(**kwargs)
        self.cols = cols
        if self.cols is None:
            self.cols = []
        self.write_batch_size = write_batch_size
        self.bootstrap_servers = bootstrap_servers
        self.topic_name = topic_name
        self.cache_data = []
        self.producer = None
        self.activate_connect()
        self.stop_flag = False
        self.start()

    def activate_connect(self):
        try:
            self._activate_connect()
        except:
            self.producer = self.create_connect(self.bootstrap_servers)

    @func_set_timeout(10)
    def _activate_connect(self):
        try:
            self.producer.flush()
        except:
            self.producer = self.create_connect(self.bootstrap_servers)

    @staticmethod
    def create_connect(bootstrap_servers):
        _producer = None
        from kafka import KafkaProducer
        try:
            _producer = KafkaProducer(bootstrap_servers=bootstrap_servers,
                                      value_serializer=lambda m: json.dumps(m).encode('ascii'))
        except Exception as e:
            print(f"connect to {bootstrap_servers} exception:{e}")
        return _producer

    def fit(self, s: dataframe_type, **kwargs):
        return self

    def transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        return s

    def udf_get_params(self) -> dict_type:
        return {"bootstrap_servers": self.bootstrap_servers, "topic_name": self.topic_name,
                "write_batch_size": self.write_batch_size, "cols": self.cols}

    def udf_set_params(self, params: dict_type):
        self.bootstrap_servers = params["bootstrap_servers"]
        self.topic_name = params["topic_name"]
        self.write_batch_size = params["write_batch_size"]
        self.cols = params["cols"]
        self.activate_connect()

    def update_connect(self, bootstrap_servers):
        self.bootstrap_servers = bootstrap_servers
        self.producer = self.create_connect(self.bootstrap_servers)

    def transform_single(self, s: dict_type, storage_base_dict: dict_type = None, **kwargs) -> dict_type:
        s_ = dict()
        for key, value in s.items():
            s_[f"{key}"] = f"{value}"
        s_[f"storage_key"] = "default" if storage_base_dict is None \
            else str(storage_base_dict.get("key", "default"))
        s_[f"storage_transform_time"] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        self.cache_data.append(s_)
        return s

    def set_stop(self):
        self.stop_flag = True

    def run(self):
        while True:
            if self.stop_flag:
                break
            time.sleep(0.001)
            if len(self.cache_data) > 0:
                self.activate_connect()
                saved_cache_data = self.cache_data
                self.cache_data = []
                try:
                    for index, data in enumerate(saved_cache_data):
                        self.producer.send(self.topic_name, data)
                        if (index + 1) % self.write_batch_size == 0:
                            # 每{batch_size}个样本保存一次
                            self.producer.flush()
                    self.producer.flush()
                except Exception as e:
                    print(f"insert data exception:{e}")
                    self.activate_connect()
