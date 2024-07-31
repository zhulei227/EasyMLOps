import time
import datetime
from easymlops.table.core import *
import threading
import random
from func_timeout import func_set_timeout


class HbaseStorage(TablePipeObjectBase, threading.Thread):
    """
    存储模块，主要用于在transform_single阶段存储数据,其余过程均透传，这里默认对key创建了索引
    """

    def __init__(self, cols=None, host="localhost", port=9090, table_name="table_name", cf_name="cf1",
                 write_batch_size=256, **kwargs):
        super().__init__(**kwargs)
        self.cols = cols
        if self.cols is None:
            self.cols = []
        self.write_batch_size = write_batch_size
        if type(host) == str:
            self.hosts = [host]
        else:
            self.hosts = host
        assert type(self.hosts) == list
        self.port = port
        self.table_name = table_name
        self.cf_name = cf_name
        self.cache_data = []
        self._conn, self._tab_key, self._tab_transform = None, None, None
        self.activate_connect()
        self.stop_flag = False
        self.start()

    def activate_connect(self):
        try:
            self._activate_connect()
        except:
            self._conn, self._tab_key, self._tab_transform = self.create_connect(self.hosts, self.port, self.table_name,
                                                                                 self.cf_name)

    @func_set_timeout(10)
    def _activate_connect(self):
        try:
            self._conn.tables()
            self._tab_key.row("u")
            self._tab_transform.row("u")
        except:
            self._conn, self._tab_key, self._tab_transform = self.create_connect(self.hosts, self.port, self.table_name,
                                                                                 self.cf_name)

    @staticmethod
    def create_connect(hosts, port, table_name, cf_name):
        _conn = None
        _tab_key = None
        _tab_transform = None
        # 创建链接
        random.shuffle(hosts)
        for host in hosts:
            try:
                import happybase
                _conn = happybase.Connection(host=host, port=port)
                _conn.open()
                # 创建key表
                if table_name in [tab.decode("utf8") for tab in _conn.tables()]:
                    _tab_key = _conn.table(table_name)
                else:
                    _conn.create_table(table_name, {cf_name: dict(max_versions=10)})
                    _tab_key = _conn.table(table_name)
                # 创建transform表
                if table_name + "_transform" in [tab.decode("utf8") for tab in _conn.tables()]:
                    _tab_transform = _conn.table(table_name + "_transform")
                else:
                    _conn.create_table(table_name + "_transform", {cf_name: dict(max_versions=10)})
                    _tab_transform = _conn.table(table_name + "_transform")
                print(f"connect to {host}:{port}/{table_name} success!")
                break
            except Exception as e:
                print(f"connect to {host}:{port}/{table_name} exception:{e}")
        return _conn, _tab_key, _tab_transform

    def fit(self, s: dataframe_type, **kwargs):
        return self

    def transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        return s

    def udf_get_params(self) -> dict_type:
        return {"hosts": self.hosts, "port": self.port, "table_name": self.table_name, "cf_name": self.cf_name,
                "write_batch_size": self.write_batch_size, "cols": self.cols}

    def udf_set_params(self, params: dict_type):
        self.hosts = params["hosts"]
        self.port = params["port"]
        self.table_name = params["table_name"]
        self.cf_name = params["cf_name"]
        self.write_batch_size = params["write_batch_size"]
        self.cols = params["cols"]
        self.activate_connect()

    def update_connect(self, host, port, table_name, cf_name):
        if type(host) == str:
            hosts = [host]
        else:
            hosts = host
        assert type(hosts) == list
        self.hosts = hosts
        self.port = port
        self.table_name = table_name
        self.cf_name = cf_name
        self.activate_connect()

    def transform_single(self, s: dict_type, storage_base_dict: dict_type = None, **kwargs) -> dict_type:
        s_ = dict()
        for key, value in s.items():
            s_[f"{self.cf_name}:{key}"] = f"{value}"
        s_[f"{self.cf_name}:storage_key"] = "default" if storage_base_dict is None \
            else str(storage_base_dict.get("key", "default"))
        s_[f"{self.cf_name}:storage_transform_time"] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        self.cache_data.append(s_)
        return s

    def select_key(self, key="default"):
        self.activate_connect()
        df = pd.DataFrame(columns=["storage_key", "storage_transform_time"] + self.cols)
        try:
            results = self._tab_key.row(f"{key}")
            new_result = dict()
            for key, value in results.items():
                new_result[key.decode("utf8").split(":")[-1]] = value.decode("utf8")
            df = df.append(new_result, ignore_index=True)
            return df
        except:
            self.activate_connect()
            return df

    def select_time(self, start_time="", stop_time="", limit=10, scan_filter=None):
        self.activate_connect()
        df = pd.DataFrame(columns=["storage_key", "storage_transform_time"] + self.cols)
        try:
            results = self._tab_transform.scan(row_start=start_time, row_stop=stop_time, limit=limit,
                                               filter=scan_filter)
            for row_key, row_data in results:
                new_result = dict()
                for key, value in row_data.items():
                    new_result[key.decode("utf8").split(":")[-1]] = value.decode("utf8")
                df = df.append(new_result, ignore_index=True)
            return df
        except:
            self.activate_connect()
            return df

    def scan(self, row_start=None, row_stop=None, row_prefix=None,
             columns=None, scan_filter=None, timestamp=None,
             include_timestamp=False, batch_size=1000, scan_batching=None,
             limit=10, sorted_columns=False, reverse=False):
        self.activate_connect()
        df = pd.DataFrame(columns=["storage_key", "storage_transform_time"] + self.cols)
        try:
            results = self._tab_key.scan(row_start=row_start, row_stop=row_stop, row_prefix=row_prefix,
                                         columns=columns, filter=scan_filter, timestamp=timestamp,
                                         include_timestamp=include_timestamp, batch_size=batch_size,
                                         scan_batching=scan_batching,
                                         limit=limit, sorted_columns=sorted_columns, reverse=reverse)
            for row_key, row_data in results:
                new_result = dict()
                for key, value in row_data.items():
                    new_result[key.decode("utf8").split(":")[-1]] = value.decode("utf8")
                df = df.append(new_result, ignore_index=True)
            return df
        except:
            self.activate_connect()
            return df

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
                    bat_key = self._tab_key.batch()
                    bat_transform = self._tab_transform.batch()
                    for index, data in enumerate(saved_cache_data):
                        row_key = data.get(f"{self.cf_name}:storage_key")
                        row_transform = data.get(f"{self.cf_name}:storage_transform_time")
                        bat_key.put(f"{row_key}", data)
                        bat_transform.put(f"{row_transform}", data)
                        if (index + 1) % self.write_batch_size == 0:
                            # 每{batch_size}个样本保存一次
                            bat_key.send()
                            bat_transform.send()
                    bat_key.send()
                    bat_transform.send()
                except:
                    self.activate_connect()
