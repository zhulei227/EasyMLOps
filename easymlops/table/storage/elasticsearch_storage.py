import time
import datetime
from easymlops.table.core import *
import threading
from func_timeout import func_set_timeout


class ElasticSearchStorage(TablePipeObjectBase, threading.Thread):
    """
    存储模块，主要用于在transform_single阶段存储数据,其余过程均透传,es默认对所有列都创建索引
    """

    def __init__(self, cols=None, index_name="default_index", es_args=None, es_kwargs=None, **kwargs):
        super().__init__(**kwargs)
        self.cols = cols
        if self.cols is None:
            self.cols = []
        self.index_name = index_name
        self.es_args = es_args
        self.es_kwargs = es_kwargs
        self.cache_data = []
        self.es = None
        self.activate_connect()
        self.stop_flag = False
        self.start()

    def activate_connect(self):
        try:
            self._activate_connect()
        except:
            self.es = self.create_connect(self.es_args, self.es_kwargs)

    @func_set_timeout(10)
    def _activate_connect(self):
        try:
            self.es.index(index=self.index_name, id="check_connect", body={})
        except:
            self.es = self.create_connect(self.es_args, self.es_kwargs)

    @staticmethod
    def create_connect(es_args, es_kwargs):
        if es_args is None:
            es_args = ()
        else:
            es_args = tuple(es_args)
        if es_kwargs is None:
            es_kwargs = dict()
        _es = None
        try:
            from elasticsearch import Elasticsearch
            _es = Elasticsearch(*es_args, **es_kwargs)
        except Exception as e:
            print(f"connect to es exception:{e}")
        return _es

    def fit(self, s: dataframe_type, **kwargs):
        return self

    def transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        return s

    def udf_get_params(self) -> dict_type:
        return {"index_name": self.index_name, "es_args": self.es_args, "es_kwargs": self.es_kwargs, "cols": self.cols}

    def udf_set_params(self, params: dict_type):
        self.index_name = params["index_name"]
        self.es_args = params["es_args"]
        self.es_kwargs = params["es_kwargs"]
        self.cols = params["cols"]
        self.activate_connect()

    def update_connect(self, es_args, es_kwargs):
        self.es_args = es_args
        self.es_kwargs = es_kwargs
        self.activate_connect()

    def transform_single(self, s: dict_type, storage_base_dict: dict_type = None, **kwargs) -> dict_type:
        s_ = copy.deepcopy(s)
        s_["storage_key"] = "default" if storage_base_dict is None \
            else str(storage_base_dict.get("key", "default"))
        s_["storage_transform_time"] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        data = {
            "_index": self.index_name,
            "_type": "_doc",
            "_id": f"{s_['storage_key']}",
            "_source": s_
        }
        self.cache_data.append(data)
        return s

    def select_key(self, key="default"):
        self.activate_connect()
        df = pd.DataFrame(columns=["storage_key", "storage_transform_time"] + self.cols)
        try:
            results = self.es.get(index=self.index_name, id=key)
            df = df.append(results["_source"], ignore_index=True)
            return df
        except:
            self.activate_connect()
            return df

    def query(self, body):
        self.activate_connect()
        df = pd.DataFrame()
        try:
            results = self.es.search(index=self.index_name, body=body)["hits"]["hits"]
            for rs in results:
                rst_source = rs["_source"]
                if rs.get("_score") is not None:
                    rst_source["hit_score_"] = rs["_score"]
                df = df.append(rst_source, ignore_index=True)
        except:
            self.activate_connect()
        return df

    def search(self, body):
        self.activate_connect()
        try:
            return self.es.search(index=self.index_name, body=body)
        except:
            return {}

    def set_stop(self):
        self.stop_flag = True

    def run(self):
        from elasticsearch import helpers
        while True:
            if self.stop_flag:
                break
            time.sleep(0.001)
            if len(self.cache_data) > 0:
                self.activate_connect()
                saved_cache_data = self.cache_data
                self.cache_data = []
                try:
                    helpers.bulk(self.es, saved_cache_data)
                except Exception as e:
                    print(f"bulk data exception:{e}")
                    self.activate_connect()
