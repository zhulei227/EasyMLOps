"""
相似文本向量检索
"""
from easymlops.table.core import *


class ElasticSearchSimilarity(TablePipeObjectBase):
    def __init__(self, col="all", search_top_k=3, es_args=None,
                 index_name="default_index", es_kwargs=None,
                 drop_input=True, add_production_data=False, tag="similarity", **kwargs):
        """
        :param col: 需要被存储的col
        :param search_top_k: 检索最相近的top k数据
        :param es_args: es入参args
        :param es_kwargs: es入参kwargs
        :param drop_input: 是否删除输入数据，默认删除
        :param tag: 输出前缀
        :param add_production_data: 是否保存生产数据到索引
        """

        super().__init__(**kwargs)
        self.col = col
        self.search_top_k = search_top_k
        self.index_name = index_name
        if es_args is None:
            es_args = ()
        else:
            es_args = tuple(es_args)
        if es_kwargs is None:
            es_kwargs = dict()
        self.es_args = es_args
        self.es_kwargs = es_kwargs
        self.drop_input = drop_input
        self.add_production_data = add_production_data
        self.tag = tag
        self.index = None
        self.es = None
        self.activate_connect()

    @staticmethod
    def create_connect(es_args, es_kwargs):
        _es = None
        try:
            from elasticsearch import Elasticsearch
            _es = Elasticsearch(*es_args, **es_kwargs)
        except Exception as e:
            print(f"connect to es exception:{e}")
        return _es

    def activate_connect(self):
        try:
            self.es.index(index=self.index_name, id="check_connect", body={})
        except:
            self.es = self.create_connect(self.es_args, self.es_kwargs)

    def before_fit(self, s: dataframe_type, **kwargs) -> dataframe_type:
        s = super().before_fit(s)
        if self.check_list_same([self.col], s.columns.tolist()):
            return s
        else:
            return s[[self.col]]

    def udf_fit(self, s: dataframe_type, **kwargs):
        from elasticsearch import helpers
        acs = []
        for record in s.to_dict("records"):
            data = {
                "_index": self.index_name,
                "_type": "_doc",
                "_source": record
            }
            acs.append(data)
        self.activate_connect()
        helpers.bulk(self.es, acs)

    def apply_single(self, s: dict_type):
        # 搜索最相近的几个值
        # 查询长度默认限制为1024
        body = {
            '_source': [self.col],
            'query': {
                'match': {
                    f'{self.col}': f'{s[self.col][:1024]}'
                }
            },
            'sort': {
                '_score': {
                    'order': 'desc'
                }
            },
            'from': 0,
            'size': self.search_top_k
        }
        if self.es is None:
            rst = {}
        else:
            rst = dict()
            try:
                for k, raw_data in enumerate(self.es.search(index=self.index_name, body=body)["hits"]["hits"]):
                    rst[f"{self.tag}_top_{k + 1}_raw_data"] = raw_data["_source"][self.col]
                    rst[f"{self.tag}_top_{k + 1}_measure"] = raw_data["_score"]
            except:
                pass
        # 空值处理
        for k in range(self.search_top_k):
            if f"{self.tag}_top_{k + 1}_raw_data" not in rst:
                rst[f"{self.tag}_top_{k + 1}_raw_data"] = "nan"
            if f"{self.tag}_top_{k + 1}_measure" not in rst:
                rst[f"{self.tag}_top_{k + 1}_measure"] = 0
        return rst

    def udf_transform(self, s: dataframe_type, drop_input=None, **kwargs) -> dataframe_type:
        drop_input = drop_input if drop_input is not None else self.drop_input
        rst = []
        for record in s.to_dict("records"):
            rst.append(self.apply_single(record))
        rst_df = pd.DataFrame(rst, index=s.index)
        sorted_cols = []
        for k in range(self.search_top_k):
            sorted_cols.append(f"{self.tag}_top_{k + 1}_raw_data")
            sorted_cols.append(f"{self.tag}_top_{k + 1}_measure")
        rst_df = rst_df[sorted_cols]
        if drop_input:
            return rst_df
        else:
            return pd.concat([s, rst_df], axis=1)

    def transform_single(self, s: dict_type, drop_input=None, add_production_data=None,
                         **kwargs) -> dict_type:
        drop_input = drop_input if drop_input is not None else self.drop_input
        add_production_data = add_production_data if add_production_data is not None else self.add_production_data
        if add_production_data and self.es is not None:
            self.es.index(index=self.index_name, body=s)
        rst = self.apply_single(s)
        if not drop_input:
            rst.update(s)
        return rst

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

    def udf_get_params(self) -> dict_type:
        return {"col": self.col, "search_top_k": self.search_top_k, "es_args": self.es_args,
                "es_kwargs": self.es_kwargs, "drop_input": self.drop_input,
                "add_production_data": self.add_production_data, "tag": self.tag}

    def udf_set_params(self, params: dict_type):
        self.col = params["col"]
        self.search_top_k = params["search_top_k"]
        self.es_args = params["es_args"]
        self.es_kwargs = params["es_kwargs"]
        self.drop_input = params["drop_input"]
        self.add_production_data = params["add_production_data"]
        self.tag = params["tag"]


class FaissSimilarity(TablePipeObjectBase):
    def __init__(self, cols="all", raw_data=None, search_top_k=3, create_index_param=None, create_index_measure=None,
                 drop_input=True, add_production_data=False, tag="similarity", **kwargs):
        """
        :param cols: 需要被存储的cols
        :param raw_data: 原始数据，与index一一对应
        :param search_top_k: 检索最相近的top k数据
        :param create_index_param: 创建索引的方式
        :param create_index_measure: 评估相似度的方式
        :param drop_input: 是否删除输入数据，默认删除
        :param tag: 输出前缀
        :param add_production_data: 是否保存生产数据到索引
        """

        super().__init__(**kwargs)
        self.cols = cols
        assert type(raw_data) == list
        self.raw_data = raw_data
        self.search_top_k = search_top_k
        self.create_index_param = create_index_param
        self.create_index_measure = create_index_measure
        self.drop_input = drop_input
        self.add_production_data = add_production_data
        self.tag = tag
        self.index = None

    def before_fit(self, s: dataframe_type, **kwargs) -> dataframe_type:
        s = super().before_fit(s)
        if type(self.cols) != list:
            self.cols = s.columns.tolist()
        if self.check_list_same(self.cols, s.columns.tolist()):
            return s
        else:
            return s[self.cols]

    def udf_fit(self, s: dataframe_type, **kwargs):
        # 构建index
        import faiss
        self.index = faiss.index_factory(s.shape[1], self.create_index_param,
                                         eval(f"faiss.{self.create_index_measure}"))
        # 训练index
        if not self.index.is_trained:
            self.index.train(s.values)
        # 添加index
        self.index.add(s.values)

    def udf_transform(self, s: dataframe_type, top_k=None, drop_input=None, **kwargs) -> dataframe_type:
        top_k = top_k if top_k is not None else self.search_top_k
        D, I = self.index.search(s.values, top_k)
        measure_dataframe = pd.DataFrame(D,
                                         columns=[f"{self.tag}_top_{k + 1}_measure" for k in range(top_k)],
                                         index=s.index)
        raw_data_dataframe = pd.DataFrame(list(map(lambda x: list(map(lambda i: self.raw_data[i], x)), I)),
                                          columns=[f"{self.tag}_top_{k + 1}_raw_data" for k in
                                                   range(top_k)], index=s.index)
        drop_input = drop_input if drop_input is not None else self.drop_input
        if drop_input:
            return pd.concat([raw_data_dataframe, measure_dataframe], axis=1)
        else:
            return pd.concat([s, raw_data_dataframe, measure_dataframe], axis=1)

    def transform_single(self, s: dict_type, top_k=None, drop_input=None, add_production_data=None,
                         **kwargs) -> dict_type:
        input_dataframe = pd.DataFrame([s])[self.cols]
        output_rst = self.transform(input_dataframe, top_k=top_k, drop_input=drop_input).to_dict("records")[0]
        add_production_data = add_production_data if add_production_data is not None else self.add_production_data
        if add_production_data:
            raw_data = kwargs.get("raw_data")
            self.index.add(input_dataframe.values)
            self.raw_data.append(raw_data)
        return output_rst

    def search(self, x, top_k=3):
        x_ = copy.deepcopy(x)
        if type(x) == dataframe_type:
            search_df = self.transform(self.transform_all_parent(x_), top_k=top_k, drop_input=True)
            search_df.index = x.index
            return pd.concat([x, search_df], axis=1)
        elif type(x) == dict_type:
            search_dict = self.transform_single(self.transform_single_all_parent(x), top_k=top_k, drop_input=True,
                                                add_production_data=False)
            return_search = copy.deepcopy(x)
            return_search.update(search_dict)
            return return_search
        else:
            raise Exception("input x type should be dataframe or dict")

    def save_index_raw_data(self, path):
        """
        :param path: 保存index和raw_data
        """
        import faiss
        faiss.write_index(self.index, path + ".index")
        pickle.dump(self.raw_data, open(path + ".raw_data", "wb"))

    def load_index_raw_data(self, path):
        """
        :param path: 加载index和raw_data
        """
        import faiss
        self.index = faiss.read_index(path + ".index")
        self.raw_data = pickle.load(open(path + ".raw_data", "rb"))

    def udf_get_params(self) -> dict_type:
        return {"cols": self.cols, "search_top_k": self.search_top_k, "create_index_param": self.create_index_param,
                "create_index_measure": self.create_index_measure, "drop_input": self.drop_input,
                "add_production_data": self.add_production_data, "tag": self.tag}

    def udf_set_params(self, params: dict_type):
        self.cols = params["cols"]
        self.search_top_k = params["search_top_k"]
        self.create_index_param = params["create_index_param"]
        self.create_index_measure = params["create_index_measure"]
        self.drop_input = params["drop_input"]
        self.add_production_data = params["add_production_data"]
        self.tag = params["tag"]
