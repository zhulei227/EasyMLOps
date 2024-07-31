from easymlops.table.core import *
import threading
import time
import datetime


class FaissStorage(TablePipeObjectBase, threading.Thread):
    def __init__(self, raw_data=None, search_top_k=3, create_index_param=None, create_index_measure=None,
                 tag="similarity", storage_path="./faiss_storage",
                 **kwargs):
        """
        :param raw_data: 原始数据，与index一一对应
        :param search_top_k: 检索最相近的top k数据
        :param create_index_param: 创建索引的方式
        :param create_index_measure: 评估相似度的方式
        :param tag: 输出前缀
        :storage_path: index和raw_data的存储地址
        """
        super().__init__(**kwargs)
        self.raw_data = raw_data
        self.search_top_k = search_top_k
        self.create_index_param = create_index_param
        self.create_index_measure = create_index_measure
        self.tag = tag
        self.storage_path = storage_path
        self.index = None
        self.stop_flag = False
        self.cache_data = []
        self.start()

    def udf_fit(self, s: dataframe_type, **kwargs):
        if self.raw_data is None:
            return self
        # 构建index
        import faiss
        self.index = faiss.index_factory(s.shape[1], self.create_index_param,
                                         eval(f"faiss.{self.create_index_measure}"))
        # 训练index
        if not self.index.is_trained:
            self.index.train(s.values)
        # 添加index
        self.index.add(s.values)
        return self

    def transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        return s

    def _get_cols(self):
        master_pipe = self.get_master_pipe()
        parent_pipe = self.get_parent_pipe()
        if master_pipe is None and parent_pipe is None:
            raise Exception("the module should band to a master pipe or add to a pipeline")
        cols = None
        if master_pipe is not None:
            cols = master_pipe.output_col_names
        if cols is None and parent_pipe is not None:
            cols = parent_pipe.output_col_names
        if cols is None:
            raise Exception("the master pipe or parent pipe should be trained")
        return cols

    def transform_single(self, s: dict_type, storage_base_dict: dict_type = None, add_index=False,
                         **kwargs) -> dict_type:
        """
        :param s: 生产输入
        :param add_index: 是否添加index
        :param storage_base_dict: 额外存储信息
        """
        if not add_index:
            return s
        cols = self._get_cols()
        input_dataframe = pd.DataFrame([s])[cols]
        storage_key = "default" if storage_base_dict is None \
            else str(storage_base_dict.get("key", "default"))
        row_data = "default" if storage_base_dict is None \
            else str(storage_base_dict.get("row_data", "default"))
        storage_transform_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        self.cache_data.append((input_dataframe.values, "|".join([row_data, storage_key, storage_transform_time])))
        return s

    def search(self, x, top_k=3):
        cols = self._get_cols()
        x_ = copy.deepcopy(x)
        if type(x) == dataframe_type:
            x_ = x_[cols]
        elif type(x) == dict_type:
            x_ = pd.DataFrame([x_])[cols]
        else:
            raise Exception("input x type should be dataframe or dict")
        top_k = top_k if top_k is not None else self.search_top_k
        D, I = self.index.search(x_.values, top_k)
        measure_dataframe = pd.DataFrame(D,
                                         columns=[f"{self.tag}_top_{k + 1}_measure" for k in range(top_k)],
                                         index=x_.index)
        raw_data_dataframe = pd.DataFrame(list(map(lambda x_tmp: list(map(lambda i: self.raw_data[i], x_tmp)), I)),
                                          columns=[f"{self.tag}_top_{k + 1}_raw_data" for k in
                                                   range(top_k)], index=x_.index)
        return pd.concat([raw_data_dataframe, measure_dataframe], axis=1)

    def save_index_raw_data(self):
        if self.storage_path is not None:
            import faiss
            faiss.write_index(self.index, self.storage_path + ".index")
            pickle.dump(self.raw_data, open(self.storage_path + ".raw_data", "wb"))

    def load_index_raw_data(self):
        if self.storage_path is not None:
            import faiss
            self.index = faiss.read_index(self.storage_path + ".index")
            self.raw_data = pickle.load(open(self.storage_path + ".raw_data", "rb"))

    def update_index_raw_data(self, path):
        self.save_index_raw_data()
        self.storage_path = path
        self.load_index_raw_data()

    def udf_get_params(self) -> dict_type:
        return {"search_top_k": self.search_top_k, "create_index_param": self.create_index_param,
                "create_index_measure": self.create_index_measure,
                "tag": self.tag, "storage_path": self.storage_path}

    def udf_set_params(self, params: dict_type):
        self.search_top_k = params["search_top_k"]
        self.create_index_param = params["create_index_param"]
        self.create_index_measure = params["create_index_measure"]
        self.tag = params["tag"]
        self.storage_path = params["storage_path"]
        self.load_index_raw_data()

    def set_stop(self):
        self.stop_flag = True

    def run(self):
        while True:
            if self.stop_flag:
                break
            time.sleep(0.001)
            if len(self.cache_data) > 0:
                saved_cache_data = self.cache_data
                self.cache_data = []
                try:
                    for item in saved_cache_data:
                        array, row_data = item
                        self.index.add(array)
                        self.raw_data.append(row_data)
                except Exception as e:
                    print(f"[insert] data exception:{e}")
