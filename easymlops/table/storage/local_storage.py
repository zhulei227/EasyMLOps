import time
import datetime
from easymlops.table.core import *
import threading

lock = threading.Lock()


class LocalStorage(TablePipeObjectBase, threading.Thread):
    """
    存储模块，主要用于在transform_single阶段存储数据,其余过程均透传，这里默认对key创建了索引
    """

    def __init__(self, cols=None, db_name="local.db", table_name="temp", **kwargs):
        super().__init__(**kwargs)
        self.cols = cols
        if self.cols is None:
            self.cols = ["value"]
        self.table_columns = ["storage_key", "storage_transform_time"] + self.cols
        self.db_name = db_name
        self.table_name = table_name
        self.cache_data = []
        self._conn, self._cur = self.create_connect(self.db_name, self.table_name)
        self.stop_flag = False
        self.start()

    def reconnect(self):
        self._conn, self._cur = self.create_connect(self.db_name, self.table_name)

    def create_connect(self, db_name, table_name):
        _conn = None
        _cur = None
        # 创建链接
        try:
            import sqlite3
            _conn = sqlite3.connect(db_name, timeout=1, check_same_thread=False)
            _cur = _conn.cursor()
            columns_str = ",".join([item + " varchar" for item in self.table_columns])
            sql = f"create table if not exists {table_name} " \
                  f"({columns_str})"
            _cur.execute(sql)
        except Exception as e:
            print(f"connect to {db_name}:{table_name} exception:{e}")
        # 创建索引:storage_key
        if _conn is not None and _cur is not None:
            try:
                sql = f"create index if not exists index_{table_name}_storage_key on {table_name}(storage_key)"
                _cur.execute(sql)
            except Exception as e:
                print(f"create index storage_key exception:{e}")
        # 创建索引:storage_transform_time
        if _conn is not None and _cur is not None:
            try:
                sql = f"create index if not exists index_{table_name}_storage_transform_time " \
                      f"on {table_name}(storage_transform_time)"
                _cur.execute(sql)
            except Exception as e:
                print(f"create index storage_transform_time exception:{e}")
        return _conn, _cur

    def fit(self, s: dataframe_type, **kwargs):
        return self

    def transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        return s

    def udf_get_params(self) -> dict_type:
        return {"db_name": self.db_name, "table_name": self.table_name, "table_columns": self.table_columns}

    def udf_set_params(self, params: dict_type):
        self.db_name = params["db_name"]
        self.table_name = params["table_name"]
        self.reconnect()
        self.table_columns = params["table_columns"]

    def update_connect(self, db_name, table_name):
        self.db_name = db_name
        self.table_name = table_name
        self._conn, self._cur = self.create_connect(self.db_name, self.table_name)

    def transform_single(self, s: dict_type, storage_base_dict: dict_type = None, **kwargs) -> dict_type:
        s_ = copy.deepcopy(s)
        s_["storage_key"] = "default" if storage_base_dict is None \
            else storage_base_dict.get("key", "default")
        s_["storage_transform_time"] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        register_data = []
        for col in self.table_columns:
            register_data.append(str(s_.get(col, "")))
        self.cache_data.append(tuple(register_data))
        return s

    def select_key(self, key="default", limit=10):
        sql = f"select * from {self.table_name} where storage_key='{key}' limit {limit}"
        try:
            lock.acquire(True)
            self._cur.execute(sql)
            columns = [tp[0] for tp in self._cur.description]
            return pd.DataFrame(self._cur.fetchmany(size=limit), columns=columns)
        except Exception as e:
            print(f"[select] data exception:{e},try reconnect db")
            self._conn, self._cur = self.create_connect(self.db_name, self.table_name)
        finally:
            lock.release()

    def where(self, where_sql="", limit=10):
        sql = f"select * from {self.table_name} where {where_sql} limit {limit}"
        try:
            lock.acquire(True)
            self._cur.execute(sql)
            columns = [tp[0] for tp in self._cur.description]
            return pd.DataFrame(self._cur.fetchmany(size=limit), columns=columns)
        except Exception as e:
            print(f"[where] data exception:{e},try reconnect db")
            self._conn, self._cur = self.create_connect(self.db_name, self.table_name)
        finally:
            lock.release()

    def group_agg_where(self, group_by="", agg_sql="", where_sql="", limit=10):
        if type(group_by) == list:
            group_by = ",".join(group_by)
        if group_by is not None and len(group_by) > 0:
            group_by = " group by " + group_by
        if where_sql is not None and len(where_sql) > 0:
            where_sql = " where " + where_sql
        sql = f"select {agg_sql} from {self.table_name} {where_sql} {group_by} limit {limit}"
        try:
            lock.acquire(True)
            self._cur.execute(sql)
            columns = [tp[0] for tp in self._cur.description]
            return pd.DataFrame(self._cur.fetchall(), columns=columns)
        except Exception as e:
            print(f"[group_agg_where] data exception:{e},try reconnect db")
            self._conn, self._cur = self.create_connect(self.db_name, self.table_name)
        finally:
            lock.release()

    def sql(self, sql=""):
        try:
            lock.acquire(True)
            self._cur.execute(sql)
            columns = [tp[0] for tp in self._cur.description]
            return pd.DataFrame(self._cur.fetchall(), columns=columns)
        except Exception as e:
            print(f"[where] data exception:{e},try reconnect db")
            self._conn, self._cur = self.create_connect(self.db_name, self.table_name)
        finally:
            lock.release()

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
                    lock.acquire(True)
                    cols = ",".join(["?"] * len(self.table_columns))
                    sql = f"INSERT INTO {self.table_name} VALUES ({cols})"
                    self._cur.executemany(sql, saved_cache_data)
                    self._conn.commit()
                except Exception as e:
                    print(f"[insert] data exception:{e},try reconnect db")
                    self._conn, self._cur = self.create_connect(self.db_name, self.table_name)
                finally:
                    lock.release()
