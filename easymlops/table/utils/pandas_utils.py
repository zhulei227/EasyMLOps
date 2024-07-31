import scipy.sparse as sp
import numpy as np
import pandas as pd

dataframe_type = pd.DataFrame


class PandasUtils(object):
    """
    pandas操作工具箱
    """

    @staticmethod
    def pd2csr(data: dataframe_type):
        try:
            if hasattr(data, "sparse"):
                sparse_matrix = sp.csr_matrix(data.sparse.to_coo().tocsr(), dtype=np.float32)
            else:
                sparse_matrix = sp.csr_matrix(data, dtype=np.float32)
        except Exception as e:
            print("can't trans to sparse,because {}\n".format(e))
            sparse_matrix = data
        return sparse_matrix

    @staticmethod
    def pd2dense(data: dataframe_type):
        return pd.DataFrame(data.values, index=data.index)

    @staticmethod
    def concat_duplicate_columns(dfs, keep="last"):
        """
        keep:last/相同column保留最后一个，first/相同column保留第一个
        """

        def flatten(nest_list: list):
            return [j for i in nest_list for j in flatten(i)] if isinstance(nest_list, list) else [nest_list]

        # 所有列保证相同的index
        df_start = dfs[0]
        total_columns = [df_start.columns.tolist()]
        for df in dfs[1:]:
            df.index = df_start.index
            total_columns.append(df.columns.tolist())
        # 删除重复列
        if keep == "first":
            dfs.reverse()
            total_columns.reverse()
        # 找出冲突列
        hint_columns = []
        for index in range(len(total_columns) - 1):
            current_columns = total_columns[index]
            remind_columns = flatten(total_columns[index + 1:])
            hint_columns.append(list(set(current_columns) & set(remind_columns)))
        hint_columns.append([])
        # 删除
        for index in range(len(dfs)):
            df = dfs[index]
            hint_column = hint_columns[index]
            for col in hint_column:
                del df[col]
        if keep == "first":
            dfs.reverse()
        return pd.concat(dfs, axis=1)
