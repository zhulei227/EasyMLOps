from easymlops.table.core import *
import numpy as np
import lightgbm as lgb

pd.options.mode.copy_on_write = True


class EncodingBase(TablePipeObjectBase):
    """
    编码操作基类。
    
    所有编码类的父类，提供列选择和数据编码的标准框架。
    
    Example:
        >>> class MyEncoding(EncodingBase):
        ...     def apply_function_series(self, col, x):
        ...         return x
    """
    
    def __init__(self, cols="all", y=None, **kwargs):
        """
        初始化编码操作。
        
        Args:
            cols: 要操作的列
            y: 目标变量（用于目标编码等）
            **kwargs: 其他父类参数
        """
        super().__init__(**kwargs)
        self.cols = cols
        self.y = y

    def apply_function_series(self, col: str, x: series_type):
        """
        对Series应用编码函数。
        
        Args:
            col: 当前列名称
            x: 当前列对应的值
            
        Returns:
            编码后的Series
        """
        raise Exception("need to implement")

    def apply_function_single(self, col: str, x):
        """
        对单个值应用编码函数。
        
        Args:
            col: 当前列名称
            x: 当前列对应的值
            
        Returns:
            编码后的值
        """
        raise Exception("need to implement")

    @staticmethod
    def extract_dict(s: dict_type, keys: list):
        """
        从字典中提取指定键的值。
        
        Args:
            s: 源字典
            keys: 要提取的键列表
            
        Returns:
            新字典
        """
        new_s = dict()
        for key in keys:
            new_s[key] = s[key]
        return new_s

    def before_fit(self, s: dataframe_type, **kwargs) -> dataframe_type:
        """
        fit前处理，确定要操作的列。
        
        Args:
            s: 输入数据
            **kwargs: 其他参数
            
        Returns:
            处理后的数据
        """
        s = super().before_fit(s, **kwargs)
        if str(self.cols) == "all" or self.cols is None or (type(self.cols) == list and len(self.cols) == 0):
            self.cols = []
            for col in s.columns.tolist():
                self.cols.append((col, col))
        else:
            if type(self.cols) == list:
                if type(self.cols[0]) == tuple or type(self.cols[0]) == list:
                    pass
                else:
                    new_cols = []
                    for col in self.cols:
                        new_cols.append((col, col))
                    self.cols = new_cols
            else:
                raise Exception("cols should be None,'all' or a list")
        return s

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        """对数据进行编码转换。"""
        for col, new_col in self.cols:
            if col in s.columns:
                s[new_col] = self.apply_function_series(col, s[col])
        return s

    def udf_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        """对单条数据进行编码转换。"""
        for col, new_col in self.cols:
            if col in s.keys():
                s[new_col] = self.apply_function_single(col, s[col])
        return s

    def udf_get_params(self) -> dict_type:
        """获取参数。"""
        params = {"cols": self.cols}
        return params

    def udf_set_params(self, params: dict):
        """设置参数。"""
        self.cols = params["cols"]


class LGBMLeafEncoder(EncodingBase):
    """
    使用LightGBM决策树进行叶子节点特征编码。

    训练一个LightGBM决策树模型，将样本映射到叶子节点位置，
    输出为稀疏向量形式：每个叶子节点对应一个维度，命中为1，未命中为0。

    这种编码方式可以捕捉原始特征之间的非线性组合关系，
    类似于将原始特征空间划分为多个子空间，每个子空间用一个维度表示。

    Example:
        >>> from easymlops.table.encoding import LGBMLeafEncoder
        >>> from easymlops.table import TablePipeLine
        >>> import pandas as pd
        >>> pipe = LGBMLeafEncoder(y=label, n_estimators=10, max_depth=3)
        >>> result = pipe.fit(df).transform(df)
    """

    def __init__(self, y=None, n_estimators=10, max_depth=5, learning_rate=0.1,
                 num_leaves=31, min_child_samples=20, cols="all", **kwargs):
        """
        初始化LightGBM叶子节点编码器。

        Args:
            y: 标签数据，用于训练决策树
            n_estimators: 决策树数量
            max_depth: 决策树最大深度
            learning_rate: 学习率
            num_leaves: 每棵树的叶子节点数量
            min_child_samples: 叶子节点最小样本数
            cols: 要操作的列，支持 "all"、列名列表或 (原列名, 新列名) 元组列表
            **kwargs: 其他父类参数
        """
        super().__init__(cols=cols, **kwargs)
        self.y = y
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.lgb_model = None
        self.num_trees = 0
        self.num_leaves_per_tree = []
        self.feature_names = []
        self.feature_name_to_idx = {}
        self.leaf_descriptions = {}
        self._feature_unique_values = {}

    def before_fit(self, s: dataframe_type, **kwargs) -> dataframe_type:
        """
        fit前处理，确定要操作的列。

        Args:
            s: 输入数据
            **kwargs: 其他参数

        Returns:
            处理后的数据
        """
        s = super().before_fit(s, **kwargs)
        return s

    def udf_fit(self, s: dataframe_type, **kwargs):
        """
        训练LightGBM决策树模型。

        Args:
            s: 输入数据，DataFrame
            **kwargs: 其他参数

        Returns:
            self
        """
        if str(self.cols) == "all" or self.cols is None or (type(self.cols) == list and len(self.cols) == 0):
            feature_cols = s.columns.tolist()
        else:
            if isinstance(self.cols, list) and len(self.cols) > 0:
                if isinstance(self.cols[0], (tuple, list)):
                    feature_cols = [c[0] for c in self.cols]
                else:
                    feature_cols = self.cols
            else:
                feature_cols = s.columns.tolist()

        X = s[feature_cols].values
        if self.y is None:
            raise ValueError("y must be provided for training")

        self.feature_names = feature_cols
        for idx, name in enumerate(self.feature_names):
            self.feature_name_to_idx[name] = idx

        params = {
            "objective": "regression",
            "boosting_type": "gbdt",
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "n_estimators": self.n_estimators,
            "min_child_samples": self.min_child_samples,
            "verbosity": -1,
            "feature_name": feature_cols
        }

        self.lgb_model = lgb.LGBMRegressor(**params)
        self.lgb_model.fit(X, self.y)

        for col in feature_cols:
            unique_vals = s[col].dropna().unique()
            if len(unique_vals) <= 20:
                self._feature_unique_values[col] = sorted(unique_vals.tolist())

        self._extract_leaf_info()
        self._build_leaf_descriptions()

        return self

    def _extract_leaf_info(self):
        """
        提取叶子节点信息。
        """
        model_dict = self.lgb_model.booster_.dump_model()

        self.num_trees = len(model_dict["tree_info"])
        self.num_leaves_per_tree = []
        self.leaf_descriptions = {}

        tree_info = model_dict["tree_info"]
        current_leaf_idx = 0

        for tree_idx, tree_data in enumerate(tree_info):
            tree_structure = tree_data["tree_structure"]
            num_leaves = tree_data.get("num_leaves", 31)

            self.num_leaves_per_tree.append(num_leaves)

            self._traverse_tree(tree_structure, tree_idx, current_leaf_idx, [])
            current_leaf_idx += num_leaves

    def _traverse_tree(self, tree, tree_idx, leaf_offset, path):
        """
        遍历树结构，提取每个叶子节点的路径信息。
        """
        if "left_child" in tree:
            left_child = tree["left_child"]
            right_child = tree["right_child"]

            threshold = tree.get("threshold")
            split_feature_idx = tree.get("split_feature")
            split_feature_name = self.feature_names[split_feature_idx] if split_feature_idx is not None else "unknown"
            decision_type = tree.get("decision_type", "<=")

            left_path = path + [{
                "feature": split_feature_name,
                "condition": "<=",
                "threshold": threshold
            }]

            right_path = path + [{
                "feature": split_feature_name,
                "condition": ">",
                "threshold": threshold
            }]

            self._traverse_tree(left_child, tree_idx, leaf_offset, left_path)
            self._traverse_tree(right_child, tree_idx, leaf_offset, right_path)
        else:
            leaf_idx = tree.get("leaf_index", 0)
            global_leaf_idx = leaf_offset + leaf_idx
            self.leaf_descriptions[global_leaf_idx] = {
                "tree_idx": tree_idx,
                "leaf_idx": leaf_idx,
                "path": path
            }

    def _build_leaf_descriptions(self):
        """
        构建叶子节点的描述信息。

        描述形如：
        - 数值特征: feature_name<=5 or feature_name>6
        - 离散特征: feature_name in (1,2,3) or feature_name not in (1,2,3)

        对于离散特征（类别较少），将连续条件转换为集合包含关系描述。
        """
        self.describe_dict = {}

        for leaf_idx, leaf_info in self.leaf_descriptions.items():
            path = leaf_info["path"]
            descriptions = []

            for condition in path:
                feature = condition["feature"]
                thresh = condition["threshold"]
                branch = condition["condition"]

                unique_vals = self._feature_unique_values.get(feature)

                if unique_vals is not None and len(unique_vals) <= 20:
                    if branch == "<=":
                        left_vals = [v for v in unique_vals if v <= thresh]
                        if len(left_vals) == len(unique_vals):
                            desc = f"{feature} in ({','.join(map(str, unique_vals))})"
                        elif len(left_vals) <= 5:
                            desc = f"{feature} in {tuple(left_vals)}"
                        else:
                            desc = f"{feature}<={thresh}"
                    elif branch == ">":
                        right_vals = [v for v in unique_vals if v > thresh]
                        if len(right_vals) == 0:
                            desc = f"{feature} not in ({','.join(map(str, unique_vals))})"
                        elif len(right_vals) <= 5:
                            desc = f"{feature} not in {tuple(sorted(right_vals))}"
                        else:
                            desc = f"{feature}>{thresh}"
                    else:
                        desc = f"{feature} {branch} {thresh}"
                else:
                    if branch == "<=":
                        desc = f"{feature}<={thresh}"
                    elif branch == ">":
                        desc = f"{feature}>{thresh}"
                    else:
                        desc = f"{feature} {branch} {thresh}"

                descriptions.append(desc)

            full_description = " and ".join(descriptions) if descriptions else "leaf_root"
            self.describe_dict[leaf_idx] = full_description

    def describe(self):
        """
        输出叶子节点的描述信息。

        返回一个字典，key为叶子节点索引，value为该叶子节点对应的特征组合条件描述。

        Returns:
            dict: 叶子节点索引到特征组合描述的映射

        Example:
            >>> pipe = LGBMLeafEncoder(y=label, n_estimators=10)
            >>> pipe.fit(df)
            >>> descriptions = pipe.describe()
            >>> for idx, desc in descriptions.items():
            ...     print(f"Leaf {idx}: {desc}")
        """
        return self.describe_dict

    def apply_function_series(self, col: str, x: series_type) -> dataframe_type:
        """
        对Series应用编码函数（需要预先调用fit）。

        Args:
            col: 当前列名称
            x: 当前列对应的值

        Returns:
            DataFrame: 叶子节点命中向量
        """
        raise Exception("Please use transform() method instead")

    def apply_function_single(self, col: str, x):
        """
        对单个值应用编码函数（需要预先调用fit）。

        Args:
            col: 当前列名称
            x: 当前列对应的值
        """
        raise Exception("Please use transform_single() method instead")

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        """
        将样本转换为叶子节点命中向量。

        Args:
            s: 输入数据，DataFrame

        Returns:
            DataFrame: 叶子节点命中向量，每列对应一个叶子节点，值为0或1
        """
        if self.lgb_model is None:
            raise ValueError("Model not fitted yet. Please call fit() first.")

        if str(self.cols) == "all" or self.cols is None or (type(self.cols) == list and len(self.cols) == 0):
            feature_cols = s.columns.tolist()
        else:
            if isinstance(self.cols, list) and len(self.cols) > 0:
                if isinstance(self.cols[0], (tuple, list)):
                    feature_cols = [c[0] for c in self.cols]
                else:
                    feature_cols = self.cols
            else:
                feature_cols = s.columns.tolist()

        X = s[feature_cols].values

        leaf_indices = self.lgb_model.predict(X)

        total_leaves = sum(self.num_leaves_per_tree)

        result = np.zeros((len(s), total_leaves), dtype=np.int32)

        for i, leaf_idx in enumerate(leaf_indices):
            tree_idx = 0
            leaf_offset = 0
            current_leaf_idx = int(leaf_idx)

            for num_leaves in self.num_leaves_per_tree:
                if current_leaf_idx < num_leaves:
                    result[i, leaf_offset + current_leaf_idx] = 1
                    break
                leaf_offset += num_leaves
                current_leaf_idx -= num_leaves
                tree_idx += 1

        leaf_cols = [f"leaf_{j}" for j in range(total_leaves)]
        result_df = pd.DataFrame(result, columns=leaf_cols, index=s.index)

        return result_df

    def udf_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        """
        单条数据的叶子节点编码。

        Args:
            s: 输入数据，字典

        Returns:
            dict: 叶子节点命中向量
        """
        if self.lgb_model is None:
            raise ValueError("Model not fitted yet. Please call fit() first.")

        if str(self.cols) == "all" or self.cols is None or (type(self.cols) == list and len(self.cols) == 0):
            feature_cols = self.feature_names
        else:
            if isinstance(self.cols, list) and len(self.cols) > 0:
                if isinstance(self.cols[0], (tuple, list)):
                    feature_cols = [c[0] for c in self.cols]
                else:
                    feature_cols = self.cols
            else:
                feature_cols = self.feature_names

        X = np.array([[s.get(col, 0) for col in feature_cols]])

        leaf_indices = self.lgb_model.predict(X)

        total_leaves = sum(self.num_leaves_per_tree)
        result = np.zeros(total_leaves, dtype=np.int32)

        current_leaf_idx = int(leaf_indices[0])
        leaf_offset = 0
        for num_leaves in self.num_leaves_per_tree:
            if current_leaf_idx < num_leaves:
                result[leaf_offset + current_leaf_idx] = 1
                break
            leaf_offset += num_leaves
            current_leaf_idx -= num_leaves

        leaf_cols = [f"leaf_{j}" for j in range(total_leaves)]
        return dict(zip(leaf_cols, result.tolist()))

    def udf_get_params(self) -> dict_type:
        """
        获取模型参数。

        Returns:
            dict: 模型参数字典
        """
        return {
            "lgb_model": self.lgb_model,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "min_child_samples": self.min_child_samples,
            "feature_names": self.feature_names,
            "num_trees": self.num_trees,
            "num_leaves_per_tree": self.num_leaves_per_tree,
            "describe_dict": self.describe_dict,
            "_feature_unique_values": self._feature_unique_values
        }

    def udf_set_params(self, params: dict_type):
        """
        设置模型参数。

        Args:
            params: 模型参数字典
        """
        self.lgb_model = params["lgb_model"]
        self.n_estimators = params["n_estimators"]
        self.max_depth = params["max_depth"]
        self.learning_rate = params["learning_rate"]
        self.num_leaves = params["num_leaves"]
        self.min_child_samples = params["min_child_samples"]
        self.feature_names = params["feature_names"]
        self.num_trees = params["num_trees"]
        self.num_leaves_per_tree = params["num_leaves_per_tree"]
        self.describe_dict = params["describe_dict"]
        self._feature_unique_values = params.get("_feature_unique_values", {})

        self.feature_name_to_idx = {name: idx for idx, name in enumerate(self.feature_names)}


class TargetEncoding(EncodingBase):
    """
    目标编码。
    
    根据目标变量的均值对类别特征进行编码。
    支持平滑处理和最小样本数限制。
    
    Args:
        error_value: 找不到列取值时返回的值
        smoothing: 是否使用平滑处理
        min_sample: 最小样本数限制
    
    Example:
        >>> encoding = TargetEncoding(cols=["city"], y=y, smoothing=True, min_sample=100)
    """
    
    def __init__(self, cols="all", error_value=-1, smoothing=False, min_sample=100, **kwargs):
        """
        初始化目标编码。
        
        Args:
            cols: 要操作的列
            error_value: 找不到col取值时返回的值，也可以设置为np.nan待后续处理
            smoothing: 是否使用平滑处理
            min_sample: 需要满足的最小样本量
            **kwargs: 其他父类参数
        """
        super().__init__(cols=cols, **kwargs)
        self.error_value = error_value
        self.target_map_detail = dict()
        self.smoothing = smoothing
        self.min_sample = min_sample

    def show_detail(self):
        """显示目标编码的详细信息。"""
        data = []
        for col, map_detail in self.target_map_detail.items():
            for bin_value, target_value in map_detail.items():
                data.append([col, bin_value, target_value])
        return pd.DataFrame(data=data, columns=["col", "bin_value", "target_value"])

    def udf_fit(self, s: dataframe_type, **kwargs):
        """
        训练阶段，计算各列取值对应的目标均值。
        
        Args:
            s: 输入数据
            **kwargs: 其他参数
            
        Returns:
            self
        """
        assert self.y is not None and len(self.y) == len(s)
        s["y_"] = self.y
        # smoothing用
        total_target_value = np.mean(self.y)
        total_count = len(self.y)
        for col, _ in self.cols:
            tmp_ = s[[col, "y_"]]
            col_map = list(tmp_.groupby([col]).agg({"y_": ["mean"]}).to_dict().values())[0]
            # update smoothing
            if self.smoothing:
                col_map_count = list(tmp_.groupby([col]).agg({"y_": ["count"]}).to_dict().values())[0]
                for col_name, target_value in col_map.items():
                    col_count = col_map_count.get(col_name)
                    col_rate = col_count / total_count
                    new_target_value = col_rate * target_value + (1 - col_rate) * total_target_value
                    col_map[col_name] = new_target_value
            # update min_sample
            if self.min_sample:
                col_map_count = list(tmp_.groupby([col]).agg({"y_": ["count"]}).to_dict().values())[0]
                for col_name, _ in col_map.items():
                    col_count = col_map_count.get(col_name)
                    if col_count < self.min_sample:
                        col_map[col_name] = self.error_value
            self.target_map_detail[col] = col_map
        del s["y_"]
        return self

    def apply_function_single(self, col: str, x):
        """对单个值进行目标编码。"""
        map_detail_ = self.target_map_detail.get(col, dict())
        return np.float64(map_detail_.get(x, self.error_value))

    def apply_function_series(self, col: str, x: series_type):
        """对Series进行目标编码。"""
        map_detail_ = self.target_map_detail.get(col, dict())
        return x.map(map_detail_).fillna(self.error_value).astype(np.float64)

    def udf_get_params(self) -> dict_type:
        """获取参数。"""
        return {"target_map_detail": self.target_map_detail, "error_value": self.error_value,
                "smoothing": self.smoothing, "min_sample": self.min_sample}

    def udf_set_params(self, params: dict_type):
        """设置参数。"""
        self.target_map_detail = params["target_map_detail"]
        self.error_value = params["error_value"]
        self.smoothing = params["smoothing"]
        self.min_sample = params["min_sample"]


class LabelEncoding(EncodingBase):
    """
    标签编码。
    
    将类别特征转换为整数标签。
    
    Example:
        >>> encoding = LabelEncoding(cols=["city", "country"])
    """
    
    def __init__(self, cols="all", error_value=0, **kwargs):
        """
        初始化标签编码。
        
        Args:
            cols: 要操作的列
            error_value: 找不到标签时返回的值
            **kwargs: 其他父类参数
        """
        super().__init__(cols=cols, **kwargs)
        self.error_value = error_value
        self.label_map_detail = dict()

    def show_detail(self):
        """显示标签编码的详细信息。"""
        return pd.DataFrame([self.label_map_detail])

    def udf_fit(self, s: dataframe_type, **kwargs):
        """
        训练阶段，构建标签映射。
        
        Args:
            s: 输入数据
            **kwargs: 其他参数
            
        Returns:
            self
        """
        for col, _ in self.cols:
            col_value_list = s[col].unique().tolist()
            c = 1
            col_map = dict()
            for key in col_value_list:
                col_map[key] = c
                c += 1
            self.label_map_detail[col] = col_map
        return self

    def apply_function_single(self, col: str, x):
        """对单个值进行标签编码。"""
        map_detail_ = self.label_map_detail.get(col, dict())
        return np.int64(map_detail_.get(x, self.error_value))

    def apply_function_series(self, col: str, x: series_type):
        map_detail_ = self.label_map_detail.get(col, dict())
        return x.map(map_detail_).fillna(self.error_value).astype(np.int64)

    def udf_get_params(self) -> dict_type:
        return {"label_map_detail": self.label_map_detail, "error_value": self.error_value}

    def udf_set_params(self, params: dict):
        self.label_map_detail = params["label_map_detail"]
        self.error_value = params["error_value"]


class OneHotEncoding(EncodingBase):

    def __init__(self, cols="all", drop_col=True, **kwargs):
        """

        :param cols:
        :param drop_col: one-hot展开后，默认删除原col
        :param kwargs:
        """
        super().__init__(cols=cols, **kwargs)
        self.drop_col = drop_col
        self.one_hot_detail = dict()

    def show_detail(self):
        return pd.DataFrame([self.one_hot_detail])

    def udf_fit(self, s: dataframe_type, **kwargs):
        for col, _ in self.cols:
            self.one_hot_detail[col] = s[col].unique().tolist()
        return self

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        for col, new_col in self.cols:
            if col not in self.one_hot_detail.keys():
                raise Exception("{} not in {}".format(col, self.one_hot_detail.keys()))
            values = self.one_hot_detail.get(col)
            for value in values:
                s["{}_{}".format(new_col, value)] = (s[col] == value).astype(np.uint8)
        if self.drop_col:
            for col in self.one_hot_detail.keys():
                del s[col]
        return s

    def udf_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        for col, new_col in self.cols:
            if col not in self.one_hot_detail.keys():
                raise Exception("{} not in {}".format(col, self.one_hot_detail.keys()))
            values = self.one_hot_detail.get(col)
            for value in values:
                if s[col] == value:
                    s["{}_{}".format(new_col, value)] = 1
                else:
                    s["{}_{}".format(new_col, value)] = 0
        if self.drop_col:
            for col in self.one_hot_detail.keys():
                del s[col]
        return s

    def udf_get_params(self) -> dict_type:
        return {"one_hot_detail": self.one_hot_detail, "drop_col": self.drop_col}

    def udf_set_params(self, params: dict):
        self.one_hot_detail = params["one_hot_detail"]
        self.drop_col = params["drop_col"]


class WOEEncoding(EncodingBase):

    def __init__(self, y=None, cols="all", drop_col=False, save_detail=True, error_value=0, **kwargs):
        """

        :param y:
        :param cols:
        :param drop_col: encoding之后默认不删除原来的col
        :param save_detail: 是否保存map细节
        :param error_value: 出错时返回默认值
        :param kwargs:
        """
        super().__init__(cols=cols, **kwargs)
        self.drop_col = drop_col
        self.error_value = error_value
        self.woe_map_detail = dict()
        self.save_detail = save_detail
        self.dist_detail = []
        self.y = y

    def udf_fit(self, s: dataframe_type, **kwargs):
        # 检测y的长度与训练数据是否一致
        assert self.y is not None and len(self.y) == len(s)
        total_bad_num = np.sum(self.y == 1)
        total_good_num = len(self.y) - total_bad_num
        if total_bad_num == 0 or total_good_num == 0:
            raise Exception("should total_bad_num > 0 and total_good_num > 0")
        self.woe_map_detail = dict()
        self.dist_detail = []
        for col, _ in self.cols:
            for bin_value in s[col].unique().tolist():
                hint_index = s[col] == bin_value
                bad_num = np.sum(hint_index & self.y == 1)
                good_num = np.sum(hint_index) - bad_num
                bad_rate = bad_num / total_bad_num
                good_rate = good_num / total_good_num
                if good_num == 0 or bad_num == 0:
                    woe = 0
                    iv = 0
                else:
                    woe = np.log(good_rate / bad_rate)
                    iv = (good_rate - bad_rate) * woe
                if self.woe_map_detail.get(col) is None:
                    self.woe_map_detail[col] = dict()
                self.woe_map_detail[col][bin_value] = woe
                self.dist_detail.append([col, bin_value, bad_num, bad_rate, good_num, good_rate, woe, iv])
        return self

    def show_detail(self):
        return pd.DataFrame(data=self.dist_detail,
                            columns=["col", "bin_value", "bad_num", "bad_rate", "good_num", "good_rate", "woe", "iv"])

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        for col, new_col in self.cols:
            if col not in self.woe_map_detail.keys():
                raise Exception("{} not in {}".format(col, self.woe_map_detail.keys()))
            s[new_col] = self.apply_function_series(col, s[col])
        if self.drop_col:
            for col in self.woe_map_detail.keys():
                del s[col]
        return s

    def udf_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        for col, new_col in self.cols:
            if col not in self.woe_map_detail.keys():
                raise Exception("{} not in {}".format(col, self.woe_map_detail.keys()))
            s[new_col] = self.apply_function_single(col, s[col])
        if self.drop_col:
            for col in self.woe_map_detail.keys():
                del s[col]
        return s

    def apply_function_single(self, col: str, x):
        map_detail_ = self.woe_map_detail.get(col, dict())
        return np.float64(map_detail_.get(x, self.error_value))

    def apply_function_series(self, col: str, x: series_type):
        map_detail_ = self.woe_map_detail.get(col, dict())
        return x.map(map_detail_).fillna(self.error_value).astype(np.float64)

    def udf_get_params(self) -> dict_type:
        params = {"woe_map_detail": self.woe_map_detail, "error_value": self.error_value,
                  "drop_col": self.drop_col, "save_detail": self.save_detail}
        if self.save_detail:
            params["dist_detail"] = self.dist_detail
        return params

    def udf_set_params(self, params: dict):
        self.save_detail = params["save_detail"]
        self.error_value = params["error_value"]
        if self.save_detail:
            self.dist_detail = params["dist_detail"]
        else:
            self.dist_detail = []
        self.woe_map_detail = params["woe_map_detail"]
        self.drop_col = params["drop_col"]
