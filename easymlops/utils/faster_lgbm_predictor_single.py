"""
对于单列预测值
"""
import math


class FasterLgbSinglePredictor(object):
    def __init__(self, model: dict, cache_num=10):
        self.model = model
        assert self.model["version"] == "v3"
        self.cache_num = cache_num
        self.feature_names = self.model["feature_names"]
        self.init_average_value()
        self.forest_cache = ForestCache(cache_num=self.cache_num)

    def init_average_value(self):
        for i in range(len(self.model["tree_info"])):
            tree = self.model["tree_info"][i]["tree_structure"]
            self.post_tree_avg_value_update(tree)

    def post_tree_avg_value_update(self, tree: dict):
        if tree.get("left_child") is not None:
            left_rst = self.post_tree_avg_value_update(tree["left_child"])
            right_rst = self.post_tree_avg_value_update(tree["right_child"])
            avg_value = (left_rst[0] * left_rst[1] + right_rst[0] * right_rst[1]) / (left_rst[1] + right_rst[1])
            tree["avg_value"] = avg_value
            rst = [avg_value, tree["internal_count"]]
        else:
            rst = [tree["leaf_value"], tree["leaf_count"]]
        return rst

    def predict(self, input_dict: dict):
        # 遍历统计每棵树的score以及contrib
        rst = dict()
        for i in range(len(self.model["tree_info"])):
            num_leaves = self.model["tree_info"][i]["num_leaves"]
            tree = self.model["tree_info"][i]["tree_structure"]
            current_rst = self.forest_cache.predict(i, input_dict, num_leaves, tree, self.feature_names)
            # 合并
            for key in current_rst.keys():
                if rst.get(key) is None:
                    rst[key] = 0
                rst[key] += current_rst.get(key)
        # 切分score和重要度分布
        score = rst["_predict_value"]
        if "sigmoid" in self.model["objective"]:
            score = MathUtils.sigmoid(score)
        elif self.model["objective"] in ["tweedie", "poisson"]:
            score = math.exp(score)
        rst.pop("_predict_value")
        rst.pop("_bias")
        return {"score": score, "contrib": rst}


class ForestCache(object):
    def __init__(self, cache_num=10):
        self.cache_num = cache_num
        assert type(self.cache_num) == int
        self.tree_conditions = dict()

    def predict(self, tree_id: int, line: dict, num_leaves: int, tree: dict, feature_names: list):
        tree_cache = self.tree_conditions.get(tree_id)
        if tree_cache is None:
            tree_cache = TreeCache(self.cache_num)
            self.tree_conditions[tree_id] = tree_cache
        return tree_cache.predict(line, num_leaves, tree, feature_names)


class TreeCache(object):
    def __init__(self, cache_num=10):
        self.count = 0
        self.cache_num = cache_num
        self.condition_maps = dict()
        self.cache_values = dict()
        self.cache_contrib_values = dict()

    def predict(self, line: dict, num_leaves: int, tree: dict, feature_names: list):
        contrib_map = self.get_cache_value(line)
        if contrib_map is not None:
            return contrib_map
        conditions = dict()
        contrib_map = dict()
        columns = set()
        current_value = tree["avg_value"]
        contrib_map["_bias"] = current_value
        for j in range(num_leaves):
            decision_type = tree["decision_type"]
            if "<=" == decision_type:
                threshold = tree["threshold"]
                default_left = tree["default_left"]
                split_feature = feature_names[tree["split_feature"]]
                columns.add(split_feature)
                next_decision = self.decision(line.get(split_feature), threshold, default_left)
            else:
                thresholds = tree["threshold"]
                default_left = tree["default_left"]
                split_feature = feature_names[tree["split_feature"]]
                columns.add(split_feature)
                next_decision = self.decision2(line.get(split_feature), thresholds, default_left)
            tree = tree.get(next_decision)
            if tree.get("avg_value") is not None:
                next_value = tree.get("avg_value")
            else:
                next_value = tree.get("leaf_value")
            if contrib_map.get(split_feature) is None:
                contrib_map[split_feature] = 0

            contrib_map[split_feature] = contrib_map.get(split_feature) + next_value - current_value
            current_value = next_value
            # 判断是否叶子节点
            if tree.get("left_child") is None:
                contrib_map["_predict_value"] = tree.get("leaf_value")
                break
        for column in columns:
            conditions[column] = line.get(column)
        self.condition_maps[self.count] = conditions
        self.cache_contrib_values[self.count] = contrib_map
        self.count += 1
        self.count %= (self.cache_num + 1)
        return contrib_map

    def get_cache_value(self, line: dict):
        if len(self.condition_maps) == 0:
            return None
        for i in range(self.cache_num):
            conditions = self.condition_maps.get(i)
            cache_contrib = self.cache_contrib_values.get(i)
            if cache_contrib is None or len(cache_contrib) == 0:
                return None
            else:
                if self.is_match(line, conditions):
                    return cache_contrib
                else:
                    return None

    @staticmethod
    def is_match(line: dict, conditions: dict):
        if conditions is None or len(conditions) == 0:
            return False
        for key, value in conditions.items():
            if value != line.get(key):
                return False
        return True

    @staticmethod
    def decision(data: float, threshold: float, default_left: bool):
        if data is None and default_left is True:
            return "left_child"
        elif data <= threshold:
            return "left_child"
        else:
            return "right_child"

    @staticmethod
    def decision2(data: float, threshold2: str, default_left: bool):
        if "|" == threshold2 and default_left is True:
            return "left_child"
        else:
            for item in threshold2.split("||"):
                item_double = float(item)
                if data == item_double:
                    return "left_child"
            return "right_child"


class MathUtils(object):
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + math.exp(-1 * x))
