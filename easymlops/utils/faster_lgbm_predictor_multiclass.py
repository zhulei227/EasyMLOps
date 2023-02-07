"""
对于多分类
"""
from .faster_lgbm_predictor_single import *
import copy


class FasterLgbMulticlassPredictor(object):
    def __init__(self, model: dict, cache_num=10):
        # 保存
        self.model = model
        # 按类别分拆model
        num_class = model["num_class"]
        assert model["version"] == "v3" and model["num_class"] > 1
        self.cache_num = cache_num
        # 分拆参数
        self.model_params_map = dict()
        for idx in range(num_class):
            self.model_params_map[idx] = copy.deepcopy(model)
            # 清洗tree_info
            tree_info = self.model_params_map[idx]["tree_info"]
            new_tree_info = []
            for tree in tree_info:
                if tree["tree_index"] % num_class == idx:
                    new_tree_info.append(tree)
            self.model_params_map[idx]["tree_info"] = new_tree_info
        # 分别初始化
        self.model_map = dict()
        for idx in range(num_class):
            self.model_map[idx] = FasterLgbSinglePredictor(model=self.model_params_map[idx], cache_num=cache_num)

    def predict(self, input_dict: dict):
        # 分别获取每类别的值
        score = {}
        contrib = {}
        for idx, single_model in self.model_map.items():
            pred = single_model.predict(input_dict)
            score[idx] = pred.get("score")
            contrib.update(pred.get("contrib"))
        objective = self.model.get("objective", "")
        if "multiclass" in objective:
            # softmax
            total_value = 0
            for key, value in score.items():
                value = math.exp(value)
                total_value += value
                score[key] = value
            for key, value in score.items():
                score[key] = value / total_value
        return {"score": score, "contrib": contrib}
