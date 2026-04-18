import numpy as np


class EvalFunction(object):
    @staticmethod
    def mae(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    @staticmethod
    def rmse(y_true, y_pred):
        return np.sqrt(np.mean((y_true - y_pred) ** 2))

    @staticmethod
    def get(function_name):
        if function_name == "mae":
            return EvalFunction.mae
        elif function_name == "rmse":
            return EvalFunction.rmse
