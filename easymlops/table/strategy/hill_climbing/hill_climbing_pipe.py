from easymlops.table.core import *
from easymlops.table.regression import RegressionBase
from .climber import ClimberCV
from sklearn.model_selection import KFold
from easymlops.table.utils import EvalFunction


class HillClimbingStackingRegression(RegressionBase):
    def __init__(self, y: series_type = None, eval_function="rmse", eval_function_direct="minimize", cols="all",
                 allow_negative_weights=True, precision=0.001, score_decimal_places=6, n_jobs=1, use_gpu=False,
                 n_splits=5, random_state=42, shuffle=True,
                 pred_name="pred", skip_check_transform_type=True,
                 drop_input_data=True, support_sparse_input=False,
                 native_init_params=None, native_fit_params=None,
                 **kwargs):
        super().__init__(y=y, cols=cols, pred_name=pred_name, skip_check_transform_type=skip_check_transform_type,
                         drop_input_data=drop_input_data, support_sparse_input=support_sparse_input,
                         native_init_params=native_init_params, native_fit_params=native_fit_params,
                         **kwargs)
        if type(eval_function) == str:
            eval_function = EvalFunction.get(eval_function)
        self.climber = ClimberCV(
            objective=eval_function_direct,
            eval_metric=eval_function,
            allow_negative_weights=allow_negative_weights,
            precision=precision,
            score_decimal_places=score_decimal_places,
            n_jobs=n_jobs,
            use_gpu=use_gpu,
            verbose=kwargs.get("show_process"),
            cv=KFold(n_splits=n_splits, random_state=random_state, shuffle=shuffle)
        )

    def udf_fit(self, s, **kwargs):
        self.climber.fit(s, self.y)
        return self

    def udf_transform(self, s, **kwargs):
        if self.drop_input_data:
            result = pd.DataFrame()
            result.index = s.index
            result[self.pred_name] = self.climber.predict(s)
            return result
        else:
            s[self.pred_name] = self.climber.predict(s)
            return s

    def udf_get_params(self) -> dict_type:
        return {"climber": self.climber}

    def udf_set_params(self, params: dict):
        self.climber = params["climber"]
