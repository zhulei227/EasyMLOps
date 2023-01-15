from ..base import *


class UserDefinedPreprocessObject(PipeObject):
    """
    只需要对单个key-value处理的类型
    """

    def __init__(self, cols="all", name=None, transform_check_max_number_error=1e-5,
                 skip_check_transform_type=False, skip_check_transform_value=False, copy_transform_data=True):
        super().__init__(name=name, transform_check_max_number_error=transform_check_max_number_error,
                         skip_check_transform_type=skip_check_transform_type,
                         skip_check_transform_value=skip_check_transform_value,
                         copy_transform_data=copy_transform_data)
        self.cols = cols

    def _user_defined_function(self, col, x):
        """
        对col的对应值x执行某种自定义操作
        """
        raise Exception("need to implement")

    def before_fit(self, s: dataframe_type) -> dataframe_type:
        s = super().before_fit(s)
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

    @fit_wrapper
    def fit(self, s: dataframe_type) -> dataframe_type:
        return self

    @transform_wrapper
    def transform(self, s: dataframe_type) -> dataframe_type:
        for col, new_col in self.cols:
            if col in s.columns:
                s[new_col] = s[col].apply(lambda x: self._user_defined_function(col, x))
        return s

    @transform_single_wrapper
    def transform_single(self, s: dict_type) -> dict_type:
        for col, new_col in self.cols:
            if col in s.keys():
                s[new_col] = self._user_defined_function(col, s[col])
        return s

    def _get_params(self):
        return {"cols": self.cols}

    def _set_params(self, params: dict):
        self.cols = params["cols"]


class FixInput(UserDefinedPreprocessObject):
    """
    固定输入，对不存在的col用None填充
    """

    def __init__(self, fill_value=None, name=None, copy_transform_data=True):
        super().__init__(cols="all", name=name, skip_check_transform_type=True,
                         skip_check_transform_value=True, copy_transform_data=copy_transform_data)
        self.fill_value = fill_value

    @fit_wrapper
    def fit(self, s: dataframe_type):
        self.output_col_names = self.input_col_names
        return self

    def transform(self, s: dataframe_type) -> dataframe_type:
        if self.copy_transform_data:
            s_ = copy.copy(s)
        else:
            s_ = s
        for col in self.output_col_names:
            if col not in s_.columns:
                s_[col] = self.fill_value
        return s_[self.output_col_names]

    def transform_single(self, s: dict_type) -> dict_type:
        s_ = copy.copy(s)
        for col in self.output_col_names:
            if col not in s_.keys():
                s_[col] = self.fill_value
        return self.extract_dict(s_, self.output_col_names)

    def _get_params(self) -> dict_type:
        return {"fill_value": self.fill_value}

    def _set_params(self, params: dict_type):
        self.fill_value = params["fill_value"]


class DropCols(UserDefinedPreprocessObject):
    @transform_wrapper
    def transform(self, s: dataframe_type) -> dataframe_type:
        for col, _ in self.cols:
            if col in s.columns.tolist():
                del s[col]
        return s

    @transform_single_wrapper
    def transform_single(self, s: dict_type) -> dict_type:
        for col, _ in self.cols:
            if col in s.keys():
                s.pop(col)
        return s


class SelectCols(UserDefinedPreprocessObject):
    @transform_wrapper
    def transform(self, s: dataframe_type) -> dataframe_type:
        selected_cols = [col for col, _ in self.cols]
        for col in selected_cols:
            if col not in s.columns:
                raise Exception("{} not in {}".format(col, s.columns.tolist))
        s = s[selected_cols]
        return s

    @transform_single_wrapper
    def transform_single(self, s: dict_type) -> dict_type:
        s_ = dict()
        selected_cols = [col for col, _ in self.cols]
        for col in selected_cols:
            if col not in s.keys():
                raise Exception("{} not in {}".format(col, s.keys()))
        for col in selected_cols:
            s_[col] = s[col]
        return s_

    def _get_params(self):
        return {}


class FillNa(UserDefinedPreprocessObject):
    """
    优先级fill_detail>fill_mode>fill_value
    fill_mode可选:mean,median,mode
    """

    def __init__(self, cols="all", fill_mode=None, fill_value=None, fill_detail=None, error_value=0, name=None,
                 copy_transform_data=True):
        super().__init__(cols=cols, name=name, skip_check_transform_type=True,
                         skip_check_transform_value=True, copy_transform_data=copy_transform_data)
        self.fill_value = fill_value
        self.fill_mode = fill_mode
        self.fill_detail = fill_detail
        self.error_value = error_value

    @fit_wrapper
    def fit(self, s: dataframe_type):
        if self.fill_detail is None:
            self.fill_detail = dict()
            if self.fill_mode is not None:
                for col, _ in self.cols:
                    if self.fill_mode == "mean":
                        self.fill_detail[col] = s[col].mean()
                    elif self.fill_mode == "median":
                        self.fill_detail[col] = s[col].median()
                    elif self.fill_mode == "mode":
                        self.fill_detail[col] = s[col].mode()[0]
                    else:
                        raise Exception("fill_model should be [mean,median,mode]")
            else:
                for col, _ in self.cols:
                    self.fill_detail[col] = self.fill_value
        else:
            new_cols = []
            for col, new_col in self.cols:
                if col in self.fill_detail.keys():
                    new_cols.append((col, new_col))
            self.cols = new_cols
        return self

    def _user_defined_function(self, col, x):
        if str(x).lower() in ["none", "nan", "np.nan", "null"]:
            try:
                x = self.fill_detail.get(col, self.error_value)
            except:
                x = self.error_value
        return x

    def _get_params(self) -> dict_type:
        return {"fill_detail": self.fill_detail, "error_value": self.error_value}

    def _set_params(self, params: dict):
        self.fill_detail = params["fill_detail"]
        self.error_value = params["error_value"]


class TransToCategory(UserDefinedPreprocessObject):
    def __init__(self, cols="all", map_detail=(["None", "nan", "", " ", "*", "inf"], "nan"), name=None,
                 copy_transform_data=True):
        super().__init__(cols=cols, name=name, skip_check_transform_type=False,
                         copy_transform_data=copy_transform_data)
        self.map_detail = map_detail

    def _user_defined_function(self, col, x):
        x = str(x)
        try:
            x = str(int(float(x)))
        except:
            pass
        if x in self.map_detail[0]:
            x = self.map_detail[1]
        return x

    def _get_params(self) -> dict_type:
        return {"map_detail": self.map_detail}

    def _set_params(self, params: dict_type):
        self.map_detail = params["map_detail"]


class TransToFloat(UserDefinedPreprocessObject):
    def __init__(self, cols="all", nan_fill_value=0, name=None, transform_check_max_number_error=1e-3,
                 copy_transform_data=True):
        super().__init__(cols=cols, name=name,
                         transform_check_max_number_error=transform_check_max_number_error,
                         copy_transform_data=copy_transform_data)
        self.nan_fill_value = nan_fill_value

    def _user_defined_function(self, col, x):
        try:
            return float(x)
        except:
            x = self.nan_fill_value
        return x

    def _get_params(self) -> dict_type:
        return {"nan_fill_value": self.nan_fill_value}

    def _set_params(self, params: dict):
        self.nan_fill_value = params["nan_fill_value"]


class TransToInt(UserDefinedPreprocessObject):
    def __init__(self, cols="all", nan_fill_value=0, name=None, transform_check_max_number_error=1e-3,
                 copy_transform_data=True):
        super().__init__(cols=cols, name=name,
                         transform_check_max_number_error=transform_check_max_number_error,
                         copy_transform_data=copy_transform_data)
        self.nan_fill_value = nan_fill_value

    def _user_defined_function(self, col, x):
        try:
            return int(float(x))
        except:
            x = self.nan_fill_value
        return x

    def _get_params(self) -> dict_type:
        return {"nan_fill_value": self.nan_fill_value}

    def _set_params(self, params: dict):
        self.nan_fill_value = params["nan_fill_value"]


class TransToLower(UserDefinedPreprocessObject):
    def _user_defined_function(self, col, x):
        try:
            return str(x).lower()
        except:
            return ""

    def _get_params(self):
        return {}


class TransToUpper(UserDefinedPreprocessObject):
    def _user_defined_function(self, col, x):
        try:
            return str(x).upper()
        except:
            return ""

    def _get_params(self):
        return {}


class CategoryMapValues(UserDefinedPreprocessObject):
    def __init__(self, cols="all", default_map=([""], ""), map_detail=None, name=None, copy_transform_data=True):
        super().__init__(cols=cols, name=name, copy_transform_data=copy_transform_data)
        self.default_map = default_map
        self.map_detail = map_detail

    def _user_defined_function(self, col, x):
        if x in self.map_detail.get(col, ([""], ""))[0]:
            return self.map_detail.get(col, ([""], ""))[1]
        else:
            return x

    @fit_wrapper
    def fit(self, s: dataframe_type):
        if self.map_detail is None:
            self.map_detail = dict()
            for col, _ in self.cols:
                self.map_detail[col] = self.default_map
        else:
            new_cols = []
            for col, new_col in self.cols:
                if col in self.map_detail.keys():
                    new_cols.append((col, new_col))
            self.cols = new_cols
        return self

    def _get_params(self) -> dict_type:
        return {"map_detail": self.map_detail}

    def _set_params(self, params: dict):
        self.map_detail = params["map_detail"]


class Clip(UserDefinedPreprocessObject):
    def __init__(self, cols="all", default_clip=None, clip_detail=None, percent_range=None, name=None,
                 skip_check_transform_type=False, copy_transform_data=True):
        """
       优先级clip_detail>percent_range>default_clip
        """
        super().__init__(cols=cols, name=name,
                         skip_check_transform_type=skip_check_transform_type,
                         copy_transform_data=copy_transform_data)
        self.default_clip = default_clip
        self.clip_detail = clip_detail
        self.percent_range = percent_range

    def _user_defined_function(self, col, x):
        clip_detail_ = self.clip_detail.get(col, (None, None))
        if clip_detail_[0] is not None and str(x).lower() not in ["none", "nan", "null", "", "inf", "np.nan", "np.inf"]:
            x = clip_detail_[0] if x <= clip_detail_[0] else x
        if clip_detail_[1] is not None and str(x).lower() not in ["none", "nan", "null", "", "inf", "np.nan", "np.inf"]:
            x = clip_detail_[1] if x >= clip_detail_[1] else x
        return x

    @fit_wrapper
    def fit(self, s: dataframe_type):
        if self.clip_detail is None:
            self.clip_detail = dict()
            if self.percent_range is None:
                for col, _ in self.cols:
                    self.clip_detail[col] = self.default_clip
            else:
                for col, _ in self.cols:
                    if self.percent_range[0] is not None:
                        low = np.percentile(s[col], self.percent_range[0])
                    else:
                        low = None
                    if self.percent_range[1] is not None:
                        top = np.percentile(s[col], self.percent_range[1])
                    else:
                        top = None
                    self.clip_detail[col] = (low, top)
        else:
            new_cols = []
            for col, new_col in self.cols:
                if col in self.clip_detail.keys():
                    new_cols.append((col, new_col))
            self.cols = new_cols
        return self

    def _get_params(self) -> dict_type:
        return {"clip_detail": self.clip_detail}

    def _set_params(self, params: dict):
        self.clip_detail = params["clip_detail"]
