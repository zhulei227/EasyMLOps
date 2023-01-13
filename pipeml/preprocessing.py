from .base import *


class UserDefinedPreprocessObject(PipeObject):
    """
    只需要对单个key-value处理的类型
    """

    def __init__(self, cols="all", name=None, transform_check_max_number_error=1e-5,
                 skip_check_transform_type=False, skip_check_transform_value=False, copy_transform_data=True):
        PipeObject.__init__(self, name=name, transform_check_max_number_error=transform_check_max_number_error,
                            skip_check_transform_type=skip_check_transform_type,
                            skip_check_transform_value=skip_check_transform_value,
                            copy_transform_data=copy_transform_data)
        self.cols = cols

    def _user_defined_function(self, col, x):
        """
        对col的对应值x执行某种自定义操作
        """
        raise Exception("need to implement")

    @check_dataframe_type
    def _fit(self, s: dataframe_type) -> dataframe_type:
        """
        补充fit阶段的自定义操作
        """
        return self

    @check_dataframe_type
    def fit(self, s: dataframe_type) -> dataframe_type:
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
        self.input_col_names = s.columns.tolist()
        self._fit(s)
        return self

    @check_dataframe_type
    def transform(self, s: dataframe_type) -> dataframe_type:
        if self.copy_transform_data:
            s_ = s[self.input_col_names]
        else:
            s = s[self.input_col_names]
            s_ = s
        for col, new_col in self.cols:
            if col in s_.columns:
                s_[new_col] = s_[col].apply(lambda x: self._user_defined_function(col, x))
        self.output_col_names = s_.columns.tolist()
        return s_[self.output_col_names]

    @staticmethod
    def extract_dict(s: dict_type, keys: list):
        new_s = dict()
        for key in keys:
            new_s[key] = s[key]
        return new_s

    @check_dict_type
    def transform_single(self, s: dict_type) -> dict_type:
        s_ = self.extract_dict(s, self.input_col_names)
        for col, new_col in self.cols:
            if col in s_.keys():
                s_[new_col] = self._user_defined_function(col, s_[col])
        return self.extract_dict(s_, self.output_col_names)

    def get_params(self) -> dict_type:
        params = PipeObject.get_params(self)
        params.update({"cols": self.cols})
        return params

    def set_params(self, params: dict):
        PipeObject.set_params(self, params)
        self.cols = params["cols"]


class FixInput(UserDefinedPreprocessObject):
    """
    固定输入，对不存在的col用None填充
    """

    def __init__(self, fill_value=None, name=None, copy_transform_data=True):
        UserDefinedPreprocessObject.__init__(self, cols="all", name=name, skip_check_transform_type=True,
                                             skip_check_transform_value=True, copy_transform_data=copy_transform_data)
        self.fill_value = fill_value

    @check_dataframe_type
    def _fit(self, s: dataframe_type) -> dataframe_type:
        self.input_col_names = [col for col, _ in self.cols]
        self.output_col_names = [col for col, _ in self.cols]
        return self

    @check_dataframe_type
    def transform(self, s: dataframe_type) -> dataframe_type:
        if self.copy_transform_data:
            s_ = copy.copy(s)
        else:
            s_ = s
        for col in self.output_col_names:
            if col not in s_.columns:
                s_[col] = self.fill_value
        return s_[self.output_col_names]

    @check_dict_type
    def transform_single(self, s: dict_type) -> dict_type:
        s_ = copy.copy(s)
        for col in self.output_col_names:
            if col not in s_.keys():
                s_[col] = self.fill_value
        return self.extract_dict(s_, self.output_col_names)

    def get_params(self) -> dict_type:
        params = UserDefinedPreprocessObject.get_params(self)
        params.update({"fill_value": self.fill_value})
        return params

    def set_params(self, params: dict):
        UserDefinedPreprocessObject.set_params(self, params)
        self.fill_value = params["fill_value"]


class DropCols(UserDefinedPreprocessObject):
    @check_dataframe_type
    def transform(self, s: dataframe_type) -> dataframe_type:
        if self.copy_transform_data:
            s_ = s[self.input_col_names]
        else:
            s = s[self.input_col_names]
            s_ = s
        for col, _ in self.cols:
            if col in s_.columns.tolist():
                del s_[col]
        self.output_col_names = s_.columns.tolist()
        return s_

    @check_dict_type
    def transform_single(self, s: dict_type) -> dict_type:
        s_ = self.extract_dict(s, self.input_col_names)
        for col, _ in self.cols:
            if col in s_.keys():
                s_.pop(col)
        return self.extract_dict(s_, self.output_col_names)


class SelectCols(UserDefinedPreprocessObject):
    @check_dataframe_type
    def transform(self, s: dataframe_type) -> dataframe_type:
        s_ = copy.copy(s)
        selected_cols = [col for col, _ in self.cols]
        for col in selected_cols:
            if col not in s_.columns:
                raise Exception("{} not in {}".format(col, s_.columns.tolist))
        s_ = s_[selected_cols]
        self.output_col_names = s_.columns.tolist()
        return s_

    @check_dict_type
    def transform_single(self, s: dict_type) -> dict_type:
        s_ = dict()
        selected_cols = [col for col, _ in self.cols]
        for col in selected_cols:
            if col not in s.keys():
                raise Exception("{} not in {}".format(col, s.keys()))
        for col in selected_cols:
            s_[col] = s[col]
        return self.extract_dict(s_, self.output_col_names)


class FillNa(UserDefinedPreprocessObject):
    """
    优先级fill_detail>fill_mode>fill_value
    fill_mode可选:mean,median,mode
    """

    def __init__(self, cols="all", fill_mode=None, fill_value=None, fill_detail=None, error_value=0, name=None,
                 copy_transform_data=True):
        UserDefinedPreprocessObject.__init__(self, cols=cols, name=name, skip_check_transform_type=True,
                                             skip_check_transform_value=True, copy_transform_data=copy_transform_data)
        self.fill_value = fill_value
        self.fill_mode = fill_mode
        self.fill_detail = fill_detail
        self.error_value = error_value

    @check_dataframe_type
    def _fit(self, s: dataframe_type) -> dataframe_type:
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

    def get_params(self) -> dict_type:
        params = UserDefinedPreprocessObject.get_params(self)
        params.update({"fill_detail": self.fill_detail, "error_value": self.error_value})
        return params

    def set_params(self, params: dict):
        UserDefinedPreprocessObject.set_params(self, params)
        self.fill_detail = params["fill_detail"]
        self.error_value = params["error_value"]


class TransToCategory(UserDefinedPreprocessObject):
    def __init__(self, cols="all", map_detail=(["None", "nan", "", " ", "*", "inf"], "nan"), name=None,
                 copy_transform_data=True):
        UserDefinedPreprocessObject.__init__(self, cols=cols, name=name, skip_check_transform_type=False,
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

    def get_params(self) -> dict_type:
        params = UserDefinedPreprocessObject.get_params(self)
        params.update({"map_detail": self.map_detail})
        return params

    def set_params(self, params: dict):
        UserDefinedPreprocessObject.set_params(self, params)
        self.map_detail = params["map_detail"]


class TransToFloat(UserDefinedPreprocessObject):
    def __init__(self, cols="all", nan_fill_value=0, name=None, transform_check_max_number_error=1e-3,
                 copy_transform_data=True):
        UserDefinedPreprocessObject.__init__(self, cols=cols, name=name,
                                             transform_check_max_number_error=transform_check_max_number_error,
                                             copy_transform_data=copy_transform_data)
        self.nan_fill_value = nan_fill_value

    def _user_defined_function(self, col, x):
        try:
            return float(x)
        except:
            x = self.nan_fill_value
        return x

    def get_params(self) -> dict_type:
        params = UserDefinedPreprocessObject.get_params(self)
        params.update({"nan_fill_value": self.nan_fill_value})
        return params

    def set_params(self, params: dict):
        UserDefinedPreprocessObject.set_params(self, params)
        self.nan_fill_value = params["nan_fill_value"]


class TransToInt(UserDefinedPreprocessObject):
    def __init__(self, cols="all", nan_fill_value=0, name=None, transform_check_max_number_error=1e-3,
                 copy_transform_data=True):
        UserDefinedPreprocessObject.__init__(self, cols=cols, name=name,
                                             transform_check_max_number_error=transform_check_max_number_error,
                                             copy_transform_data=copy_transform_data)
        self.nan_fill_value = nan_fill_value

    def _user_defined_function(self, col, x):
        try:
            return int(float(x))
        except:
            x = self.nan_fill_value
        return x

    def get_params(self) -> dict_type:
        params = UserDefinedPreprocessObject.get_params(self)
        params.update({"nan_fill_value": self.nan_fill_value})
        return params

    def set_params(self, params: dict):
        UserDefinedPreprocessObject.set_params(self, params)
        self.nan_fill_value = params["nan_fill_value"]


class TransToLower(UserDefinedPreprocessObject):
    def _user_defined_function(self, col, x):
        try:
            return str(x).lower()
        except:
            return ""


class TransToUpper(UserDefinedPreprocessObject):
    def _user_defined_function(self, col, x):
        try:
            return str(x).upper()
        except:
            return ""


class CategoryMapValues(UserDefinedPreprocessObject):
    def __init__(self, cols="all", default_map=([""], ""), map_detail=None, name=None, copy_transform_data=True):
        UserDefinedPreprocessObject.__init__(self, cols=cols, name=name, copy_transform_data=copy_transform_data)
        self.default_map = default_map
        self.map_detail = map_detail

    def _user_defined_function(self, col, x):
        if x in self.map_detail.get(col, ([""], ""))[0]:
            return self.map_detail.get(col, ([""], ""))[1]
        else:
            return x

    @check_dataframe_type
    def _fit(self, s: dataframe_type) -> dataframe_type:
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

    def get_params(self) -> dict_type:
        params = UserDefinedPreprocessObject.get_params(self)
        params.update({"map_detail": self.map_detail})
        return params

    def set_params(self, params: dict):
        UserDefinedPreprocessObject.set_params(self, params)
        self.map_detail = params["map_detail"]


class Clip(UserDefinedPreprocessObject):
    def __init__(self, cols="all", default_clip=None, clip_detail=None, percent_range=None, name=None,
                 skip_check_transform_type=False, copy_transform_data=True):
        """
       优先级clip_detail>percent_range>default_clip
        """
        UserDefinedPreprocessObject.__init__(self, cols=cols, name=name,
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

    @check_dataframe_type
    def _fit(self, s: dataframe_type) -> dataframe_type:
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

    def get_params(self) -> dict_type:
        params = UserDefinedPreprocessObject.get_params(self)
        params.update({"clip_detail": self.clip_detail})
        return params

    def set_params(self, params: dict):
        UserDefinedPreprocessObject.set_params(self, params)
        self.clip_detail = params["clip_detail"]
