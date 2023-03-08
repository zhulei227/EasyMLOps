import re
from easymlops.table.core import *
from easymlops.nlp.core import *


class PreprocessBase(NLPPipeObjectBase):
    """
    文本清洗基础类
    """

    def __init__(self, cols="all", skip_check_transform_type=True, **kwargs):
        super().__init__(skip_check_transform_type=skip_check_transform_type, **kwargs)
        self.cols = cols

    def before_fit(self, s: dataframe_type, **kwargs) -> dataframe_type:
        s = super().before_fit(s, **kwargs)
        if str(self.cols).lower() in ["none", "all", "null"]:
            self.cols = self.input_col_names
        assert type(self.cols) == list and type(self.cols[0]) == str
        return s

    def udf_get_params(self):
        return {"cols": self.cols}

    def udf_set_params(self, params: dict_type):
        self.cols = params["cols"]


class Lower(PreprocessBase):
    """
    所有英文字符转小写
    """

    def __init__(self, cols="all", **kwargs):
        super().__init__(cols=cols, **kwargs)

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        for col in self.cols:
            s[col] = s[col].astype(str).str.lower()
        return s

    def udf_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        for col in self.cols:
            s[col] = str(s[col]).lower()
        return s

    def udf_get_params(self):
        return {}


class Upper(PreprocessBase):
    """
    所有英文字符转大写
    """

    def __init__(self, cols="all", **kwargs):
        super().__init__(cols=cols, **kwargs)

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        for col in self.cols:
            s[col] = s[col].astype(str).str.upper()
        return s

    def udf_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        for col in self.cols:
            s[col] = str(s[col]).upper()
        return s

    def udf_get_params(self):
        return {}


class RemoveDigits(PreprocessBase):
    """
    移除所有数字字符
    """

    def __init__(self, cols="all", **kwargs):
        super().__init__(cols=cols, **kwargs)

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        for col in self.cols:
            s[col] = s[col].astype(str).str.replace(r"\d+", "")
        return s

    def udf_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        for col in self.cols:
            s[col] = re.sub(r"\d+", "", str(s[col]))
        return s

    def udf_get_params(self):
        return {}


class ReplaceDigits(PreprocessBase):
    """
    替换数字为指定的字符
    """

    def __init__(self, cols="all", symbols="[d]", **kwargs):
        super().__init__(cols=cols, **kwargs)
        self.symbols = symbols

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        for col in self.cols:
            s[col] = s[col].astype(str).str.replace(r"\d+", self.symbols)
        return s

    def udf_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        for col in self.cols:
            s[col] = re.sub(r"\d+", self.symbols, str(s[col]))
        return s

    def udf_get_params(self):
        return {"symbols": self.symbols}

    def _set_params(self, params: dict_type):
        self.symbols = params["symbols"]


class RemovePunctuation(PreprocessBase):
    """
    移除标点符号
    """

    def __init__(self, cols="all", **kwargs):
        super().__init__(cols=cols, **kwargs)

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        for col in self.cols:
            s[col] = s[col].astype(str).str.replace(r"[^\w\s]", "")
        return s

    def udf_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        puns = r"[^\w\s]"
        for col in self.cols:
            s[col] = re.sub(puns, "", str(s[col]))
        return s

    def udf_get_params(self):
        return {}


class ReplacePunctuation(PreprocessBase):
    """
    替换标点符号
    """

    def __init__(self, cols="all", symbols="[p]", **kwargs):
        super().__init__(cols=cols, **kwargs)
        self.symbols = symbols

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        for col in self.cols:
            s[col] = s[col].astype(str).str.replace(r"[^\w\s]", self.symbols)
        return s

    def udf_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        puns = r"[^\w\s]"
        for col in self.cols:
            s[col] = re.sub(puns, self.symbols, str(s[col]))
        return s

    def udf_get_params(self):
        return {"symbols": self.symbols}

    def udf_set_params(self, params: dict_type):
        self.symbols = params["symbols"]


class Replace(PreprocessBase):
    """
    局部替换
    """

    def __init__(self, cols="all", source_values=None, target_value="", **kwargs):
        super().__init__(cols=cols, **kwargs)
        self.source_values = source_values
        self.target_value = target_value

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        for col in self.cols:
            for source_value in self.source_values:
                s[col] = s[col].astype(str).str.replace(source_value, self.target_value)
        return s

    def udf_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        for col in self.cols:
            for source_value in self.source_values:
                s[col] = str(s[col]).replace(source_value, self.target_value)
        return s

    def udf_get_params(self):
        return {"source_values": self.source_values, "target_value": self.target_value}

    def udf_set_params(self, params: dict_type):
        self.source_values = params["source_values"]
        self.target_value = params["target_value"]


class RemoveWhitespace(PreprocessBase):
    """
    移除空格，包括空格、制表符、回车符
    """

    def __init__(self, cols="all", **kwargs):
        super().__init__(cols=cols, **kwargs)

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        for col in self.cols:
            s[col] = s[col].astype(str).str.replace("\xa0", " ").str.split().str.join("")
        return s

    def udf_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        for col in self.cols:
            text = re.sub(r"\xa0", " ", str(s[col]))
            text = "".join(text.split())
            s[col] = text
        return s

    def udf_get_params(self):
        return {}


class RemoveStopWords(PreprocessBase):
    """
    移除停用词
    """

    def __init__(self, cols="all", stop_words=None, stop_words_path=None, **kwargs):
        super().__init__(cols=cols, **kwargs)
        self.stop_words = set()
        if stop_words is not None:
            for word in stop_words:
                self.stop_words.add(word)
        if stop_words_path is not None:
            for line in open(stop_words_path, encoding="utf8"):
                self.stop_words.add(line.strip())

    def apply_function(self, s):
        words = []
        for word in str(s).split(" "):
            if word not in self.stop_words:
                words.append(word)
        return " ".join(words)

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        for col in self.cols:
            s[col] = s[col].apply(lambda x: self.apply_function(x))
        return s

    def udf_transform_single(self, s: dict_type, **kwargs):
        for col in self.cols:
            s[col] = self.apply_function(s[col])
        return s

    def udf_get_params(self) -> dict_type:
        return {"stop_words": self.stop_words}

    def udf_set_params(self, params: dict_type):
        self.stop_words = params["stop_words"]


class ExtractKeyWords(PreprocessBase):
    """
    提取关键词
    """

    def __init__(self, cols="all", key_words=None, key_words_path=None, **kwargs):
        super().__init__(cols=cols, **kwargs)
        import ahocorasick
        self.actree = ahocorasick.Automaton()
        if key_words is not None:
            for word in key_words:
                self.actree.add_word(word, word)
        if key_words_path is not None:
            for line in open(key_words_path, encoding="utf8"):
                word = line.strip()
                self.actree.add_word(word, word)
        self.actree.make_automaton()

    def apply_function(self, s):
        words = []
        for i in self.actree.iter(str(s)):
            words.append(i[1])
        return " ".join(words)

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        for col in self.cols:
            s[col] = s[col].apply(lambda x: self.apply_function(x))
        return s

    def udf_transform_single(self, s: dict_type, **kwargs):
        for col in self.cols:
            s[col] = self.apply_function(s[col])
        return s

    def udf_get_params(self) -> dict_type:
        return {"actree": self.actree}

    def udf_set_params(self, params: dict_type):
        self.actree = params["actree"]


class AppendKeyWords(PreprocessBase):
    """
    提取关键词，并追加到原文后面
    """

    def __init__(self, cols="all", key_words=None, key_words_path=None, **kwargs):
        super().__init__(cols=cols, **kwargs)
        import ahocorasick
        self.actree = ahocorasick.Automaton()
        if key_words is not None:
            for word in key_words:
                self.actree.add_word(word, word)
        if key_words_path is not None:
            for line in open(key_words_path, encoding="utf8"):
                word = line.strip()
                self.actree.add_word(word, word)
        self.actree.make_automaton()

    def apply_function(self, s):
        words = []
        for i in self.actree.iter(s):
            words.append(i[1])
        return s + " " + " ".join(words)

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        for col in self.cols:
            s[col] = s[col].apply(lambda x: self.apply_function(x))
        return s

    def udf_transform_single(self, s: dict_type, **kwargs):
        for col in self.cols:
            s[col] = self.apply_function(s[col])
        return s

    def udf_get_params(self) -> dict_type:
        return {"actree": self.actree}

    def udf_set_params(self, params: dict_type):
        self.actree = params["actree"]


class ExtractChineseWords(PreprocessBase):
    """
    抽取中文字符
    """

    def __init__(self, cols="all", **kwargs):
        super().__init__(cols=cols, **kwargs)

    @staticmethod
    def apply_function(s):
        import re
        return "".join(re.findall(r'[\u4e00-\u9fa5]', str(s)))

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        for col in self.cols:
            s[col] = s[col].apply(lambda x: self.apply_function(x))
        return s

    def udf_transform_single(self, s: dict_type, **kwargs):
        for col in self.cols:
            s[col] = self.apply_function(s[col])
        return s

    def udf_get_params(self):
        return {}


class ExtractNGramWords(PreprocessBase):
    """
    抽取n-gram特征
    """

    def __init__(self, cols="all", n_grams=None, **kwargs):
        super().__init__(cols=cols, **kwargs)
        if n_grams is not None:
            self.n_grams = n_grams
        else:
            self.n_grams = [2]

    def apply_function(self, s):
        if " " in s:
            s = s.split(" ")
        words = []
        if 1 in self.n_grams:
            for word in s:
                words.append(word)
        if 2 in self.n_grams:
            for i in range(len(s) - 1):
                words.append("".join(s[i:i + 2]))
        if 3 in self.n_grams:
            for i in range(len(s) - 2):
                words.append("".join(s[i:i + 3]))
        return " ".join(words)

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        for col in self.cols:
            s[col] = s[col].apply(lambda x: self.apply_function(x))
        return s

    def udf_transform_single(self, s: dict_type, **kwargs):
        for col in self.cols:
            s[col] = self.apply_function(s[col])
        return s

    def udf_get_params(self) -> dict_type:
        return {"n_grams": self.n_grams}

    def udf_set_params(self, params: dict_type):
        self.n_grams = params["n_grams"]


class ExtractJieBaWords(PreprocessBase):
    """
    提取jieba分词
    """

    def __init__(self, cols="all", cut_all=False, **kwargs):
        super().__init__(cols=cols, **kwargs)
        self.cut_all = cut_all

    def apply_function(self, s):
        import jieba
        jieba.setLogLevel(jieba.logging.INFO)
        return " ".join(jieba.cut(str(s), cut_all=self.cut_all))

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        for col in self.cols:
            s[col] = s[col].apply(lambda x: self.apply_function(x))
        return s

    def udf_transform_single(self, s: dict_type, **kwargs):
        for col in self.cols:
            s[col] = self.apply_function(s[col])
        return s

    def udf_get_params(self) -> dict_type:
        return {"cut_all": self.cut_all}

    def udf_set_params(self, params: dict_type):
        self.cut_all = params["cut_all"]
