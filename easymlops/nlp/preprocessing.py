import re
import string

from ..base import *


class PreprocessBase(PipeObject):
    def __init__(self, cols="all", skip_check_transform_type=True, **kwargs):
        super().__init__(skip_check_transform_type=skip_check_transform_type, **kwargs)
        self.cols = cols

    def before_fit(self, s: dataframe_type) -> dataframe_type:
        s = super().before_fit(s)
        if str(self.cols).lower() in ["none", "all", "null"]:
            self.cols = self.input_col_names
        assert type(self.cols) == list and type(self.cols[0]) == str
        return s

    def _get_params(self):
        return {"cols": self.cols}

    def _set_params(self, params: dict_type):
        self.cols = params["cols"]


class Lower(PreprocessBase):
    """
    input type:pandas.series
    input like:
    |input|
    |Abc中午|
    |jKK|
    -------------------------
    output type:pandas.series
    output like:
    |output|
    |abc中午|
    |jkk|
    """

    def __init__(self, cols="all", **kwargs):
        super().__init__(cols=cols, **kwargs)

    def _transform(self, s: dataframe_type) -> dataframe_type:
        for col in self.cols:
            s[col] = s[col].astype(str).str.lower()
        return s

    def _transform_single(self, s: dict_type) -> dict_type:
        for col in self.cols:
            s[col] = str(s[col]).lower()
        return s

    def _get_params(self):
        return {}


class Upper(PreprocessBase):
    """
    input type:pandas.series
    input like:
    |input|
    |Abc中午|
    |jKK|
    -------------------------
    output type:pandas.series
    output like:
    |output|
    |ABC中午|
    |JKK|
    """

    def __init__(self, cols="all", **kwargs):
        super().__init__(cols=cols, **kwargs)

    def _transform(self, s: dataframe_type) -> dataframe_type:
        for col in self.cols:
            s[col] = s[col].astype(str).str.upper()
        return s

    def _transform_single(self, s: dict_type) -> dict_type:
        for col in self.cols:
            s[col] = str(s[col]).upper()
        return s

    def _get_params(self):
        return {}


class RemoveDigits(PreprocessBase):
    """
    input type:pandas.series
    input like:
    |input|
    |abc123|
    |j1k2|
    -------------------------
    output type:pandas.series
    output like:
    |output|
    |abc|
    |jk|
    """

    def __init__(self, cols="all", **kwargs):
        super().__init__(cols=cols, **kwargs)

    def _transform(self, s: dataframe_type) -> dataframe_type:
        for col in self.cols:
            s[col] = s[col].astype(str).str.replace(r"\d+", "")
        return s

    def _transform_single(self, s: dict_type) -> dict_type:
        for col in self.cols:
            s[col] = re.sub(r"\d+", "", str(s[col]))
        return s

    def _get_params(self):
        return {}


class ReplaceDigits(PreprocessBase):
    """
    input type:pandas.series
    input like:
    |input|
    |abc123|
    |j1k2|
    -------------------------
    output type:pandas.series
    output like:
    |output|
    |abc[d]|
    |j[d]k[d]|
    """

    def __init__(self, cols="all", symbols="[d]", **kwargs):
        super().__init__(cols=cols, **kwargs)
        self.symbols = symbols

    def _transform(self, s: dataframe_type) -> dataframe_type:
        for col in self.cols:
            s[col] = s[col].astype(str).str.replace(r"\d+", self.symbols)
        return s

    def _transform_single(self, s: dict_type) -> dict_type:
        for col in self.cols:
            s[col] = re.sub(r"\d+", self.symbols, str(s[col]))
        return s

    def _get_params(self):
        return {"symbols": self.symbols}

    def _set_params(self, params: dict_type):
        self.symbols = params["symbols"]


class RemovePunctuation(PreprocessBase):
    """
    input type:pandas.series
    input like:
    |input|
    |ab,cd;ef|
    |abc!|
    -------------------------
    output type:pandas.series
    output like:
    |output|
    |abcdef|
    |abc|
    """

    def __init__(self, cols="all", **kwargs):
        super().__init__(cols=cols, **kwargs)

    def _transform(self, s: dataframe_type) -> dataframe_type:
        for col in self.cols:
            s[col] = s[col].astype(str).str.replace(rf"([{string.punctuation}+'，。！（）“‘？：；】【、'])+", "")
        return s

    def _transform_single(self, s: dict_type) -> dict_type:
        puns = rf"([{string.punctuation}+'，。！（）“‘？：；】【、'])+"
        for col in self.cols:
            s[col] = re.sub(puns, "", str(s[col]))
        return s

    def _get_params(self):
        return {}


class ReplacePunctuation(PreprocessBase):
    """
    input type:pandas.series
    input like:
    |input|
    |ab,cd;ef|
    |abc!|
    -------------------------
    output type:pandas.series
    output like:
    |output|
    |ab[p]cd[p]ef|
    |abc[p]|
    """

    def __init__(self, cols="all", symbols="[p]", **kwargs):
        super().__init__(cols=cols, **kwargs)
        self.symbols = symbols

    def _transform(self, s: dataframe_type) -> dataframe_type:
        for col in self.cols:
            s[col] = s[col].astype(str).str.replace(rf"([{string.punctuation}+'，。！（）“‘？：；】【、'])+", self.symbols)
        return s

    def _transform_single(self, s: dict_type) -> dict_type:
        puns = rf"([{string.punctuation}+'，。！（）“‘？：；】【、'])+"
        for col in self.cols:
            s[col] = re.sub(puns, self.symbols, str(s[col]))
        return s

    def _get_params(self):
        return {"symbols": self.symbols}

    def _set_params(self, params: dict_type):
        self.symbols = params["symbols"]


class RemoveWhitespace(PreprocessBase):
    """
    input type:pandas.series
    input like:
    |input|
    |ab cd ef|
    |a b c|
    -------------------------
    output type:pandas.series
    output like:
    |output|
    |abcdef|
    |abc|
    """

    def __init__(self, cols="all", **kwargs):
        super().__init__(cols=cols, **kwargs)

    def _transform(self, s: dataframe_type) -> dataframe_type:
        for col in self.cols:
            s[col] = s[col].astype(str).str.replace("\xa0", " ").str.split().str.join("")
        return s

    def _transform_single(self, s: dict_type) -> dict_type:
        for col in self.cols:
            text = re.sub(r"\xa0", " ", str(s[col]))
            text = "".join(text.split())
            s[col] = text
        return s

    def _get_params(self):
        return {}


class RemoveStopWords(PreprocessBase):
    """
    stop_words=["ab"]
    ----------------------------
    input type:pandas.series
    input like:(space separation)
    |input|
    |ab cd ef|
    |abc|
    -------------------------
    output type:pandas.series
    output like:(space separation)
    |output|
    |cd ef|
    |abc|
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

    def _transform(self, s: dataframe_type) -> dataframe_type:
        for col in self.cols:
            s[col] = s[col].apply(lambda x: self.apply_function(x))
        return s

    def _transform_single(self, s: dict_type):
        for col in self.cols:
            s[col] = self.apply_function(s[col])
        return s

    def _get_params(self) -> dict_type:
        return {"stop_words": self.stop_words}

    def _set_params(self, params: dict_type):
        self.stop_words = params["stop_words"]


class ExtractKeyWords(PreprocessBase):
    """
    key_words=["ab","cd"]
    ----------------------------
    input type:pandas.series
    input like:(space separation)
    |input|
    |ab cd ef|
    |abc def|
    -------------------------
    output type:pandas.series
    output like:(space separation)
    |output|
    |ab cd|
    |ab|
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
        return " ".join(words)

    def _transform(self, s: dataframe_type) -> dataframe_type:
        for col in self.cols:
            s[col] = s[col].apply(lambda x: self.apply_function(x))
        return s

    def _transform_single(self, s: dict_type):
        for col in self.cols:
            s[col] = self.apply_function(s[col])
        return s

    def _get_params(self) -> dict_type:
        return {"actree": self.actree}

    def _set_params(self, params: dict_type):
        self.actree = params["actree"]


class AppendKeyWords(PreprocessBase):
    """
    key_words=["ab","cd"]
    ----------------------------
    input type:pandas.series
    input like:(space separation)
    |input|
    |ab cd ef|
    |abc def|
    -------------------------
    output type:pandas.series
    output like:(space separation)
    |output|
    |ab cd ef ab cd|
    |abc def ab|
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

    def _transform(self, s: dataframe_type) -> dataframe_type:
        for col in self.cols:
            s[col] = s[col].apply(lambda x: self.apply_function(x))
        return s

    def _transform_single(self, s: dict_type):
        for col in self.cols:
            s[col] = self.apply_function(s[col])
        return s

    def _get_params(self) -> dict_type:
        return {"actree": self.actree}

    def _set_params(self, params: dict_type):
        self.actree = params["actree"]


class ExtractChineseWords(PreprocessBase):
    """
    input type:pandas.series
    input like:
    |input|
    |ab中文cd， e输入f|
    |a中b文c|
    -------------------------
    output type:pandas.series
    output like:
    |output|
    |中文输入|
    |中文|
    """

    def __init__(self, cols="all", **kwargs):
        super().__init__(cols=cols, **kwargs)

    @staticmethod
    def apply_function(s):
        import re
        return "".join(re.findall(r'[\u4e00-\u9fa5]', s))

    def _transform(self, s: dataframe_type) -> dataframe_type:
        for col in self.cols:
            s[col] = s[col].apply(lambda x: self.apply_function(x))
        return s

    def _transform_single(self, s: dict_type):
        for col in self.cols:
            s[col] = self.apply_function(s[col])
        return s

    def _get_params(self):
        return {}


class ExtractNGramWords(PreprocessBase):
    """
    demo1:
    n_grams=[2]
    ----------------------------
    input type:pandas.series
    input like:(space separation)
    |input|
    |ab cd ef|
    |abc def|
    -------------------------
    output type:pandas.series
    output like:(space separation)
    |output|
    |abcd cdef|
    |abcdef|
    -------------------------
    demo2:
    n_grams=[1,2]
    ----------------------------
    input type:pandas.series
    input like:(space separation)
    |input|
    |ab cd ef|
    |abc def|
    -------------------------
    output type:pandas.series
    output like:(space separation)
    |output|
    |ab cd ef abcd cdef|
    |abc def abcdef|
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

    def _transform(self, s: dataframe_type) -> dataframe_type:
        for col in self.cols:
            s[col] = s[col].apply(lambda x: self.apply_function(x))
        return s

    def _transform_single(self, s: dict_type):
        for col in self.cols:
            s[col] = self.apply_function(s[col])
        return s

    def _get_params(self) -> dict_type:
        return {"n_grams": self.n_grams}

    def _set_params(self, params: dict_type):
        self.n_grams = params["n_grams"]


class ExtractJieBaWords(PreprocessBase):
    """
    input type:pandas.series
    input like:
    |input|
    |北京天安门|
    |北京清华大学|
    -------------------------
    output type:pandas.series
    output like:(space separation)
    |output|
    |北京 天安门|
    |北京 清华大学|
    """

    def __init__(self, cols="all", cut_all=False, **kwargs):
        super().__init__(cols=cols, **kwargs)
        self.cut_all = cut_all

    def apply_function(self, s):
        import jieba
        jieba.setLogLevel(jieba.logging.INFO)
        return " ".join(jieba.cut(s, cut_all=self.cut_all))

    def _transform(self, s: dataframe_type) -> dataframe_type:
        for col in self.cols:
            s[col] = s[col].apply(lambda x: self.apply_function(x))
        return s

    def _transform_single(self, s: dict_type):
        for col in self.cols:
            s[col] = self.apply_function(s[col])
        return s

    def _get_params(self) -> dict_type:
        return {"cut_all": self.cut_all}

    def _set_params(self, params: dict_type):
        self.cut_all = params["cut_all"]
