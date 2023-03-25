from easymlops.table.core import *
from easymlops.nlp.core import *
import numpy as np
import pandas as pd
from easymlops.table.utils import PandasUtils


class RepresentationBase(TablePipeObjectBase):
    """
    所有文本特征提取类的基础类
    """

    def __init__(self, cols="all", skip_check_transform_type=True, native_init_params=None, native_fit_params=None,
                 drop_input_data=True,
                 **kwargs):
        super().__init__(skip_check_transform_type=skip_check_transform_type, **kwargs)
        self.cols = cols
        # 底层模型自带参数
        self.native_init_params = copy.deepcopy(native_init_params)
        self.native_fit_params = copy.deepcopy(native_fit_params)
        if self.native_init_params is None:
            self.native_init_params = dict()
        if self.native_fit_params is None:
            self.native_fit_params = dict()
        self.drop_input_data = drop_input_data

    def before_fit(self, s: dataframe_type, **kwargs) -> dataframe_type:
        s = super().before_fit(s, **kwargs)
        if str(self.cols).lower() in ["none", "all", "null", "nan"]:
            self.cols = self.input_col_names
        assert type(self.cols) == list and type(self.cols[0]) == str
        return s

    def after_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        self.output_col_names = list(s.columns)
        return s.fillna(0)

    def transform_single(self, s: dict_type, **kwargs):
        return self.transform(pd.DataFrame([s])).to_dict("record")[0]

    def udf_get_params(self):
        return {"cols": self.cols, "native_init_params": self.native_init_params,
                "native_fit_params": self.native_fit_params, "drop_input_data": self.drop_input_data}

    def udf_set_params(self, params: dict_type):
        self.cols = params["cols"]
        self.native_init_params = params["native_init_params"]
        self.native_fit_params = params["native_fit_params"]
        self.drop_input_data = params["drop_input_data"]


class BagOfWords(RepresentationBase):
    """
    词袋模型
    """

    def __init__(self, cols="all", prefix="bag", **kwargs):
        super().__init__(cols=cols, prefix=prefix, **kwargs)
        self.tf = {}
        self.default_native_init_params = {"max_features": None, "tokenizer": self.tokenizer_func,
                                           "preprocessor": self.preprocessor_func, "min_df": 1, "max_df": 1.0,
                                           "binary": False}
        self.default_native_init_params.update(self.native_init_params)

    @staticmethod
    def tokenizer_func(x):
        return x.split(" ")

    @staticmethod
    def preprocessor_func(x):
        return x

    def udf_fit(self, s: dataframe_type, **kwargs):
        from sklearn.feature_extraction.text import CountVectorizer
        for col in self.cols:
            tf = CountVectorizer(**self.default_native_init_params)
            tf.fit(s[col], **self.native_fit_params)
            self.tf[col] = tf
        return self

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        # 先聚合稀疏矩阵，在添加稠密矩阵
        dfs = [] if self.drop_input_data else [s]
        for col in self.cols:
            tf = self.tf.get(col)
            tf_vectors_csr = tf.transform(s[col])
            try:
                feature_names = tf.get_feature_names()
            except:
                feature_names = tf.get_feature_names_out()
            df = pd.DataFrame.sparse.from_spmatrix(data=tf_vectors_csr,
                                                   columns=["{}_{}_{}".format(self.prefix, col, name) for name in
                                                            feature_names], index=s.index)
            dfs.append(df)

        return PandasUtils.concat_duplicate_columns(dfs)

    def udf_get_params(self) -> dict_type:
        return {"tf": self.tf}

    def udf_set_params(self, params: dict_type):
        self.tf = params["tf"]


class TFIDF(RepresentationBase):
    """
    TFIDF模型
    """

    def __init__(self, cols="all", prefix="tfidf", **kwargs):
        super().__init__(cols=cols, prefix=prefix, **kwargs)
        self.tfidf = {}
        self.default_native_init_params = {"max_features": None, "tokenizer": self.tokenizer_func,
                                           "preprocessor": self.preprocessor_func, "min_df": 1, "max_df": 1.0,
                                           "binary": False}
        self.default_native_init_params.update(self.native_init_params)

    @staticmethod
    def tokenizer_func(x):
        return x.split(" ")

    @staticmethod
    def preprocessor_func(x):
        return x

    def udf_fit(self, s: dataframe_type, **kwargs):
        from sklearn.feature_extraction.text import TfidfVectorizer
        for col in self.cols:
            tfidf = TfidfVectorizer(**self.default_native_init_params)
            tfidf.fit(s[col], **self.native_fit_params)
            self.tfidf[col] = tfidf
        return self

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        dfs = [] if self.drop_input_data else [s]
        for col in self.cols:
            tfidf = self.tfidf.get(col)
            tfidf_vectors_csr = tfidf.transform(s[col])
            try:
                feature_names = tfidf.get_feature_names()
            except:
                feature_names = tfidf.get_feature_names_out()
            df = pd.DataFrame.sparse.from_spmatrix(data=tfidf_vectors_csr,
                                                   columns=["{}_{}_{}".format(self.prefix, col, name) for name in
                                                            feature_names], index=s.index)
            dfs.append(df)

        return PandasUtils.concat_duplicate_columns(dfs)

    def udf_get_params(self) -> dict_type:
        return {"tfidf": self.tfidf}

    def udf_set_params(self, params: dict_type):
        self.tfidf = params["tfidf"]


class LdaTopicModel(RepresentationBase):
    """
    LDA主题模型
    """

    def __init__(self, cols="all", num_topics=10, prefix="lda", **kwargs):
        super().__init__(cols=cols, prefix=prefix, **kwargs)
        self.native_init_params.update({"num_topics": num_topics})
        self.native_init_params.update(self.native_fit_params)
        self.common_dictionary = {}
        self.lda_model = {}

    def udf_fit(self, s: dataframe_type, **kwargs):
        from gensim.corpora.dictionary import Dictionary
        from gensim.models.ldamulticore import LdaModel
        for col in self.cols:
            texts = s[col].values.tolist()
            texts = [line.split(" ") for line in texts]
            common_dictionary = Dictionary(texts)
            common_corpus = [common_dictionary.doc2bow(text) for text in texts]
            lda_model = LdaModel(common_corpus, **self.native_init_params)
            self.lda_model[col] = lda_model
            self.common_dictionary[col] = common_dictionary
        return self

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        from gensim import matutils
        dfs = [] if self.drop_input_data else [s]
        for col in self.cols:
            common_dictionary = self.common_dictionary[col]
            lda_model = self.lda_model[col]
            texts = s[col].values.tolist()
            texts = [line.split(" ") for line in texts]
            common_corpus = [common_dictionary.doc2bow(text) for text in texts]
            vectors = matutils.corpus2dense(lda_model[common_corpus],
                                            num_terms=self.native_init_params.get("num_topics")).T
            result = pd.DataFrame(vectors, index=s.index)
            result.columns = ["{}_{}_{}".format(self.prefix, col, name) for name in result.columns]
            dfs.append(result)

        return PandasUtils.concat_duplicate_columns(dfs)

    def udf_get_params(self) -> dict_type:
        return {"common_dictionary": self.common_dictionary, "lda_model": self.lda_model}

    def udf_set_params(self, params: dict):
        self.common_dictionary = params["common_dictionary"]
        self.lda_model = params["lda_model"]


class LsiTopicModel(RepresentationBase):
    """
    LSI主题模型
    """

    def __init__(self, cols="all", num_topics=10, prefix="lsi", **kwargs):
        super().__init__(cols=cols, prefix=prefix, **kwargs)
        self.native_init_params.update({"num_topics": num_topics})
        self.native_init_params.update(self.native_fit_params)
        self.common_dictionary = {}
        self.lsi_model = {}

    def udf_fit(self, s: dataframe_type, **kwargs):
        from gensim.corpora.dictionary import Dictionary
        from gensim.models import LsiModel
        for col in self.cols:
            texts = s[col].values.tolist()
            texts = [line.split(" ") for line in texts]
            common_dictionary = Dictionary(texts)
            common_corpus = [common_dictionary.doc2bow(text) for text in texts]
            lsi_model = LsiModel(common_corpus, id2word=common_dictionary, **self.native_init_params)
            self.lsi_model[col] = lsi_model
            self.common_dictionary[col] = common_dictionary
        return self

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        from gensim import matutils
        dfs = [] if self.drop_input_data else [s]
        for col in self.cols:
            lsi_model = self.lsi_model[col]
            common_dictionary = self.common_dictionary[col]
            texts = s[col].values.tolist()
            texts = [line.split(" ") for line in texts]
            common_corpus = [common_dictionary.doc2bow(text) for text in texts]
            vectors = matutils.corpus2dense(lsi_model[common_corpus],
                                            num_terms=self.native_init_params.get("num_topics")).T
            result = pd.DataFrame(vectors, index=s.index)
            result.columns = ["{}_{}_{}".format(self.prefix, col, name) for name in result.columns]
            dfs.append(result)
        return PandasUtils.concat_duplicate_columns(dfs)

    def udf_get_params(self) -> dict_type:
        return {"common_dictionary": self.common_dictionary, "lsi_model": self.lsi_model}

    def udf_set_params(self, params: dict):
        self.common_dictionary = params["common_dictionary"]
        self.lsi_model = params["lsi_model"]


class Word2VecModel(RepresentationBase):
    """
    Word2Vec词向量模型
    """

    def __init__(self, cols="all", embedding_size=16, min_count=5, prefix="w2v", **kwargs):
        super().__init__(cols=cols, prefix=prefix, **kwargs)
        self.w2v_model = {}
        self.native_init_params.update(self.native_fit_params)
        self.native_init_params.update({"vector_size": embedding_size, "min_count": min_count})

    def udf_fit(self, s: dataframe_type, **kwargs):
        from gensim.models import Word2Vec
        for col in self.cols:
            texts = s[col].values.tolist()
            texts = [line.split(" ") for line in texts]
            w2v_model = Word2Vec(sentences=texts, **self.native_init_params)
            self.w2v_model[col] = w2v_model
        return self

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        dfs = [] if self.drop_input_data else [s]
        for col in self.cols:
            w2v_model = self.w2v_model[col]
            texts = s[col].values.tolist()
            texts = [line.split(" ") for line in texts]
            vectors = [np.mean(np.asarray([w2v_model.wv[word] for word in line if word in w2v_model.wv]),
                               axis=0) + np.zeros(shape=(self.native_init_params.get("vector_size"),))
                       for line in texts]
            result = pd.DataFrame(vectors, index=s.index)
            result.columns = ["{}_{}_{}".format(self.prefix, col, name) for name in result.columns]
            dfs.append(result)

        return PandasUtils.concat_duplicate_columns(dfs)

    def udf_get_params(self) -> dict_type:
        return {"w2v_model": self.w2v_model}

    def udf_set_params(self, params: dict):
        self.w2v_model = params["w2v_model"]


class Doc2VecModel(RepresentationBase):
    """
    Doc2Vec词向量模型
    """

    def __init__(self, cols="all", embedding_size=16, min_count=5, prefix="d2v", **kwargs):
        super().__init__(cols=cols, prefix=prefix, **kwargs)
        self.d2v_model = {}
        self.native_init_params.update(self.native_fit_params)
        self.native_init_params.update({"vector_size": embedding_size, "min_count": min_count})

    def udf_fit(self, s: dataframe_type, **kwargs):
        from gensim.models.doc2vec import Doc2Vec, TaggedDocument
        for col in self.cols:
            texts = s[col].values.tolist()
            texts = [line.split(" ") for line in texts]
            documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(texts)]
            d2v_model = Doc2Vec(documents, **self.native_init_params)
            self.d2v_model[col] = d2v_model
        return self

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        dfs = [] if self.drop_input_data else [s]
        for col in self.cols:
            d2v_model = self.d2v_model[col]
            texts = s[col].values.tolist()
            texts = [line.split(" ") for line in texts]
            vectors = [d2v_model.infer_vector(line) for line in texts]
            result = pd.DataFrame(vectors, index=s.index)
            result.columns = ["{}_{}_{}".format(self.prefix, col, name) for name in result.columns]
            dfs.append(result)

        return PandasUtils.concat_duplicate_columns(dfs)

    def udf_get_params(self) -> dict_type:
        return {"d2v_model": self.d2v_model}

    def udf_set_params(self, params: dict):
        self.d2v_model = params["d2v_model"]


class FastTextModel(RepresentationBase):
    """
    FastText词向量模型
    """

    def __init__(self, cols="all", embedding_size=16, min_count=5, prefix="fasttext", **kwargs):
        super().__init__(cols=cols, prefix=prefix, **kwargs)
        self.fasttext_model = {}
        self.native_init_params.update(self.native_fit_params)
        self.native_init_params.update({"vector_size": embedding_size, "min_count": min_count})

    def udf_fit(self, s: dataframe_type, **kwargs):
        from gensim.models import FastText
        for col in self.cols:
            texts = s[col].values.tolist()
            texts = [line.split(" ") for line in texts]
            fasttext_model = FastText(sentences=texts, **self.native_init_params)
            self.fasttext_model[col] = fasttext_model
        return self

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        dfs = [] if self.drop_input_data else [s]
        for col in self.cols:
            fasttext_model = self.fasttext_model[col]
            texts = s[col].values.tolist()
            texts = [line.split(" ") for line in texts]
            vectors = [np.mean(np.asarray([fasttext_model.wv[word]
                                           for word in line if word in fasttext_model.wv]), axis=0) + np.zeros(
                shape=(self.native_init_params.get("vector_size"),))
                       for line in texts]
            result = pd.DataFrame(vectors, index=s.index)
            result.columns = ["{}_{}_{}".format(self.prefix, col, name) for name in result.columns]
            dfs.append(result)

        return PandasUtils.concat_duplicate_columns(dfs)

    def udf_get_params(self) -> dict_type:
        return {"fasttext_model": self.fasttext_model}

    def udf_set_params(self, params: dict_type):
        self.fasttext_model = params["fasttext_model"]
