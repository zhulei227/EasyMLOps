from ..base import *
import numpy as np
import pandas as pd


class UserDefinedRepresentation(PipeObject):
    def __init__(self, cols="all", name=None, prefix=None, copy_transform_data=False,
                 transform_check_max_number_error=1e-5, skip_check_transform_type=True):
        super().__init__(name=name, prefix=prefix, copy_transform_data=copy_transform_data,
                         transform_check_max_number_error=transform_check_max_number_error,
                         skip_check_transform_type=skip_check_transform_type)
        self.cols = cols

    def before_fit(self, s: dataframe_type) -> dataframe_type:
        s = super().before_fit(s)
        if str(self.cols).lower() in ["none", "all", "null", "nan"]:
            self.cols = self.input_col_names
        assert type(self.cols) == list and type(self.cols[0]) == str
        return s

    def after_transform(self, s: dataframe_type) -> dataframe_type:
        self.output_col_names = list(s.columns)
        s = s.fillna(0)
        return s

    def after_transform_single(self, s: dict_type) -> dict_type:
        self.extract_dict(s, self.output_col_names)
        for key, value in s.items():
            if str(value).lower() in ["none", "null", "nan"]:
                s[key] = 0
        return s

    def _transform_single(self, s: dict_type):
        return self._transform(pd.DataFrame([s])).to_dict("record")[0]

    def _get_params(self):
        return {"cols": self.cols}

    def _set_params(self, params: dict_type):
        self.cols = params["cols"]


class BagOfWords(UserDefinedRepresentation):
    """
    input type:pandas.series
    input like:(space separation)
    |input|
    |i love eat apple|
    |i love china|
    ---------------------------
    output type:pandas.dataframe
    output like:
    | i |love|eat|apple|china|
    | 1 |  1 | 1 |  1  |  0  |
    | 1 |  1 | 0 |  0  |  1  |
    """

    def __init__(self, cols="all", name=None, transform_check_max_number_error=1e-5, skip_check_transform_type=True,
                 prefix="bag"):
        super().__init__(cols=cols, name=name, transform_check_max_number_error=transform_check_max_number_error,
                         skip_check_transform_type=skip_check_transform_type, prefix=prefix)
        self.tf = {}

    @staticmethod
    def tokenizer_func(x):
        return x.split(" ")

    @staticmethod
    def preprocessor_func(x):
        return x

    def _fit(self, s: dataframe_type) -> dataframe_type:
        from sklearn.feature_extraction.text import CountVectorizer
        for col in self.cols:
            tf = CountVectorizer(
                max_features=None,
                tokenizer=self.tokenizer_func,
                preprocessor=self.preprocessor_func,
                min_df=1,
                max_df=1.0,
                binary=False,
            )
            tf.fit(s[col])
            self.tf[col] = tf
        return self

    def _transform(self, s: dataframe_type) -> dataframe_type:
        for col in self.cols:
            tf = self.tf.get(col)
            tf_vectors_csr = tf.transform(s[col])
            try:
                feature_names = tf.get_feature_names()
            except:
                feature_names = tf.get_feature_names_out()
            df = pandas.DataFrame.sparse.from_spmatrix(data=tf_vectors_csr,
                                                       columns=["{}_{}_{}".format(self.prefix, col, name) for name in
                                                                feature_names])
            for icol in df.columns:
                s[icol] = df[icol]
                s[icol] = df[icol].values
        return s

    def _get_params(self) -> dict_type:
        return {"tf": self.tf}

    def _set_params(self, params: dict_type):
        self.tf = params["tf"]


class TFIDF(UserDefinedRepresentation):
    """
    input type:pandas.series
    input like:(space separation)
    |input|
    |i love eat apple|
    |i love china|
    -----------------------------
    output type:pandas.dataframe
    output like:
    | i |love|eat|apple|china|
    |0.2|0.3 |0.4| 0.5 |  0  |
    |0.3|0.2 | 0 |  0  | 0.2 |
    """

    def __init__(self, cols="all", name=None, transform_check_max_number_error=1e-5, skip_check_transform_type=True,
                 prefix="tfidf"):
        super().__init__(cols=cols, name=name, transform_check_max_number_error=transform_check_max_number_error,
                         skip_check_transform_type=skip_check_transform_type, prefix=prefix)
        self.tfidf = {}

    @staticmethod
    def tokenizer_func(x):
        return x.split(" ")

    @staticmethod
    def preprocessor_func(x):
        return x

    def _fit(self, s: dataframe_type) -> dataframe_type:
        from sklearn.feature_extraction.text import TfidfVectorizer
        for col in self.cols:
            tfidf = TfidfVectorizer(
                max_features=None,
                tokenizer=self.tokenizer_func,
                preprocessor=self.preprocessor_func,
                min_df=1,
                max_df=1.0,
                binary=False,
            )
            tfidf.fit(s[col])
            self.tfidf[col] = tfidf
        return self

    def _transform(self, s: dataframe_type) -> dataframe_type:
        for col in self.cols:
            tfidf = self.tfidf.get(col)
            tfidf_vectors_csr = tfidf.transform(s[col])
            try:
                feature_names = tfidf.get_feature_names()
            except:
                feature_names = tfidf.get_feature_names_out()
            df = pandas.DataFrame.sparse.from_spmatrix(data=tfidf_vectors_csr,
                                                       columns=["{}_{}_{}".format(self.prefix, col, name) for name in
                                                                feature_names])
            for icol in df.columns:
                s[icol] = df[icol]
                s[icol] = df[icol].values
        return s

    def _get_params(self) -> dict_type:
        return {"tfidf": self.tfidf}

    def _set_params(self, params: dict_type):
        self.tfidf = params["tfidf"]


class LdaTopicModel(UserDefinedRepresentation):
    """
    input type:pandas.series
    input like:(space separation)
    |input|
    |i love eat apple|
    |i love china|
    ----------------------------
    output type:pandas.dataframe
    output like:
    | 0 | 1 | 2 | 3 |
    |0.2|0.3|0.4|0.5|
    |0.3|0.2|0.6|0.1|
    """

    def __init__(self, cols="all", num_topics=10, name=None, transform_check_max_number_error=1e-1,
                 skip_check_transform_type=True, prefix="lda"):
        super().__init__(cols=cols, name=name, transform_check_max_number_error=transform_check_max_number_error,
                         skip_check_transform_type=skip_check_transform_type, prefix=prefix)
        self.num_topics = num_topics
        self.common_dictionary = {}
        self.lda_model = {}

    def _fit(self, s: dataframe_type) -> dataframe_type:
        from gensim.corpora.dictionary import Dictionary
        from gensim.models.ldamulticore import LdaModel
        for col in self.cols:
            texts = s[col].values.tolist()
            texts = [line.split(" ") for line in texts]
            common_dictionary = Dictionary(texts)
            common_corpus = [common_dictionary.doc2bow(text) for text in texts]
            lda_model = LdaModel(common_corpus, num_topics=self.num_topics)
            self.lda_model[col] = lda_model
            self.common_dictionary[col] = common_dictionary
        return self

    def _transform(self, s: dataframe_type) -> dataframe_type:
        from gensim import matutils
        for col in self.cols:
            common_dictionary = self.common_dictionary[col]
            lda_model = self.lda_model[col]
            texts = s[col].values.tolist()
            texts = [line.split(" ") for line in texts]
            common_corpus = [common_dictionary.doc2bow(text) for text in texts]
            vectors = matutils.corpus2dense(lda_model[common_corpus], num_terms=self.num_topics).T
            result = pandas.DataFrame(vectors)
            result.columns = ["{}_{}_{}".format(self.prefix, col, name) for name in result.columns]
            for icol in result.columns:
                s[icol] = result[icol]
                s[icol] = result[icol].values
        return s

    def _get_params(self) -> dict_type:
        return {"num_topics": self.num_topics, "common_dictionary": self.common_dictionary, "lda_model": self.lda_model}

    def _set_params(self, params: dict):
        self.num_topics = params["num_topics"]
        self.common_dictionary = params["common_dictionary"]
        self.lda_model = params["lda_model"]


class LsiTopicModel(UserDefinedRepresentation):
    """
    input type:pandas.series
    input like:(space separation)
    |input|
    |i love eat apple|
    |i love china|
    ----------------------------
    output type:pandas.dataframe
    output like:
    | 0 | 1 | 2 | 3 |
    |0.2|0.3|0.4|0.5|
    |0.3|0.2|0.6|0.1|
    """

    def __init__(self, cols="all", num_topics=10, name=None, transform_check_max_number_error=1e-5,
                 skip_check_transform_type=True, prefix="lsi"):
        super().__init__(cols=cols, name=name, transform_check_max_number_error=transform_check_max_number_error,
                         skip_check_transform_type=skip_check_transform_type, prefix=prefix)
        self.num_topics = num_topics
        self.common_dictionary = {}
        self.lsi_model = {}

    def _fit(self, s: dataframe_type) -> dataframe_type:
        from gensim.corpora.dictionary import Dictionary
        from gensim.models import LsiModel
        for col in self.cols:
            texts = s[col].values.tolist()
            texts = [line.split(" ") for line in texts]
            common_dictionary = Dictionary(texts)
            common_corpus = [common_dictionary.doc2bow(text) for text in texts]
            lsi_model = LsiModel(common_corpus, num_topics=self.num_topics, id2word=common_dictionary)
            self.lsi_model[col] = lsi_model
            self.common_dictionary[col] = common_dictionary
        return self

    def _transform(self, s: dataframe_type) -> dataframe_type:
        from gensim import matutils
        for col in self.cols:
            lsi_model = self.lsi_model[col]
            common_dictionary = self.common_dictionary[col]
            texts = s[col].values.tolist()
            texts = [line.split(" ") for line in texts]
            common_corpus = [common_dictionary.doc2bow(text) for text in texts]
            vectors = matutils.corpus2dense(lsi_model[common_corpus], num_terms=self.num_topics).T
            result = pandas.DataFrame(vectors)
            result.columns = ["{}_{}_{}".format(self.prefix, col, name) for name in result.columns]
            for icol in result.columns:
                s[icol] = result[icol]
                s[icol] = result[icol].values
        return s

    def _get_params(self) -> dict_type:
        return {"num_topics": self.num_topics, "common_dictionary": self.common_dictionary, "lsi_model": self.lsi_model}

    def _set_params(self, params: dict):
        self.num_topics = params["num_topics"]
        self.common_dictionary = params["common_dictionary"]
        self.lsi_model = params["lsi_model"]


class Word2VecModel(UserDefinedRepresentation):
    """
    input type:pandas.series
    input like:(space separation)
    |input|
    |i love eat apple|
    |i love china|
    ----------------------------
    output type:pandas.dataframe
    output like:
    | 0 | 1 | 2 | 3 |
    |0.2|0.3|0.4|0.5|
    |0.3|0.2|0.6|0.1|
    """

    def __init__(self, cols="all", embedding_size=16, min_count=5, name=None, transform_check_max_number_error=1e-1,
                 skip_check_transform_type=True, prefix="w2v"):
        super().__init__(cols=cols, name=name, transform_check_max_number_error=transform_check_max_number_error,
                         skip_check_transform_type=skip_check_transform_type, prefix=prefix)
        self.min_count = min_count
        self.embedding_size = embedding_size
        self.w2v_model = {}

    def _fit(self, s: dataframe_type) -> dataframe_type:
        from gensim.models import Word2Vec
        for col in self.cols:
            texts = s[col].values.tolist()
            texts = [line.split(" ") for line in texts]
            w2v_model = Word2Vec(sentences=texts, vector_size=self.embedding_size, min_count=self.min_count)
            self.w2v_model[col] = w2v_model
        return self

    def _transform(self, s: dataframe_type) -> dataframe_type:
        for col in self.cols:
            w2v_model = self.w2v_model[col]
            texts = s[col].values.tolist()
            texts = [line.split(" ") for line in texts]
            vectors = [np.mean(np.asarray([w2v_model.wv[word] for word in line if word in w2v_model.wv]),
                               axis=0) + np.zeros(shape=(self.embedding_size,))
                       for line in texts]
            result = pandas.DataFrame(vectors)
            result.columns = ["{}_{}_{}".format(self.prefix, col, name) for name in result.columns]
            for icol in result.columns:
                s[icol] = result[icol]
                s[icol] = result[icol].values
        return s

    def _get_params(self) -> dict_type:
        return {"embedding_size": self.embedding_size, "w2v_model": self.w2v_model, "min_count": self.min_count}

    def _set_params(self, params: dict):
        self.embedding_size = params["embedding_size"]
        self.w2v_model = params["w2v_model"]
        self.min_count = params["min_count"]


class Doc2VecModel(UserDefinedRepresentation):
    """
    input type:pandas.series
    input like:(space separation)
    |input|
    |i love eat apple|
    |i love china|
    ----------------------------
    output type:pandas.dataframe
    output like:
    | 0 | 1 | 2 | 3 |
    |0.2|0.3|0.4|0.5|
    |0.3|0.2|0.6|0.1|
    """

    def __init__(self, cols="all", embedding_size=16, min_count=5, name=None, transform_check_max_number_error=1e-1,
                 skip_check_transform_type=True, prefix="d2v"):
        super().__init__(cols=cols, name=name, transform_check_max_number_error=transform_check_max_number_error,
                         skip_check_transform_type=skip_check_transform_type, prefix=prefix)
        self.min_count = min_count
        self.embedding_size = embedding_size
        self.d2v_model = {}

    def _fit(self, s: dataframe_type) -> dataframe_type:
        from gensim.models.doc2vec import Doc2Vec, TaggedDocument
        for col in self.cols:
            texts = s[col].values.tolist()
            texts = [line.split(" ") for line in texts]
            documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(texts)]
            d2v_model = Doc2Vec(documents, vector_size=self.embedding_size, min_count=self.min_count)
            self.d2v_model[col] = d2v_model
        return self

    def _transform(self, s: dataframe_type) -> dataframe_type:
        for col in self.cols:
            d2v_model = self.d2v_model[col]
            texts = s[col].values.tolist()
            texts = [line.split(" ") for line in texts]
            vectors = [d2v_model.infer_vector(line) for line in texts]
            result = pandas.DataFrame(vectors)
            result.columns = ["{}_{}_{}".format(self.prefix, col, name) for name in result.columns]
            for icol in result.columns:
                s[icol] = result[icol]
                s[icol] = result[icol].values
        return s

    def _get_params(self) -> dict_type:
        return {"embedding_size": self.embedding_size, "d2v_model": self.d2v_model, "min_count": self.min_count}

    def _set_params(self, params: dict):
        self.embedding_size = params["embedding_size"]
        self.d2v_model = params["d2v_model"]
        self.min_count = params["min_count"]


class FastTextModel(UserDefinedRepresentation):
    """
    input type:pandas.series
    input like:(space separation)
    |input|
    |i love eat apple|
    |i love china|
    ----------------------------
    output type:pandas.dataframe
    output like:
    | 0 | 1 | 2 | 3 |
    |0.2|0.3|0.4|0.5|
    |0.3|0.2|0.6|0.1|
    """

    def __init__(self, cols="all", embedding_size=16, min_count=5, name=None, transform_check_max_number_error=1e-5,
                 skip_check_transform_type=True, prefix="fast"):
        super().__init__(cols=cols, name=name, transform_check_max_number_error=transform_check_max_number_error,
                         skip_check_transform_type=skip_check_transform_type, prefix=prefix)
        self.min_count = min_count
        self.embedding_size = embedding_size
        self.fasttext_model = {}

    def _fit(self, s: dataframe_type) -> dataframe_type:
        from gensim.models import FastText
        for col in self.cols:
            texts = s[col].values.tolist()
            texts = [line.split(" ") for line in texts]
            fasttext_model = FastText(sentences=texts, vector_size=self.embedding_size, min_count=self.min_count)
            self.fasttext_model[col] = fasttext_model
        return self

    def _transform(self, s: dataframe_type) -> dataframe_type:
        for col in self.cols:
            fasttext_model = self.fasttext_model[col]
            texts = s[col].values.tolist()
            texts = [line.split(" ") for line in texts]
            vectors = [np.mean(np.asarray([fasttext_model.wv[word]
                                           for word in line if word in fasttext_model.wv]), axis=0) + np.zeros(
                shape=(self.embedding_size,))
                       for line in texts]
            result = pandas.DataFrame(vectors)
            result.columns = ["{}_{}_{}".format(self.prefix, col, name) for name in result.columns]
            for icol in result.columns:
                s[icol] = result[icol]
                s[icol] = result[icol].values
        return s

    def _get_params(self) -> dict_type:
        return {"embedding_size": self.embedding_size, "fasttext_model": self.fasttext_model,
                "min_count": self.min_count}

    def _set_params(self, params: dict_type):
        self.embedding_size = params["embedding_size"]
        self.fasttext_model = params["fasttext_model"]
        self.min_count = params["min_count"]
