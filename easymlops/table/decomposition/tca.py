from easymlops.table.core import *
import scipy.io
import scipy.linalg
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel


def kernel(ker, X1, X2, gamma):
    K = None
    if not ker or ker == 'primal':
        K = X1
    elif ker == 'linear':
        if X2 is not None:
            K = linear_kernel(np.asarray(X1).T, np.asarray(X2).T)
        else:
            K = linear_kernel(np.asarray(X1).T)
    elif ker == 'rbf':
        if X2 is not None:
            K = rbf_kernel(np.asarray(X1).T, np.asarray(X2).T, gamma)
        else:
            K = rbf_kernel(np.asarray(X1).T, None, gamma)
    return K


class TCADecomposition(TablePipeObjectBase):

    def __init__(self, dim=30, decomposition_class_label="random_5", train_type="mvo", agg_type="average",
                 kernel_type='primal', lamb=1, gamma=1,
                 native_init_params=None, native_fit_params=None, **kwargs):
        """
        :param n_components: 保留的主成分数量
        :param decomposition_class_label:降维类别标签,random_5表示随机切分为5组
        :param train_type:mvo多对一，ovo一对一
        :param agg_type:聚合方式
        :param kernel_type:核函数
        :param lamb:
        :param gamma
        :param native_init_params: 初始化参数
        :param native_fit_params: fit除X以外的参数
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.dim = dim
        self.decomposition_class_label = decomposition_class_label
        self.train_type = train_type
        self.agg_type = agg_type
        self.kernel_type = kernel_type
        self.lamb = lamb
        self.gamma = gamma
        self.native_init_params = native_init_params
        self.native_fit_params = native_fit_params
        if self.native_init_params is None:
            self.native_init_params = dict()
        if self.native_fit_params is None:
            self.native_fit_params = dict()
        self.col_types = dict()
        self.models = []

    def udf_fit(self, s: dataframe_type, **kwargs):
        # 记录数据类型
        for col in s.columns.tolist():
            self.col_types[col] = self.get_col_type(s[col])
        s = s.fillna(0)
        # 切组
        if type(self.decomposition_class_label) == str:
            if "random" in self.decomposition_class_label:
                g_num = int(self.decomposition_class_label.replace("random_", ""))
                s["__g_idx"] = [np.random.randint(g_num) for _ in range(len(s))]
        else:
            s["__g_idx"] = self.decomposition_class_label

        # 训练
        if self.train_type == "mvo":
            for g_idx in s["__g_idx"].unique():
                Xs = s[s["__g_idx"] != g_idx]
                Xt = s[s["__g_idx"] == g_idx]
                del Xs["__g_idx"]
                del Xt["__g_idx"]
                Xs = Xs.values
                Xt = Xt.values
                X = np.hstack((Xs.T, Xt.T))
                X /= np.linalg.norm(X, axis=0)
                m, n = X.shape
                ns, nt = len(Xs), len(Xt)
                e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
                M = e * e.T
                M = M / np.linalg.norm(M, 'fro')
                H = np.eye(n) - 1 / n * np.ones((n, n))
                K = kernel(self.kernel_type, X, None, gamma=self.gamma)
                n_eye = m if self.kernel_type == 'primal' else n
                a, b = K @ M @ K.T + self.lamb * np.eye(n_eye), K @ H @ K.T
                w, V = scipy.linalg.eig(a, b)
                ind = np.argsort(w)
                A = V[:, ind[:self.dim]]
                self.models.append(A)

        else:  # ovo
            for g_idx1 in s["__g_idx"].unique():
                for g_idx2 in s["__g_idx"].unique():
                    if g_idx1 != g_idx2:
                        Xs = s[s["__g_idx"] == g_idx1]
                        Xt = s[s["__g_idx"] == g_idx2]
                        del Xs["__g_idx"]
                        del Xt["__g_idx"]
                        Xs = Xs.values
                        Xt = Xt.values
                        X = np.hstack((Xs.T, Xt.T))
                        X /= np.linalg.norm(X, axis=0)
                        m, n = X.shape
                        ns, nt = len(Xs), len(Xt)
                        e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
                        M = e * e.T
                        M = M / np.linalg.norm(M, 'fro')
                        H = np.eye(n) - 1 / n * np.ones((n, n))
                        K = kernel(self.kernel_type, X, None, gamma=self.gamma)
                        n_eye = m if self.kernel_type == 'primal' else n
                        a, b = K @ M @ K.T + self.lamb * np.eye(n_eye), K @ H @ K.T
                        w, V = scipy.linalg.eig(a, b)
                        ind = np.argsort(w)
                        A = V[:, ind[:self.dim]]
                        self.models.append(A)
        del s["__g_idx"]
        return self

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        # 转换数据类型
        for col, col_type in self.col_types.items():
            s[col] = s[col].astype(col_type)
        s = s.fillna(0)
        s_ = np.clip(s.fillna(0).values, np.finfo(np.float64).min, np.finfo(np.float64).max)
        # 预测
        rsts = []
        X = s_.T
        X /= np.linalg.norm(X, axis=0)
        K = kernel(self.kernel_type, X, None, gamma=self.gamma)
        for A in self.models:
            Z = A.T @ K
            Z /= np.linalg.norm(Z, axis=0)
            rsts.append(Z.T)
        # 聚合方式
        if self.agg_type == "concat":
            result = np.concatenate(rsts, axis=1)
        else:
            result = np.mean(rsts, axis=0)
        result = pd.DataFrame(result, index=s.index)
        return result

    def transform_single(self, s: dict_type, **kwargs):
        input_dataframe = pd.DataFrame([s])
        return self.transform(input_dataframe, **kwargs).to_dict("records")[0]

    def udf_get_params(self) -> dict_type:
        params = {"dim": self.dim,
                  "train_type": self.train_type,
                  "agg_type": self.agg_type,
                  "kernel_type": self.kernel_type,
                  "lamb": self.lamb,
                  "gamma": self.gamma,
                  "native_init_params": self.native_init_params,
                  "native_fit_params": self.native_fit_params,
                  "col_types": self.col_types,
                  "models": self.models}
        return params

    def udf_set_params(self, params: dict):
        self.dim = params["dim"]
        self.train_type = params["train_type"]
        self.agg_type = params["agg_type"]
        self.kernel_type = params["kernel_type"]
        self.lamb = params["lamb"]
        self.gamma = params["gamma"]
        self.native_init_params = params["native_init_params"]
        self.native_fit_params = params["native_fit_params"]
        self.col_types = params["col_types"]
        self.models = params["models"]
