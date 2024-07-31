from ..core import *
import pandas as pd
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pickle

"""
代码参考:https://github.com/WoBruceWu/text-classification
"""


class TSCNNTensorDatasetXY(TensorDataset):
    def __init__(self, *tensors: Tensor, time_period=7) -> None:
        super().__init__(*tensors)
        self.time_period = time_period

    def __getitem__(self, index):
        x_ = self.tensors[0]
        y_ = self.tensors[1]
        m, n = x_.shape
        new_x = torch.zeros((self.time_period, n))
        end_row = min(m, index + 1)
        start_row = max(0, index - self.time_period + 1)
        num_rows_to_copy = end_row - start_row
        new_x[self.time_period - num_rows_to_copy:self.time_period, :] = x_[start_row:end_row, :]
        return tuple([new_x, y_[index]])

    def __len__(self):
        return self.tensors[0].size(0)


class TSCNNTensorDatasetX(TensorDataset):
    def __init__(self, *tensors: Tensor, time_period=7) -> None:
        super().__init__(*tensors)
        self.time_period = time_period

    def __getitem__(self, index):
        x_ = self.tensors[0]
        m, n = x_.shape
        new_x = torch.zeros((self.time_period, n))
        end_row = min(m, index + 1)
        start_row = max(0, index - self.time_period + 1)
        num_rows_to_copy = end_row - start_row
        new_x[self.time_period - num_rows_to_copy:self.time_period, :] = x_[start_row:end_row, :]
        return new_x

    def __len__(self):
        return self.tensors[0].size(0)


class TSCNNRegression(M2MRollingPipeBase):
    def __init__(self, output_col="ts_cnn_regression_predict", input_cols: list = None, y=None,
                 native_init_params: dict = None,
                 native_fit_params: dict = None, time_period=15, **kwargs):
        super().__init__(output_cols=[output_col], input_cols=input_cols, min_cache_length=time_period, **kwargs)
        self.time_period = time_period
        self.ts_cnn = None
        self.y = y
        if native_init_params is None:
            self.native_init_params = {}
        else:
            self.native_init_params = native_init_params

        if native_fit_params is None:
            self.native_fit_params = {}
        else:
            self.native_fit_params = native_fit_params
        self.cuda = self.native_fit_params.get("cuda", "cpu")
        self.early_stopping = self.native_fit_params.get("early_stopping", {})

    def before_fit(self, s: dataframe_type, **kwargs) -> dataframe_type:
        s = super().before_fit(s, **kwargs)
        if self.input_cols is None or len(self.input_cols) == 0:
            self.input_cols = self.input_col_names
            # 更正缓存初始化
            self.history_data = {}
            for input_col_name in self.input_cols:
                self.history_data[input_col_name] = []
        self.native_init_params["input_dim"] = len(self.input_cols)
        return s

    def early_stopping_before(self, x, y):
        # 是否early_stopping，并获取early stopping参数
        if self.early_stopping is not None and len(self.early_stopping) > 0:
            self.eval_steps = self.early_stopping.get("eval_steps", 100)
            self.early_stopping_steps = self.early_stopping.get("early_stopping_steps", 500)
            self.eval_func = self.early_stopping.get("eval_func", F.mse_loss)
            self.eval_up = self.early_stopping.get("eval_up", False)  # eval_up=True,越大越优
            self.save_best_path = self.early_stopping.get("save_best_path", "./tmp/ts_cnn_best.pt")
            val_ratio = self.early_stopping.get("val_ratio", 0.1)
            train_size = int(len(x) * (1 - val_ratio))
            test_size = len(x) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                TSCNNTensorDatasetXY(x, y, time_period=self.time_period),
                [train_size, test_size])
            self.early_stopping_best_step = 0
            self.early_stopping_best_eval_score = None
        else:
            train_dataset = TSCNNTensorDatasetXY(x, y, time_period=self.time_period)
            val_dataset = None
        return train_dataset, val_dataset

    def early_stopping_run_step(self, step, val_dataset):
        stop_flag = False
        if self.early_stopping is not None and len(self.early_stopping) > 0 and step % self.eval_steps == 0:
            self.ts_cnn.eval()
            dataloader_early_stopping_eval = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=0,
                                                        drop_last=False)
            logits = []
            val_y = []
            for batch_eval_x, batch_eval_y in dataloader_early_stopping_eval:
                batch_eval_x, batch_eval_y = batch_eval_x.to(self.cuda), batch_eval_y.to(self.cuda)
                logits_ = self.ts_cnn(batch_eval_x)
                logits.append(logits_)
                val_y.append(batch_eval_y)
            logits = torch.cat(logits)
            val_y = torch.cat(val_y)
            eval_score = self.eval_func(logits, val_y)
            if self.early_stopping_best_eval_score is None:
                self.early_stopping_best_eval_score = eval_score
                self.early_stopping_best_step = step
                pickle.dump(self.ts_cnn.state_dict(), open(self.save_best_path, "wb"))
            else:
                # 判断当前是否是最优，如果不是最优，判断是否超过了early_stopping的限制
                if (self.eval_up and eval_score > self.early_stopping_best_eval_score) or (
                        not self.eval_up and eval_score < self.early_stopping_best_eval_score):
                    self.early_stopping_best_eval_score = eval_score
                    self.early_stopping_best_step = step
                    pickle.dump(self.ts_cnn.state_dict(), open(self.save_best_path, "wb"))
                else:
                    if step - self.early_stopping_best_step > self.early_stopping_steps:
                        self.ts_cnn.load_state_dict(pickle.load(open(self.save_best_path, "rb")))
                        stop_flag = True
            self.ts_cnn.train()
        return stop_flag

    def udf_fit(self, s, **kwargs):
        x = torch.tensor(s[self.input_cols].values)
        y = torch.tensor(self.y.values)

        # early stopping
        train_dataset, val_dataset = self.early_stopping_before(x, y)

        # 2.构建text_cnn模型
        self.ts_cnn = TsCNN(filter_num=self.native_init_params.get("filter_num", 128),
                            filter_sizes=self.native_init_params.get("filter_sizes", [3, 5, 10, 15]),
                            input_dim=self.native_init_params.get("input_dim", 128),
                            dropout=self.native_fit_params.get("dropout", 0.5),
                            use_max_pooling=self.native_init_params.get("use_max_pooling", False),
                            linear_lens=self.native_init_params.get("linear_lens", [1]),
                            max_len=self.time_period)
        self.ts_cnn.to(self.cuda)
        # 3.训练
        self.ts_cnn.train()
        epoch = self.native_fit_params.get("epoch", 10)
        batch_size = self.native_fit_params.get("batch_size", 128)
        lr = self.native_fit_params.get("lr", 0.001)
        optimizer = torch.optim.Adam(self.ts_cnn.parameters(), lr=lr)
        step = 0
        for _ in range(epoch):
            dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                                          drop_last=True)
            for batch_x, batch_y in dataloader_train:
                self.ts_cnn.train()
                batch_x, batch_y = batch_x.to(self.cuda), batch_y.to(self.cuda).float()
                optimizer.zero_grad()
                predict = self.ts_cnn(batch_x)
                loss = F.mse_loss(predict, batch_y)
                loss.backward()
                optimizer.step()
                step += 1
                if self.early_stopping_run_step(step, val_dataset):
                    break
        return self

    def apply_rolling_func(self, *inputs):
        self.ts_cnn.eval()
        x = torch.zeros((1, self.time_period, len(inputs)))
        x[0] = torch.asarray(inputs).T
        x = x.to(self.cuda)
        return self.ts_cnn(x).cpu().detach().numpy()[0][0]

    def udf_transform(self, s, **kwargs):
        self.ts_cnn.eval()
        x = torch.tensor(s[self.input_cols].values)
        dataloader_transform = DataLoader(TSCNNTensorDatasetX(x, time_period=self.time_period),
                                          batch_size=128, shuffle=False, num_workers=0, drop_last=False)
        preds = []
        for batch_x in dataloader_transform:
            batch_x = batch_x.to(self.cuda)
            preds.append(self.ts_cnn(batch_x).cpu().detach().numpy())
        result = pd.DataFrame(np.concatenate(preds), columns=[self.output_cols[0]], index=s.index)
        # 前time_period-1强制设置为0
        for i in range(self.time_period - 1):
            result.iloc[i][self.output_cols[0]] = 0.0
        return result

    def update_device(self, cuda):
        self.cuda = cuda
        self.ts_cnn.to(cuda)

    def udf_set_params(self, params: dict):
        self.native_init_params = params["native_init_params"]
        self.time_period = params["time_period"]
        self.ts_cnn = TsCNN(filter_num=self.native_init_params.get("filter_num", 128),
                            filter_sizes=self.native_init_params.get("filter_sizes", [3, 4, 5]),
                            input_dim=self.native_init_params.get("input_dim", 128),
                            use_max_pooling=self.native_init_params.get("use_max_pooling", False),
                            linear_lens=self.native_init_params.get("linear_lens", [1]),
                            max_len=self.time_period)
        self.ts_cnn.load_state_dict(params["ts_cnn_state_dict"])

    def udf_get_params(self) -> dict_type:
        return {"ts_cnn_state_dict": self.ts_cnn.state_dict(), "native_init_params": self.native_init_params,
                "native_fit_params": self.native_fit_params, "time_period": self.time_period}


class TsCNN(nn.Module):
    def __init__(self, filter_num=256, filter_sizes=[3, 5, 10, 15], max_len=15, use_max_pooling=False, input_dim=128,
                 dropout=0.5, linear_lens=[1]):
        super().__init__()
        self.input_dim = input_dim
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, filter_num, (fsz, self.input_dim)) for fsz in filter_sizes])
        self.dropout = nn.Dropout(dropout)
        self.use_max_pooling = use_max_pooling
        self.max_len = max_len
        self.linear_lens = linear_lens
        # 使用使用max_pooling
        if self.use_max_pooling:
            size_len = len(filter_sizes)
        else:
            size_len = 0
            for size in filter_sizes:
                size_len = size_len + max_len - size + 1
        # 全链接
        self.linear_layers = []
        self.bn_layers = []
        self.linear_lens.insert(0, size_len * filter_num)
        for idx in range(len(self.linear_lens) - 1):
            self.linear_layers.append(nn.Linear(self.linear_lens[idx], self.linear_lens[idx + 1]))
            self.bn_layers.append(nn.BatchNorm1d(self.linear_lens[idx + 1]))
        self.linear_layers = nn.ModuleList(self.linear_layers)
        self.bn_layers = nn.ModuleList(self.bn_layers)

    def forward(self, x):
        # 经过view函数x的维度变为(batch_size, input_chanel=1, w=max_len, h=input_dim)
        x = x.view(x.size(0), 1, x.size(1), self.input_dim)

        # 经过卷积运算,x中每个运算结果维度为(batch_size, out_chanel, w, h=1)
        x = [F.relu(conv(x)) for conv in self.convs]

        # 经过最大池化层,维度变为(batch_size, out_chanel, w=1, h=1)
        if self.use_max_pooling:
            x = [F.max_pool2d(input=x_item, kernel_size=(x_item.size(2), x_item.size(3))) for x_item in x]

        # 将不同卷积核运算结果维度（batch，out_chanel,w,h=1）展平为（batch, outchanel*w*h）
        x = [x_item.view(x_item.size(0), -1) for x_item in x]

        # 将不同卷积核提取的特征组合起来,维度变为(batch, sum:outchanel*w*h)
        x = torch.cat(x, 1)
        # dropout层
        x = self.dropout(x)

        # 全连接层
        for idx in range(len(self.linear_layers) - 1):
            linear_model = self.linear_layers[idx]
            bn_model = self.bn_layers[idx]
            x = linear_model(x)
            x = bn_model(x)
        x = self.linear_layers[-1](x)
        return x
