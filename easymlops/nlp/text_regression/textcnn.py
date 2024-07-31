from .base import *
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

"""
代码参考:https://github.com/WoBruceWu/text-classification
"""


class TextCNNRegression(RegressionBase):
    def __init__(self, y=None, **kwargs):
        super().__init__(y=y, **kwargs)
        self.text_cnn = None
        self.cuda = self.native_fit_params.get("cuda", "cpu")
        self.early_stopping = self.native_fit_params.get("early_stopping")
        self.vocab_size = None

    def early_stopping_before(self, x, y):
        # 是否early_stopping，并获取early stopping参数
        if self.early_stopping is not None:
            self.eval_steps = self.early_stopping.get("eval_steps", 100)
            self.early_stopping_steps = self.early_stopping.get("early_stopping_steps", 500)
            self.eval_func = self.early_stopping.get("eval_func", F.cross_entropy)
            self.eval_up = self.early_stopping.get("eval_up", False)  # eval_up=True,越大越优
            self.save_best_path = self.early_stopping.get("save_best_path", "./tmp/text_cnn_best.pt")
            val_ratio = self.early_stopping.get("val_ratio", 0.1)
            train_size = int(len(x) * (1 - val_ratio))
            test_size = len(x) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(TensorDataset(x, y), [train_size, test_size])
            val_x, val_y = val_dataset.x.to(self.cuda), val_dataset.y.to(self.cuda)
            self.early_stopping_best_step = 0
            self.early_stopping_best_eval_score = None
        else:
            train_dataset = TensorDataset(x, y)
            val_x, val_y = None, None
        return train_dataset, val_x, val_y

    def early_stopping_run_step(self, step, val_x, val_y):
        stop_flag = False
        if self.early_stopping is not None and step % self.eval_steps == 0:
            self.text_cnn.eval()
            logits = self.text_cnn(val_x)
            eval_score = self.eval_func(logits, val_y)
            if self.early_stopping_best_eval_score is None:
                self.early_stopping_best_eval_score = eval_score
                self.early_stopping_best_step = step
                pickle.dump(self.text_cnn.state_dict(), open(self.save_best_path, "wb"))
            else:
                # 判断当前是否是最优，如果不是最优，判断是否超过了early_stopping的限制
                if (self.eval_up and eval_score > self.early_stopping_best_eval_score) or (
                        not self.eval_up and eval_score < self.early_stopping_best_eval_score):
                    self.early_stopping_best_eval_score = eval_score
                    self.early_stopping_best_step = step
                    pickle.dump(self.text_cnn.state_dict(), open(self.save_best_path, "wb"))
                else:
                    if step - self.early_stopping_best_step > self.early_stopping_steps:
                        self.text_cnn.load_state_dict(pickle.load(open(self.save_best_path, "rb")))
                        stop_flag = True
            self.text_cnn.train()
        return stop_flag

    def udf_fit(self, s, **kwargs):
        # 1.提取index,注意输入TextCNN的前置特征需要是VocabIndex
        x = s[self.col].tolist()
        x = torch.tensor(list(map(lambda line: [int(item) for item in line.strip().split(" ")], x)))
        y = torch.tensor(self.y.values)
        self.vocab_size = int(x.max()) + 1

        # early stopping
        train_dataset, val_x, val_y = self.early_stopping_before(x, y)

        # 2.构建text_cnn模型
        self.text_cnn = TextCNN(vocab_size=self.vocab_size,
                                filter_num=self.native_init_params.get("filter_num", 128),
                                filter_sizes=self.native_init_params.get("filter_sizes", [3, 4, 5]),
                                embedding_dim=self.native_init_params.get("embedding_dim", 128),
                                use_pretrained_embedding=self.native_fit_params.get("use_pretrained_embedding", False),
                                pretrained_embedding=self.native_fit_params.get("pretrained_embedding", None),
                                fine_tune=self.native_fit_params.get("fine_tune", True),
                                dropout=self.native_fit_params.get("dropout", 0.5))
        self.text_cnn.to(self.cuda)
        # 3.训练
        self.text_cnn.train()
        epoch = self.native_fit_params.get("epoch", 10)
        batch_size = self.native_fit_params.get("batch_size", 128)
        lr = self.native_fit_params.get("lr", 0.001)
        optimizer = torch.optim.Adam(self.text_cnn.parameters(), lr=lr)
        step = 0
        for _ in range(epoch):
            dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                                    drop_last=True)
            for batch_x, batch_y in dataloader:
                self.text_cnn.train()
                batch_x, batch_y = batch_x.to(self.cuda), batch_y.to(self.cuda).float()
                optimizer.zero_grad()
                predict = self.text_cnn(batch_x)
                loss = F.mse_loss(predict, batch_y)
                loss.backward()
                optimizer.step()
                step += 1
                if self.early_stopping_run_step(step, val_x, val_y):
                    break
        return self

    def udf_transform(self, s, **kwargs):
        self.text_cnn.eval()
        x = s[self.col].tolist()
        x = torch.tensor(list(map(lambda line: [int(item) for item in line.strip().split(" ")], x)))
        x = x.to(self.cuda)
        result = pd.DataFrame(self.text_cnn(x).detach().numpy(), columns=[self.pred_name], index=s.index)
        return result

    def update_device(self, cuda):
        self.cuda = cuda
        self.text_cnn.to(cuda)

    def udf_set_params(self, params: dict):
        self.vocab_size = params["vocab_size"]
        self.native_init_params = params["native_init_params"]
        self.text_cnn = TextCNN(vocab_size=self.vocab_size,
                                filter_num=self.native_init_params.get("filter_num", 128),
                                filter_sizes=self.native_init_params.get("filter_sizes", [3, 4, 5]),
                                embedding_dim=self.native_init_params.get("embedding_dim", 128))
        self.text_cnn.load_state_dict(params["text_cnn_state_dict"])

    def udf_get_params(self) -> dict_type:
        return {"vocab_size": self.vocab_size, "text_cnn_state_dict": self.text_cnn.state_dict(),
                "native_init_params": self.native_init_params}


class TextCNN(nn.Module):
    def __init__(self, vocab_size, filter_num=256, filter_sizes=[3, 4, 5], embedding_dim=128,
                 use_pretrained_embedding=False,
                 pretrained_embedding=None, fine_tune=True, dropout=0.5):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if use_pretrained_embedding:  # 如果使用预训练词向量，则提前加载，当不需要微调时设置freeze为True
            self.embedding = self.embedding.from_pretrained(pretrained_embedding, freeze=not fine_tune)

        self.convs = nn.ModuleList(
            [nn.Conv2d(1, filter_num, (fsz, embedding_dim)) for fsz in filter_sizes])
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(len(filter_sizes) * filter_num, 1)

    def forward(self, x):
        # 输入x的维度为(batch_size, max_len), max_len可以通过torchtext设置或自动获取为训练样本的最大=长度
        x = self.embedding(x)  # 经过embedding,x的维度为(batch_size, max_len, embedding_dim)

        # 经过view函数x的维度变为(batch_size, input_chanel=1, w=max_len, h=embedding_dim)
        x = x.view(x.size(0), 1, x.size(1), self.embedding_dim)

        # 经过卷积运算,x中每个运算结果维度为(batch_size, out_chanel, w, h=1)
        x = [F.relu(conv(x)) for conv in self.convs]

        # 经过最大池化层,维度变为(batch_size, out_chanel, w=1, h=1)
        x = [F.max_pool2d(input=x_item, kernel_size=(x_item.size(2), x_item.size(3))) for x_item in x]

        # 将不同卷积核运算结果维度（batch，out_chanel,w,h=1）展平为（batch, outchanel*w*h）
        x = [x_item.view(x_item.size(0), -1) for x_item in x]

        # 将不同卷积核提取的特征组合起来,维度变为(batch, sum:outchanel*w*h)
        x = torch.cat(x, 1)

        # dropout层
        x = self.dropout(x)

        # 全连接层
        logits = self.linear(x)
        return logits
