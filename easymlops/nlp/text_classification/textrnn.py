from .base import *
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

"""
代码参考:https://github.com/WoBruceWu/text-classification
"""


class TextRNNClassification(ClassificationBase):
    def __init__(self, y=None, **kwargs):
        super().__init__(y=y, **kwargs)
        self.text_rnn = None
        self.cuda = self.native_fit_params.get("cuda", "cpu")
        self.vocab_size = None

    def udf_fit(self, s, **kwargs):
        # 1.提取index,注意输入TextCNN的前置特征需要是VocabIndex
        x = s[self.col].tolist()
        x = torch.tensor(list(map(lambda line: [int(item) for item in line.strip().split(" ")], x)))
        y = torch.tensor(self.y.values)
        self.vocab_size = int(x.max()) + 1
        # 2.构建text_rnn模型
        self.text_rnn = TextRNN(self.num_class, self.vocab_size,
                                embedding_dim=self.native_init_params.get("embedding_dim", 128),
                                hidden_size=self.native_init_params.get("hidden_size", 64),
                                layer_num=self.native_init_params.get("layer_num", 1),
                                bidirectional=self.native_init_params.get("bidirectional", True),
                                use_pretrained_embedding=self.native_fit_params.get("use_pretrained_embedding", False),
                                pretrained_embedding=self.native_fit_params.get("pretrained_embedding", None),
                                fine_tune=self.native_fit_params.get("fine_tune", True))
        self.text_rnn.to(self.cuda)
        # 3.训练
        self.text_rnn.train()
        epoch = self.native_fit_params.get("epoch", 10)
        batch_size = self.native_fit_params.get("batch_size", 128)
        lr = self.native_fit_params.get("lr", 0.001)
        optimizer = torch.optim.Adam(self.text_rnn.parameters(), lr=lr)
        for _ in range(epoch):
            dataloader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True, num_workers=0,
                                    drop_last=True)
            for batch_x, batch_y in dataloader:
                batch_x, batch_y = batch_x.to(self.cuda), batch_y.to(self.cuda)
                optimizer.zero_grad()
                logits = self.text_rnn(batch_x)
                loss = F.cross_entropy(logits, batch_y)
                loss.backward()
                optimizer.step()
        return self

    def udf_transform(self, s, **kwargs):
        self.text_rnn.eval()
        x = s[self.col].tolist()
        x = torch.tensor(list(map(lambda line: [int(item) for item in line.strip().split(" ")], x)))
        x = x.to(self.cuda)
        result = pd.DataFrame(torch.softmax(self.text_rnn(x), dim=1).detach().numpy(),
                              columns=[self.id2label.get(i) for i in range(self.num_class)], index=s.index)
        return result

    def update_device(self, cuda):
        self.cuda = cuda
        self.text_rnn.to(cuda)

    def udf_set_params(self, params: dict):
        self.vocab_size = params["vocab_size"]
        self.native_init_params = params["native_init_params"]
        self.text_rnn = TextRNN(self.num_class, self.vocab_size,
                                embedding_dim=self.native_init_params.get("embedding_dim", 128),
                                hidden_size=self.native_init_params.get("hidden_size", 64),
                                layer_num=self.native_init_params.get("layer_num", 1),
                                bidirectional=self.native_init_params.get("bidirectional", True))

        self.text_rnn.load_state_dict(params["text_rnn_state_dict"])

    def udf_get_params(self) -> dict_type:
        return {"vocab_size": self.vocab_size, "text_rnn_state_dict": self.text_rnn.state_dict(),
                "native_init_params": self.native_init_params}


# 循环神经网络 (many-to-one)
class TextRNN(nn.Module):
    def __init__(self, num_class, vocab_size, embedding_dim=128, hidden_size=64, layer_num=1, bidirectional=True,
                 use_pretrained_embedding=False,
                 pretrained_embedding=None, fine_tune=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if use_pretrained_embedding:  # 如果使用预训练词向量，则提前加载，当不需要微调时设置freeze为True
            self.embedding = self.embedding.from_pretrained(pretrained_embedding, freeze=not fine_tune)

        self.lstm = nn.LSTM(embedding_dim,  # x的特征维度,即embedding_dim
                            self.hidden_size,  # 隐藏层单元数
                            self.layer_num,  # 层数
                            batch_first=True,  # 第一个维度设为 batch, 即:(batch_size, seq_length, embedding_dim)
                            bidirectional=self.bidirectional)  # 是否用双向
        self.fc = nn.Linear(self.hidden_size * 2, num_class) if self.bidirectional else nn.Linear(self.hidden_size,
                                                                                                  num_class)

    def forward(self, x):
        # 输入x的维度为(batch_size, max_len), max_len可以通过torchtext设置或自动获取为训练样本的最大长度
        x = self.embedding(x)  # 经过embedding,x的维度为(batch_size, time_step, input_size=embedding_dim)

        # 隐层初始化
        # h0维度为(num_layers*direction_num, batch_size, hidden_size)
        # c0维度为(num_layers*direction_num, batch_size, hidden_size)
        h0 = torch.zeros(self.layer_num * 2, x.size(0), self.hidden_size) if self.bidirectional else torch.zeros(
            self.layer_num, x.size(0), self.hidden_size)

        c0 = torch.zeros(self.layer_num * 2, x.size(0), self.hidden_size) if self.bidirectional else torch.zeros(
            self.layer_num, x.size(0), self.hidden_size)

        # LSTM前向传播，此时out维度为(batch_size, seq_length, hidden_size*direction_num)
        # hn,cn表示最后一个状态?维度与h0和c0一样
        out, (hn, cn) = self.lstm(x, (h0, c0))

        # 我们只需要最后一步的输出,即(batch_size, -1, output_size)
        out = self.fc(out[:, -1, :])
        return out
