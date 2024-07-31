from .base import *
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

"""
代码参考:https://github.com/WoBruceWu/text-classification
"""


class HANClassification(ClassificationBase):
    def __init__(self, y=None, **kwargs):
        super().__init__(y=y, **kwargs)
        self.word_attn_model = None
        self.sent_attn_model = None
        self.cuda = self.native_fit_params.get("cuda", "cpu")
        self.vocab_size = None

    def udf_fit(self, s, **kwargs):
        # 1.提取index,注意输入HAN的前置特征需要是VocabIndexWithSentSplit
        x = s[self.col].tolist()
        x = torch.tensor(list(
            map(lambda line: [[int(item) for item in sent.strip().split(" ")] for sent in line.strip().split("|")], x)))
        y = torch.tensor(self.y.values)
        self.vocab_size = int(x.max()) + 1
        # 2.构建han模型
        self.word_attn_model = AttentionWordGRU(self.vocab_size,
                                                embedding_dim=self.native_init_params.get("embedding_dim", 128),
                                                word_hidden_size=self.native_init_params.get("word_hidden_size", 128),
                                                bidirectional=self.native_init_params.get("bidirectional", True),
                                                use_pretrained_embedding=self.native_fit_params.get(
                                                    "use_pretrained_embedding", False),
                                                pretrained_embedding=self.native_fit_params.get("pretrained_embedding",
                                                                                                None),
                                                fine_tune=self.native_fit_params.get("fine_tune", True))

        self.sent_attn_model = AttentionSentGRU(self.num_class,
                                                sent_gru_hidden_size=self.native_init_params.get("sent_gru_hidden_size",
                                                                                                 128),
                                                bidirectional=self.native_init_params.get("bidirectional", True),
                                                word_gru_hidden_size=self.native_init_params.get("word_gru_hidden_size",
                                                                                                 128))
        self.word_attn_model.to(self.cuda)
        self.sent_attn_model.to(self.cuda)
        # 3.训练
        self.word_attn_model.train()
        self.sent_attn_model.train()
        epoch = self.native_fit_params.get("epoch", 10)
        batch_size = self.native_fit_params.get("batch_size", 128)
        lr = self.native_fit_params.get("lr", 0.001)
        word_optimizer = torch.optim.Adam(self.word_attn_model.parameters(), lr=lr)
        sent_optimizer = torch.optim.Adam(self.sent_attn_model.parameters(), lr=lr)
        for _ in range(epoch):
            dataloader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True, num_workers=0,
                                    drop_last=True)
            for batch_x, batch_y in dataloader:
                batch_x, batch_y = batch_x.to(self.cuda), batch_y.to(self.cuda)
                word_optimizer.zero_grad()
                sent_optimizer.zero_grad()

                # doc_texts的维度为(batch_size, sents_num, words_num)
                word_attn_vectors = None
                for doc_text in batch_x:
                    # word_attn_vector的维度为(sent_num, hidden_size)
                    word_attn_vector = self.word_attn_model(doc_text)
                    # 将word_attn_vector的维度变为(1, sent_num, hidden_size)
                    word_attn_vector = word_attn_vector.unsqueeze(0)
                    if word_attn_vectors is None:
                        word_attn_vectors = word_attn_vector
                    else:
                        # word_attn_vectors的维度为(batch_size, sent_num, hidden_size)
                        word_attn_vectors = torch.cat((word_attn_vectors, word_attn_vector), 0)
                logits = self.sent_attn_model(word_attn_vectors)
                loss = F.cross_entropy(logits, batch_y)
                loss.backward()
                word_optimizer.step()
                sent_optimizer.step()
        return self

    def udf_transform(self, s, **kwargs):
        self.sent_attn_model.eval()
        x = s[self.col].tolist()
        x = torch.tensor(list(
            map(lambda line: [[int(item) for item in sent.strip().split(" ")] for sent in line.strip().split("|")], x)))
        x = x.to(self.cuda)
        word_attn_vectors = None
        for doc_text in x:
            # word_attn_vector的维度为(sent_num, hidden_size)
            word_attn_vector = self.word_attn_model(doc_text)
            # 将word_attn_vector的维度变为(1, sent_num, hidden_size)
            word_attn_vector = word_attn_vector.unsqueeze(0)
            if word_attn_vectors is None:
                word_attn_vectors = word_attn_vector
            else:
                # word_attn_vectors的维度为(batch_size, sent_num, hidden_size)
                word_attn_vectors = torch.cat((word_attn_vectors, word_attn_vector), 0)
        logits = self.sent_attn_model(word_attn_vectors)
        result = pd.DataFrame(torch.softmax(logits, dim=1).detach().numpy(),
                              columns=[self.id2label.get(i) for i in range(self.num_class)], index=s.index)
        return result

    def update_device(self, cuda):
        self.cuda = cuda
        self.sent_attn_model.to(cuda)

    def udf_set_params(self, params: dict):
        self.vocab_size = params["vocab_size"]
        self.native_init_params = params["native_init_params"]
        self.word_attn_model = AttentionWordGRU(self.vocab_size,
                                                embedding_dim=self.native_init_params.get("embedding_dim", 128),
                                                word_hidden_size=self.native_init_params.get("word_hidden_size", 128),
                                                bidirectional=self.native_init_params.get("bidirectional", True))
        self.word_attn_model.load_state_dict(params["word_attn_state_dict"])

        self.sent_attn_model = AttentionSentGRU(self.num_class,
                                                sent_gru_hidden_size=self.native_init_params.get("sent_gru_hidden_size",
                                                                                                 128),
                                                bidirectional=self.native_init_params.get("bidirectional", True),
                                                word_gru_hidden_size=self.native_init_params.get("word_gru_hidden_size",
                                                                                                 128))
        self.sent_attn_model.load_state_dict(params["sent_attn_state_dict"])

    def udf_get_params(self) -> dict_type:
        return {"vocab_size": self.vocab_size, "word_attn_state_dict": self.word_attn_model.state_dict(),
                "sent_attn_state_dict": self.sent_attn_model.state_dict(),
                "native_init_params": self.native_init_params}


def batch_matmul(seq, weight, activation=''):
    # seq维度为(batch_size, seq_length, hidden_size)
    # weight此时为query_vec,维度为(hidden_size, 1)
    s = None
    for i in range(seq.size(0)):
        _s = torch.mm(seq[i], weight)
        if activation == 'tanh':
            s = torch.tanh(_s)

        # 将_s的维度从(seq_length, 1)变为(1, seq_length, 1)
        _s = _s.unsqueeze(0)

        if s is None:
            s = _s
        else:
            s = torch.cat((s, _s), 0)
    # 经运算，s维度为(batch_size, seq_length, 1),经squeeze变为(batch_size, seq_length)
    return s.squeeze()


def batch_matmul_bias(seq, weight, bias, activation=''):
    # seq维度为(batch_size, seq_length, hidden_size)
    # weight维度为(hidden_size, hidden_size)
    # bias维度为(hidden_size, 1)
    s = None
    bias_dim = bias.size()
    for i in range(seq.size(0)):
        _s = torch.mm(seq[i], weight)  # _s维度为(seq_length, hidden_size)
        _s_bias = _s + bias.expand(bias_dim[0], _s.size()[0]).t()  # _s_bias维度为(seq_length, hidden_size)
        if activation == 'tanh':
            _s_bias = torch.tanh(_s_bias)
        # 将_s_bias的维度从(seq_length, hidden_size)变为(1, seq_length, hidden_size)
        _s_bias = _s_bias.unsqueeze(0)
        if s is None:
            s = _s_bias
        else:
            s = torch.cat((s, _s_bias), 0)
    # s的维度为(batch_size, seq_length, hidden_size)
    return s


def attention_mul(rnn_outputs, att_weights):
    # rnn_outputs的维度为(batch_size, seq_length, hidden_size)
    # att_weights的维度为(batch_size, seq_length)
    attn_vectors = None
    for i in range(rnn_outputs.size(0)):
        h_i = rnn_outputs[i]  # h_i维度为(seq_length, hidden_size)
        a_i = att_weights[i].unsqueeze(1).expand_as(h_i)  # a_i维度为(seq_length, hidden_size)
        h_i = a_i * h_i  # 运算后的h_i维度为(seq_length, hidden_size)
        h_i = h_i.unsqueeze(0)  # h_i的维度为(1, seq_length, hidden_size)
        if attn_vectors is None:
            attn_vectors = h_i
        else:
            attn_vectors = torch.cat((attn_vectors, h_i), 0)
    # attn_vetors维度为(batch_size, seq_length, hidden_size)
    # 经过sum, attn_vectors维度为(batch_size, hidden_size)
    return torch.sum(attn_vectors, 1)


# 词语级GRU
class AttentionWordGRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, word_hidden_size, bidirectional, use_pretrained_embedding=False,
                 pretrained_embedding=None, fine_tune=True):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.word_gru_hidden_size = word_hidden_size
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        if use_pretrained_embedding:  # 如果使用预训练词向量，则提前加载，当不需要微调时设置freeze为True
            self.embedding = self.embedding.from_pretrained(pretrained_embedding, freeze=not fine_tune)

        self.direction_num = 2 if self.bidirectional else 1

        # 词语对应的GRU层
        self.word_gru = nn.GRU(self.embedding_dim, self.word_gru_hidden_size,
                               bidirectional=self.bidirectional, batch_first=True)

        # attention参数中矩阵参数，维度为(hidden_size*direction_num, hidden_size*direction_num)
        self.weights_w_word = nn.Parameter(torch.Tensor(self.word_gru_hidden_size * self.direction_num,
                                                        self.word_gru_hidden_size * self.direction_num))

        # attention参数中矩阵对应的偏差项，维度为(hidden_size*direction_num, 1)
        self.bias_word = nn.Parameter(torch.Tensor(self.word_gru_hidden_size * self.direction_num, 1))

        # 对每个词的表示做attention的向量
        self.query_vec_word = nn.Parameter(torch.Tensor(self.word_gru_hidden_size * self.direction_num, 1))

        # 初始化attention矩阵和向量
        self.weights_w_word.data.uniform_(-0.1, 0.1)
        self.query_vec_word.data.uniform_(-0.1, 0.1)

    def forward(self, x):
        # 输入x的维度为(sent_num, max_len), max_len可以通过torchtext设置或自动获取为训练样本的最大长度
        x = self.embedding(x)  # 经过embedding,x的维度为(sent_num, time_step, input_size=embedding_dim)

        # GRU隐层初始化,维度为(num_layers*direction_num, sent_num, hidden_size)，本论文结构中
        h0 = torch.zeros(self.direction_num, x.size(0), self.word_gru_hidden_size)

        # GRU层运算,out的维度为(sent_num, seq_length, hidden_size)
        out, hn = self.word_gru(x, h0)

        # word_squish维度为(sent_num, seq_length, hidden_size)
        word_squish = batch_matmul_bias(out, self.weights_w_word, self.bias_word, 'tanh')

        # word_attn维度为(sent_num, seq_length)
        word_attn = batch_matmul(word_squish, self.query_vec_word, '')

        # word_attn_norm维度为(sent_num, seq_length)
        word_attn_norm = F.softmax(word_attn)

        word_attn_vectors = attention_mul(out, word_attn_norm)
        # word_attn_vectors:(sent_num, hidden_size)
        # hn: (num_layers*direction_num, sent_num, hidden_size)
        # word_attn_norm:(sent_num, seq_length)
        return word_attn_vectors


# ## 句子级attention

class AttentionSentGRU(nn.Module):

    def __init__(self, num_class, sent_gru_hidden_size=128, bidirectional=True, word_gru_hidden_size=128):
        super().__init__()

        direction_num = 2 if bidirectional else 1
        self.sent_gru_hidden_size = sent_gru_hidden_size
        self.direction_num = direction_num

        # 句子对应的GRU层
        self.sent_gru = nn.GRU(direction_num * word_gru_hidden_size, sent_gru_hidden_size,
                               bidirectional=bidirectional, batch_first=True)

        # attention参数中矩阵参数，维度为(hidden_size*direction_num, hidden_size*direction_num)
        self.weight_w_sent = nn.Parameter(
            torch.Tensor(direction_num * sent_gru_hidden_size, direction_num * sent_gru_hidden_size))

        # attention参数中矩阵对应的偏差项，维度为(hidden_size*direction_num, 1)
        self.bias_sent = nn.Parameter(torch.Tensor(direction_num * sent_gru_hidden_size, 1))

        # 对每个句子的表示做attention的向量
        self.query_vec_sent = nn.Parameter(torch.Tensor(direction_num * sent_gru_hidden_size, 1))

        self.linear = nn.Linear(direction_num * sent_gru_hidden_size, num_class)

        # 初始化attention矩阵和向量
        self.weight_w_sent.data.uniform_(-0.1, 0.1)
        self.query_vec_sent.data.uniform_(-0.1, 0.1)

    # word_attn_vectors的维度为(batch_size, sent_num, word_hidden_size),sent_num为一篇文章中句子的数量,batch_size为文章的数量
    def forward(self, word_attn_vectors):
        # h0维度为((num_layers*direction_num, batch_size, sent_hidden_size))
        h0 = torch.zeros(self.direction_num, word_attn_vectors.size(0), self.sent_gru_hidden_size)

        # GRU层运算,out的维度为(batch_size, sent_length, sent_hidden_size)
        out, hn = self.sent_gru(word_attn_vectors, h0)

        # sent_squish的维度为(batch_size, sent_length, sent_hidden_size)
        sent_squish = batch_matmul_bias(out, self.weight_w_sent, self.bias_sent, activation='tanh')

        # sent_attn的维度为(batch_size, sent_length)
        sent_attn = batch_matmul(sent_squish, self.query_vec_sent)

        # sent_attn的维度为(batch_size, sent_length)
        sent_attn_norm = F.softmax(sent_attn)

        # sent_attn_vectors的维度为(batch_size, sent_hidden_size)
        sent_attn_vectors = attention_mul(out, sent_attn_norm)

        # 全连接层，返回的logits维度为(batch_size, label_num)
        logits = self.linear(sent_attn_vectors)

        return logits
