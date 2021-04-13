"""
这份代码实现了我们要训练的基于LSTM的RNN
"""

import torch
import torch.nn as nn


class LSTM_Net(nn.Module):
    def __init__(self, embedding, embedding_dim, hidden_dim, num_layers, dropout=0.5, fix_embedding=True):
        super(LSTM_Net, self).__init__()
        self.embedding = torch.nn.Embedding(embedding.size(0),embedding.size(1)) # 两个参数：word的数量，embedding所得vector的维度数量
        self.embedding.weight = torch.nn.Parameter(embedding)
        self.embedding.weight.requires_grad = False if fix_embedding else True # 如果fix_embedding为False，则训练过程中embedding也会被跟着训练
        self.embedding_dim = embedding.size(1) # embedding所得vector的维度数量
        self.hidden_dim = hidden_dim # LSTM中隐藏层的维度数
        self.num_layers = num_layers # LSTM的层数
        self.dropout = dropout
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.classifier = nn.Sequential( nn.Dropout(dropout), nn.Linear(hidden_dim, 1), nn.Sigmoid())
    
    def forward(self, inputs):
        inputs = self.embedding(inputs) # 进行word embedding
        x, _ = self.lstm(inputs, None) # x的dimension为(batch, seq_len, hidden_size)
        x = x[:, -1, :]  # 取LSTM最后一层的hidden state
        x = self.classifier(x)
        return x
