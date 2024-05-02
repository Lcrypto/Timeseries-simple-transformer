# Imports
import torch, math
import numpy as np
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn.modules.transformer import TransformerEncoderLayer

# Positional Encoding - https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

# A forcasting model
class ForcastingModel(torch.nn.Module):
    def __init__(self, 
                 seq_len=200,
                 embed_size = 16,
                 nhead = 4,
                 dim_feedforward = 64,
                 dropout = 0.1,
                 device = "cuda"):
        super(ForcastingModel, self).__init__()
        self.device = device
        self.seq_len = seq_len
        self.embed_size = embed_size
        self.transformer_encoder = TransformerEncoderLayer(
            d_model = embed_size,
            nhead = nhead,
            dim_feedforward = dim_feedforward,
            dropout = dropout
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.position_encoder = PositionalEncoding(d_model=embed_size, 
                                                   dropout=dropout,
                                                   max_len=seq_len)
        self.input_embedding  = nn.Linear(1, embed_size)
        self.linear1 = nn.Linear(seq_len*embed_size, int(dim_feedforward))
        self.linear2 = nn.Linear(int(dim_feedforward), int(dim_feedforward/2))
        self.linear3 = nn.Linear(int(dim_feedforward/2), int(dim_feedforward/4))
        self.outlayer = nn.Linear(int(dim_feedforward/4), 1)
    def forward(self, x):
        src_mask = self._generate_square_subsequent_mask()
        src_mask.to(self.device)
        x = self.input_embedding(x)
        x = self.position_encoder(x)
        x = self.transformer_encoder(x).reshape((-1, self.seq_len*self.embed_size))
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear3(x)
        x = self.relu(x)
        return self.outlayer(x)
    def _generate_square_subsequent_mask(self):
        return torch.triu(
            torch.full((self.seq_len, self.seq_len), float('-inf'), dtype=torch.float32, device=self.device),
            diagonal=1,
        )

#https://github.com/ctxj/Time-Series-Transformer-Pytorch
#https://github.com/oliverguhr/transformer-time-series-prediction/blob/master/transformer-singlestep.py
#https://github.com/huggingface/transformers/blob/v4.40.1/src/transformers/models/time_series_transformer/modeling_time_series_transformer.py#L1179
