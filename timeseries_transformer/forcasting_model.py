# Imports
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.transformer import TransformerEncoderLayer


# A forcasting model
class ForcastingModel(torch.nn.Module):
    def __init__(self, 
                 seq_len=200, 
                 nhead=8,
                 dim_feedforward = 2048,
                 dropout = 0.1,
                 device = "cuda"):
        super(ForcastingModel, self).__init__()
        self.device = device
        self.seq_len = seq_len
        self.transformer_encoder = TransformerEncoderLayer(
            d_model = seq_len,
            nhead = nhead,
            dim_feedforward = dim_feedforward,
            dropout = dropout
        )
        # self.relu = nn.ReLU()
        # self.linear1 = nn.Linear(seq_len, int(ffdim))
        # self.linear2 = nn.Linear(int(ffdim), int(ffdim/2))
        # self.linear3 = nn.Linear(int(ffdim/2), int(ffdim/4))
        # self.outlayer = nn.Linear(int(ffdim/4), 1)
    def forward(self, x):
        x.reshape(x.shape[0], 1, x.shape[1])
        x = self.transformer_encoder(x)
        # maskmat = torch.tril(torch.ones((x.shape[1], x.shape[1]))).to(self.device)
        # x = self.attention(x)
        # x = self.linear1(x)
        # x = self.relu(x)
        # x = self.linear2(x)
        # x = self.relu(x)
        # x = self.linear3(x)
        # x = self.relu(x)
        # return self.outlayer(x)
        return x
