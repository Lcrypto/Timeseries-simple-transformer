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
                 dim_feedforward = 512,
                 dropout = 0.1,
                 device = "cuda"):
        super(ForcastingModel, self).__init__()
        self.device = device
        self.seq_len = seq_len
        self.transformer_encoder = TransformerEncoderLayer(
            d_model = 1,
            nhead = 1,
            dim_feedforward = dim_feedforward,
            dropout = dropout
        )
        self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(0.1)
        self.linear1 = nn.Linear(seq_len, int(dim_feedforward))
        self.linear2 = nn.Linear(int(dim_feedforward), int(dim_feedforward/2))
        self.linear3 = nn.Linear(int(dim_feedforward/2), int(dim_feedforward/4))
        self.outlayer = nn.Linear(int(dim_feedforward/4), 1)
    def forward(self, x):
        # src_mask = torch.rand((x.shape[0], self.seq_len, self.seq_len)).bool()
        # src_mask = src_mask.to(self.device)
        x = self.transformer_encoder(x).reshape((-1, self.seq_len))
        x = self.linear1(x)
        x = self.relu(x)
        # x = self.dropout(x)
        x = self.linear2(x)
        x = self.relu(x)
        # x = self.dropout(x)
        x = self.linear3(x)
        x = self.relu(x)
        return self.outlayer(x)
        return x
