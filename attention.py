import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(torch.nn.Module):
    def __init__(self, embed_dim=1024):
        super(Attention, self).__init__()
        self.embed_dim = embed_dim
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
    def forward(self, x, mask=True):
        keys = self.key(x)
        queries = self.query(x)
        values = self.value(x)
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.embed_dim, dtype=torch.float32))
        if mask:
            maskmat = torch.tril(torch.ones((self.embed_dim, self.embed_dim)))
            scores = scores.masked_fill(maskmat == 0, -1e9)
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, values)
        return output


from torch.utils.data import DataLoader
attention = Attention(10)
data = torch.tensor([[1,2,3,4,5,4,3,2,1,0],[1,2,3,4,5,4,3,2,1,0]], dtype=torch.float32)
dataloader = DataLoader(data, batch_size=2)
for batch in dataloader:
    attention(batch)