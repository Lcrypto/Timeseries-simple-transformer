# Imports
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


# A basic attention mechanism
class Attention(torch.nn.Module):
    def __init__(self, seq_len=200):
        super(Attention, self).__init__()
        self.queries = nn.Linear(seq_len, seq_len)
        self.keys = nn.Linear(seq_len, seq_len)
        self.values = nn.Linear(seq_len, seq_len)
    def forward(self, x, mask=True):
        q = self.queries(x).reshape(x.shape[0], x.shape[1], 1)
        k = self.keys(x).reshape(x.shape[0], x.shape[1], 1)
        v = self.values(x).reshape(x.shape[0], x.shape[1], 1)
        scores = torch.bmm(q, k.transpose(-2, -1))
        if mask:
            maskmat = torch.tril(torch.ones((x.shape[1], x.shape[1])))
            scores = scores.masked_fill(maskmat == 0, -1e9)
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.bmm(attention_weights, v)
        return output.reshape(output.shape[0], output.shape[1])

# A forcasting model
class ForcastingModel(torch.nn.Module):
    def __init__(self, seq_len=200, ffdim=64):
        super(ForcastingModel, self).__init__()
        self.relu = nn.ReLU()
        self.attention = Attention(seq_len)
        self.linear1 = nn.Linear(seq_len, int(ffdim))
        self.linear2 = nn.Linear(int(ffdim), int(ffdim/2))
        self.linear3 = nn.Linear(int(ffdim/2), int(ffdim/4))
        self.outlayer = nn.Linear(int(ffdim/4), 1)
    def forward(self, x):
        x = self.attention(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        return self.outlayer(x)


# Get a noisy sin wave
DATA_SIZE = 1000
x = np.sin(np.linspace(0, 10, DATA_SIZE))
x = x + np.random.normal(0, 0.05, DATA_SIZE)
import matplotlib.pyplot as plt
plt.plot(range(x.shape[0]), x)
plt.show()


# Create a dataset
seq_len = 200
X = np.array([x[ii:ii+seq_len] for ii in range(0, x.shape[0]-seq_len)])
Y = np.array([x[ii+seq_len] for ii in range(0, x.shape[0]-seq_len)])


# Training Loop
EPOCHS = 20
BATCH_SIZE = 8
LEARNING_RATE = 4.12e-5
model = ForcastingModel(seq_len)
model.train()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
dataset = TensorDataset(torch.Tensor(X), torch.Tensor(Y))
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
for epoch in range(EPOCHS):
    for xx, yy in dataloader:
        optimizer.zero_grad()
        out = model(xx)
        loss = criterion(out, yy)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{EPOCHS}: Loss={loss}")


# Prediction Loop
FORCAST = 1000
model.eval()
for ff in range(FORCAST):
    xx = x[len(x)-seq_len:len(x)]
    yy = model(torch.Tensor(xx).reshape(1, xx.shape[0]))
    x = np.concatenate((x, yy.detach().numpy().reshape(1,)))


# Plot Predictions
plt.plot(range(x[:DATA_SIZE].shape[0]), x[:DATA_SIZE])
plt.plot(range(x[:DATA_SIZE].shape[0], x.shape[0]), x[DATA_SIZE:DATA_SIZE+FORCAST], 'r--')
plt.plot(range(x[:DATA_SIZE].shape[0], x.shape[0]), np.sin(np.linspace(10, 20, DATA_SIZE)), 'g-')
plt.show()