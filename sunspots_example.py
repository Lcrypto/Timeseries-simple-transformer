#Imports
import torch
import numpy as np
import pandas as pd
from datasets import Dataset
import matplotlib.pyplot as plt
from forcasting_model import ForcastingModel
from torch.utils.data import TensorDataset, DataLoader


# Create a dataset
seq_len = 200
data = list(pd.read_csv("sunspots.csv")["Monthly Mean Total Sunspot Number"])[1000:]
x = np.array(data[:2000])
forcast = np.array(data[2000:])
X = np.array([x[ii:ii+seq_len] for ii in range(0, x.shape[0]-seq_len)])
Y = np.array([x[ii+seq_len] for ii in range(0, x.shape[0]-seq_len)])


# New Training Loop
EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 4.12e-6
model = ForcastingModel(seq_len).to("cuda")
model.train()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
dataset = TensorDataset(torch.Tensor(X).to("cuda"), torch.Tensor(Y).to("cuda"))
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
for epoch in range(EPOCHS):
    for xx, yy in dataloader:
        optimizer.zero_grad()
        out = model(xx)
        loss = criterion(out, yy)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{EPOCHS}: Loss={loss}")


# New Prediction Loop
model.eval()
for ff in range(len(forcast)):
    xx = x[len(x)-seq_len:len(x)]
    yy = model(torch.Tensor(xx).reshape(1, xx.shape[0]).to("cuda"))
    x = np.concatenate((x, yy.detach().cpu().numpy().reshape(1,)))


# Plot New Predictions
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(6, 6))
plt.plot(range(2000), data[:2000], label="Training")
plt.plot(range(2000, len(data)), forcast, 'g-', label="Actual")
plt.plot(range(2000, len(data)), x[2000:], 'r--', label="Predicted")
plt.legend()
fig.savefig("./img/sunspots_example.png")
