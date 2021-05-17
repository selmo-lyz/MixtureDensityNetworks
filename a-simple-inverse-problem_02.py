# 以普通的 NN 解 A Simple Inverse Problem 中 x-t 的問題

import numpy as np
import matplotlib.pyplot as plt
from utils.utils import dataGen4InvProb
from models.nn import SimpleNN
from utils.data import SimpleDataset, InferDataset
from utils.train import train_closure
from utils.eval import evaluation
from utils.infer import inference
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using {}!".format(device))

# Hyperparameters
batch_size = 1
learning_rate = 1e-3
num_epoch = 1000

# Generate data
t, x = dataGen4InvProb(1000)
t_va, x_va = dataGen4InvProb(1000)
# Dataset
ds_tr = SimpleDataset(x, t)
dl_tr = DataLoader(dataset=ds_tr, batch_size=batch_size)
ds_va = SimpleDataset(x_va, t_va)
dl_va = DataLoader(dataset=ds_va, batch_size=batch_size)
# Model
model = SimpleNN().to(device)
# Loss Function
criterion = nn.MSELoss(reduction='mean')
# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training
print()
loss_min = 99
epoch_min = -1
history = np.empty(0)
loss_valid = 0.0
for epoch in range(0, num_epoch):
    # Train
    train_closure(model, criterion, optimizer, dl_tr, device)

    # Validate
    loss_valid = evaluation(model, criterion, dl_va, device)

    # Prompt messages
    print("\r Epoch [{}/{}] - Loss (valid): {}".format(
          epoch+1, num_epoch, loss_valid), end='')

    # Log
    history = np.append(history, loss_valid)
    # Save Parameters of Model
    if loss_valid < loss_min:
        loss_min = loss_valid
        epoch_min = epoch+1
        torch.save(model.state_dict(), "./x-t_param_simpleNN.pkl")
print("\n")

plt.plot(np.arange(1, num_epoch+1), history, linewidth=1)
plt.savefig("./x-t_MSE_history.png")
plt.show()
plt.clf()

# Evaluation
model.load_state_dict(torch.load("./x-t_param_simpleNN.pkl"))
model.eval()

t_te, x_te = dataGen4InvProb(1000)
ds_te = SimpleDataset(x_te, t_te)
dl_te = DataLoader(dataset=ds_te, batch_size=batch_size)
print("Loss: {}".format(evaluation(model, criterion, dl_te, device)))

# Inference
x_if = np.arange(1000) / 1000
ds_if = InferDataset(x_if)
dl_if = DataLoader(dataset=ds_if, batch_size=batch_size)
pred_t = inference(model, dl_if, device)

plt.scatter(x, t, c=(1,1,1,0), edgecolors=(0.3,0.3,0.3), linewidths=0.5)
plt.plot(x_if, pred_t, c=(0,0,0), linewidth=3)
plt.savefig(f"./images/x-t_MSE_Adam-lr-{learning_rate}_epoch-{epoch_min}.png")
plt.show()