# TODO
# - [] 以 LBFGS 完成訓練
# - [x] 以 Adam 完成訓練

import numpy as np
import matplotlib.pyplot as plt
from utils import dataGen4InvProb
from simple_model import simpleNN
from dataset import simpleDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using {}!".format(device))

# Generate data
t, x = dataGen4InvProb(1000)

# Hyperparameters
batch_size = 1
learning_rate = 1e-3
num_epoch = 1000

# Dataset
ds_tr = simpleDataset(t, x)
dl_tr = DataLoader(dataset=ds_tr)
# Model
model = simpleNN().to(device)
for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values: {param[:]}")
# Loss Function
criterion = nn.MSELoss(reduction='sum')
# Optimizer
optimizer = torch.optim.LBFGS(model.parameters(), lr=learning_rate, max_iter=100, history_size=1000)
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training
print()
loss_min = 99
epoch_min = -1
for epoch in range(0, num_epoch):
    model.train()
    for idx, (ti, xi) in enumerate(dl_tr):
        ti = ti.to(device).float()
        xi = xi.to(device).float()

        def closure():
            optimizer.zero_grad()
            pred_xi = model(ti)
            loss = criterion(pred_xi, xi)
            loss.backward()
            return loss
        optimizer.step(closure)

    model.eval()
    sumOfLoss = 0
    t_va, x_va = dataGen4InvProb(1000)
    ds_va = simpleDataset(t_va, x_va)
    dl_va = DataLoader(dataset=ds_va)
    for idx, (ti, xi) in enumerate(dl_va):
        ti = ti.to(device).float()
        xi = xi.to(device).float()
        output = model(ti)

        loss = criterion(output, xi)
        sumOfLoss += loss
    loss_avg = sumOfLoss / len(dl_va)
    print("\r Epoch [{}/{}] - Loss: {}".format(epoch+1, num_epoch, loss_avg), end='')

    # Save Parameters of Model
    if loss_avg < loss_min:
        loss_min = loss_avg
        epoch_min = epoch+1
        torch.save(model.state_dict(), "./param_simple_nn.pkl")
print()

# Show parameters
for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values: {param[:]}")

# Evaluation
pred_x = None
model.load_state_dict(torch.load("./param_simple_nn.pkl"))
model.eval()
for idx, (ti, xi) in enumerate(dl_tr):
    ti = ti.to(device).float()
    xi = xi.to(device).float()

    output = model(ti)
    pred_xi = output.detach().numpy()

    if pred_x is not None:
        pred_x = np.concatenate((pred_x, pred_xi), axis=0)
    else:
        pred_x = pred_xi

plt.scatter(t, x, c=(1,1,1,0), edgecolors=(0.3,0.3,0.3), linewidths=0.5)
plt.scatter(t, pred_x, c=(1,1,1,0), edgecolors=(0,0,1))
plt.savefig("./t-x_epoch-{}.png".format(epoch_min))
plt.show()