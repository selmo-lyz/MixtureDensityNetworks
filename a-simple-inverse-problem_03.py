# 以普通的 NN + MDN 解 A Simple Inverse Problem 中 x-t 的問題

import numpy as np
import matplotlib.pyplot as plt
from utils.utils import dataGen4InvProb
from models.mdn import MDN, LossFuncMDN
from utils.data import SimpleDataset, InferDataset
from utils.train import train_closure
from utils.eval import evaluation
import torch
from torch.utils.data import DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using {}!".format(device))

# Hyperparameters
num_gaussians = 3
num_params = 3
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
model = MDN(num_params*num_gaussians, num_gaussians).to(device)
# Loss Function
criterion = LossFuncMDN(num_gaussians, "sum")
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
    if  (loss_min - loss_valid) > 1e-8:
        loss_min = loss_valid
        epoch_min = epoch+1
        torch.save(model.state_dict(), "./x-t_param_MDN.pkl")
print("\n")

plt.plot(np.arange(1, num_epoch+1), history, linewidth=1)
plt.savefig("./x-t_MDN_history.png")
plt.show()
plt.clf()

# Evaluation
model.load_state_dict(torch.load("./x-t_param_MDN.pkl"))
model.eval()

t_te, x_te = dataGen4InvProb(1000)
ds_te = SimpleDataset(x_te, t_te)
dl_te = DataLoader(dataset=ds_te, batch_size=batch_size)
print("Loss: {}".format(evaluation(model, criterion, dl_te, device)))

# Inference
x_if = np.arange(1000) / 1000
ds_if = InferDataset(x_if)
dl_if = DataLoader(dataset=ds_if, batch_size=batch_size)

upper, mean, lower = None, None, None
for xi in dl_if:
    xi = xi.to(device).float()

    output = model(xi)
    pred_ti = output.detach().numpy()
    alpha_x, mu_x, sigma_x = np.split(pred_ti.reshape(-1, num_gaussians), num_params, axis=0)

    upper_x = np.sum(alpha_x*(mu_x + sigma_x))
    mean_x = np.sum(alpha_x*(mu_x))
    lower_x = np.sum(alpha_x*(mu_x - sigma_x))

    if upper is None:
        upper = upper_x
        mean = mean_x
        lower = lower_x
    else:
        upper = np.append(upper, upper_x)
        mean = np.append(mean, mean_x)
        lower = np.append(lower, lower_x)

plt.scatter(x, t, c=(1,1,1,0), edgecolors=(0.3,0.3,0.3), linewidths=0.5)
plt.plot(x_if, upper, c=(0,0,1), linewidth=3, linestyle="--")
plt.plot(x_if, mean, c=(0,0,1), linewidth=3)
plt.plot(x_if, lower, c=(0,0,1), linewidth=3, linestyle="--")
plt.savefig(f"./images/x-t_MDN_Adam-lr-{learning_rate}_epoch-{epoch_min}.png")
plt.show()