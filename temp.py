import numpy as np
import matplotlib.pyplot as plt
from .utils.utils import dataGen4InvProb
from .models.nn import SimpleNN
from .utils.data import SimpleDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Parameters
numOfPoints = 1000
path = "./param_simple_nn.pkl"

# Generate data
t, x = dataGen4InvProb(numOfPoints)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using {}!".format(device))

ds_te = SimpleDataset(t, x)
dl_te = DataLoader(dataset=ds_te)

# Model
model = SimpleNN().to(device)
model.load_state_dict(torch.load(path))
for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values: {param[:]}")

# Evaluation
pred_x = None
model.eval()
for idx, (ti, xi) in enumerate(dl_te):
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
plt.show()