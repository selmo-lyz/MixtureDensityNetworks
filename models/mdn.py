import torch
import torch.nn as nn
from utils.utils import gaussian_func
  
class MD3N(nn.Module):
    def __init__(self, out_channels):
        super(MD3N, self).__init__()

        self.nn = nn.Sequential(
            nn.Linear(1, 20),
            nn.Tanh(),
            nn.Linear(20, out_channels),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.nn(x)

class MDN(nn.Module):
    def __init__(self, in_channels, num_gaussians):
        super(MDN, self).__init__()

        self.nn = MD3N(in_channels)
        self.nn_alpha = nn.Sequential(
            nn.Linear(in_channels, num_gaussians),
            nn.Softmax(dim=0)
        )
        self.nn_mu = nn.Linear(in_channels, num_gaussians)
        self.nn_sigma = nn.Sequential(
            nn.Linear(in_channels, num_gaussians),
            nn.ELU()
        )
    
    def forward(self, x):
        x = self.nn(x)
        alphas = self.nn_alpha(x)
        mus = self.nn_mu(x)
        sigmas = torch.add(self.nn_sigma(x), 1)

        return torch.cat([alphas, mus, sigmas], 0)

class LossFuncMDN(nn.Module):
    def __init__(self, num_gaussians, reduction='mean'):
        super(LossFuncMDN, self).__init__()

        self.num_gaussians = num_gaussians
        self.reduction = reduction
    
    def forward(self, x, target):
        alphas, mus, sigmas = torch.split(x, self.num_gaussians)
        nsum = torch.sum(alphas * gaussian_func(mus, sigmas, target))
        loss_from_q = -torch.log(nsum)

        if self.reduction == "mean":
            loss = torch.mean(loss_from_q) 
        if self.reduction == "sum":
            loss = torch.sum(loss_from_q) 
        return loss 