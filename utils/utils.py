import math
import numpy as np
import torch

# Data Generator for A Simple Inverse Problem
def dataGen4InvProb(amountOfData):
    # t -> [0, 1) with uniform distribution
    t = np.random.rand(amountOfData)
    # epsilon -> (-0.1, 0.1) with uniform distribution
    epsilon = (np.random.rand(len(t)) - 0.5) / 5
    x = t + 0.3 * np.sin(2 * np.pi * t) + epsilon
    return t, x

def gaussian_func(mus, sigmas, target):
    gdist = (1.0 / (torch.sqrt(2.0 * math.pi * torch.pow(sigmas, 2)))) * torch.exp((-torch.pow((target - mus), 2)) / (2.0*torch.pow(sigmas, 2)))
    return gdist

# Show parameters
def display_model_params(model):
    for name, param in model.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Values: {param[:]}")