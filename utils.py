import numpy as np
import matplotlib.pyplot as plt

# Data Generator for A Simple Inverse Problem
def dataGen4InvProb(amountOfData):
    # t -> [0, 1) with uniform distribution
    t = np.random.rand(amountOfData)
    # epsilon -> (-0.1, 0.1) with uniform distribution
    epsilon = (np.random.rand(len(t)) - 0.5) / 5
    x = t + 0.3 * np.sin(2 * np.pi * t) + epsilon

    return t, x