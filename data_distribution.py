# 請參考原論文中 An Simple Inverse Problem 之章節。
# 此程式用來產生該章節中所描述的 training data，
# 並將其以 2D 的散步圖呈現。

import numpy as np
import matplotlib.pyplot as plt

# Parameters
numOfPoints = 1000

# Data Generator
# t -> [0, 1) with uniform distribution
t = np.random.rand(numOfPoints)
# epsilon -> (-0.1, 0.1) with uniform distribution
epsilon = (np.random.rand(len(t)) - 0.5) / 5
x = t + 0.3 * np.sin(2 * np.pi * t) + epsilon

# Statistics
print(np.min(x), np.max(x), x.shape)

# Figure Settings
plt.xlabel("t")
plt.xlim(0.0, 1.0)
plt.ylabel("x")
plt.ylim(0.0, 1.0)

plt.scatter(t, x, c=(1,1,1,0), edgecolors=(0,0,0), linewidths=0.5)
plt.savefig("t-x_sample-1000.png")
plt.show()

plt.clf()

# Figure Settings
plt.xlabel("x")
plt.xlim(0.0, 1.0)
plt.ylabel("t")
plt.ylim(0.0, 1.0)

plt.scatter(x, t, c=(1,1,1,0), edgecolors=(0,0,0), linewidths=0.5)
plt.savefig("x-t_sample-1000.png")
plt.show()