import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()

        self.nn = nn.Sequential(
            nn.Linear(1, 5),
            nn.Tanh(),
            nn.Linear(5, 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.nn(x)

        return x