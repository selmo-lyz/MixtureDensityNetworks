import torch
import torch.nn as nn

class simpleNN(nn.Module):
    def __init__(self):
        super(simpleNN, self).__init__()

        self.nn = nn.Sequential(
            nn.Linear(1, 5),
            nn.Tanh(),
            nn.Linear(5, 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.nn(x)

        return x
  
class MDN(nn.Module):
    def __init__(self):
        super(MDN, self).__init__()

        self.nn = nn.Sequential(
            nn.Linear(1, 20),
            nn.Tanh(),
            nn.Linear(20, 9),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.nn(x)

        return x