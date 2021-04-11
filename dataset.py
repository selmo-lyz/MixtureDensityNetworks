from torch.utils.data import Dataset

class simpleDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.length = len(self.x)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.length