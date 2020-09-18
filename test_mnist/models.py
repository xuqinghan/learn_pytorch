from torch import nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, dim_in, n_h1, n_h2, dim_out):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(dim_in, n_h1), nn.BatchNorm1d(n_h1))
        self.layer2 = nn.Sequential(nn.Linear(n_h1, n_h2), nn.BatchNorm1d(n_h2))
        self.layer3 = nn.Sequential(nn.Linear(n_h2, dim_out))
    
    def  forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x
