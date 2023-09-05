import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(28*28,512)
        self.batch_norm1 = nn.BatchNorm1d(512)
        self.layer2 = nn.Linear(512,512)
        self.layer3 = nn.Linear(512,256)
        self.layer4 = nn.Linear(256,64)
        self.batch_norm2 = nn.BatchNorm1d(64)
        self.layer5 = nn.Linear(64,64)
        self.final = nn.Linear(64,10)
    def forward(self,x):
        x = f.relu(self.batch_norm1(self.layer1(x)))
        x = f.relu(self.layer2(x))
        x = f.relu(self.layer3(x))
        x = f.relu(self.batch_norm2(self.layer4(x)))
        x = f.relu(self.layer5(x))
        x = self.final(x)
        x = f.log_softmax(x,dim=1)
        return x        