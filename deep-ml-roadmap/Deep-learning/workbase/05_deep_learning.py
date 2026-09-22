import torch
from torch import nn
model=nn.Sequential(nn.Linear(4,16),nn.ReLU(),nn.Linear(16,1))
print(model(torch.randn(8,4)).shape)