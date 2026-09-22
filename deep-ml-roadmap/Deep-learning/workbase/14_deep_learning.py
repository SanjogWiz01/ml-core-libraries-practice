import torch
from torch import nn
print(nn.Sequential(nn.Linear(20,64),nn.BatchNorm1d(64),nn.ReLU(),nn.Linear(64,2)))