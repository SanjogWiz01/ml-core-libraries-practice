import torch
from torch import nn
print(nn.Sequential(nn.Linear(20,64),nn.ReLU(),nn.Dropout(.3),nn.Linear(64,2)))