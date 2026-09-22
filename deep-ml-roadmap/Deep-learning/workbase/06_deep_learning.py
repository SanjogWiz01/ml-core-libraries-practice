import torch
from torch import nn
x=torch.tensor([-2.,-.5,0.,.5,2.])
for n,f in [("ReLU",nn.ReLU()),("Sigmoid",nn.Sigmoid()),("Tanh",nn.Tanh())]: print(n,f(x))