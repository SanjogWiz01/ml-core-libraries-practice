import torch
from torch import nn
m=nn.Embedding(10000,64); tokens=torch.randint(0,10000,(16,20)); print(m(tokens).shape)
