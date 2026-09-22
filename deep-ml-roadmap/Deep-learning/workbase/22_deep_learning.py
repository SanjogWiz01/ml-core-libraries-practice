import torch
from torch import nn
m=nn.Embedding(1000,64); print(m(torch.randint(0,1000,(16,20))).shape)