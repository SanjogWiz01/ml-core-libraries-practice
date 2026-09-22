import torch
from torch import nn
m=nn.GRU(8,32,batch_first=True); out,h=m(torch.randn(16,20,8)); print(out.shape,h.shape)