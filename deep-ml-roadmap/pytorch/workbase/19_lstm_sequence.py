import torch
from torch import nn
m=nn.LSTM(8,64,batch_first=True); out,(h,c)=m(torch.randn(32,30,8)); print(out.shape,h.shape)
