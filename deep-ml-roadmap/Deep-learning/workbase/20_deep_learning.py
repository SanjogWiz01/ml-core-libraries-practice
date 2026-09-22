import torch
from torch import nn
m=nn.LSTM(8,32,batch_first=True); out,(h,c)=m(torch.randn(16,20,8)); print(out.shape,h.shape)