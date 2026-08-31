import torch
from torch import nn
m=nn.Linear(10,1)
for name,opt in [('SGD',torch.optim.SGD(m.parameters(),lr=.01)),('Adam',torch.optim.Adam(m.parameters(),lr=.001)),('AdamW',torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=1e-4))]: print(name,opt)
