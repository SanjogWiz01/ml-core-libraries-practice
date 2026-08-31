import torch
from torch import nn
m=nn.Linear(5,1); opt=torch.optim.Adam(m.parameters(),lr=1e-3); ck={'epoch':10,'model_state_dict':m.state_dict(),'optimizer_state_dict':opt.state_dict()}; torch.save(ck,'checkpoint.pth'); z=torch.load('checkpoint.pth',weights_only=True); m.load_state_dict(z['model_state_dict']); opt.load_state_dict(z['optimizer_state_dict']); print(z['epoch'])
