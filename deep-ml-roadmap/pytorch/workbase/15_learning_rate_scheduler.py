import torch
from torch import nn
m=nn.Linear(5,1); opt=torch.optim.Adam(m.parameters(),lr=1e-3); sch=torch.optim.lr_scheduler.StepLR(opt,5,.5)
for e in range(12): print(e,opt.param_groups[0]['lr']); opt.step(); sch.step()
