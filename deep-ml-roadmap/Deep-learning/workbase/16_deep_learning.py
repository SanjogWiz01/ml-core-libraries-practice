import torch
m=torch.nn.Linear(4,2); opt=torch.optim.Adam(m.parameters(),.01); sch=torch.optim.lr_scheduler.StepLR(opt,5,.5)
for e in range(10): print(e,opt.param_groups[0]["lr"]); sch.step()