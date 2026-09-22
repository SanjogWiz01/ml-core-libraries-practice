import torch
from torch import nn
m=nn.Linear(4,2); o=torch.optim.Adam(m.parameters(),.001)
torch.save({"model":m.state_dict(),"optimizer":o.state_dict(),"epoch":5},"checkpoint.pth"); print("checkpoint saved")