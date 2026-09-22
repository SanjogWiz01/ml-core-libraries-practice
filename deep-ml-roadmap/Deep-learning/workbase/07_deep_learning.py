import torch
from torch import nn
torch.manual_seed(42); x=torch.linspace(-2,2,200).unsqueeze(1); y=3*x+1+.2*torch.randn_like(x)
m=nn.Sequential(nn.Linear(1,16),nn.ReLU(),nn.Linear(16,1)); loss=nn.MSELoss(); opt=torch.optim.Adam(m.parameters(),.01)
for _ in range(200):
 p=m(x); l=loss(p,y); opt.zero_grad(); l.backward(); opt.step()
print(l.item())