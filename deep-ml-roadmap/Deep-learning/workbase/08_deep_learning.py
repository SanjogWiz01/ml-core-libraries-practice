import torch
from torch import nn
torch.manual_seed(42); x=torch.randn(500,2); y=((x.sum(1)>0).float()).unsqueeze(1)
m=nn.Sequential(nn.Linear(2,16),nn.ReLU(),nn.Linear(16,1)); loss=nn.BCEWithLogitsLoss(); opt=torch.optim.Adam(m.parameters(),.01)
for _ in range(100):
 l=loss(m(x),y); opt.zero_grad(); l.backward(); opt.step()
with torch.no_grad(): print(((torch.sigmoid(m(x))>.5)==y).float().mean().item())