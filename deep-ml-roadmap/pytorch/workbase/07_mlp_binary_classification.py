import torch
from torch import nn
X=torch.randn(1000,8); y=(X[:,0]+X[:,1]>0).float().unsqueeze(1)
m=nn.Sequential(nn.Linear(8,32),nn.ReLU(),nn.Dropout(.2),nn.Linear(32,1)); loss=nn.BCEWithLogitsLoss(); opt=torch.optim.Adam(m.parameters(),lr=1e-3)
for _ in range(20):
 l=loss(m(X),y); opt.zero_grad(); l.backward(); opt.step()
print(l.item())
