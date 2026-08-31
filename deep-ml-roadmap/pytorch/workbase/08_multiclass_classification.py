import torch
from torch import nn
X=torch.randn(1200,10); y=torch.randint(0,4,(1200,)); m=nn.Sequential(nn.Linear(10,64),nn.ReLU(),nn.Linear(64,4)); loss=nn.CrossEntropyLoss(); opt=torch.optim.Adam(m.parameters(),lr=1e-3)
for _ in range(20):
 l=loss(m(X),y); opt.zero_grad(); l.backward(); opt.step()
print(l.item())
