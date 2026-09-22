import torch
from torch import nn
X=torch.randn(300,5); y=(X.sum(1)>0).long()
loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X,y),32,shuffle=True)
m=nn.Sequential(nn.Linear(5,16),nn.ReLU(),nn.Linear(16,2)); loss=nn.CrossEntropyLoss(); opt=torch.optim.Adam(m.parameters(),.01)
for e in range(5):
 for xb,yb in loader:
  l=loss(m(xb),yb); opt.zero_grad(); l.backward(); opt.step()
 print(e+1,l.item())