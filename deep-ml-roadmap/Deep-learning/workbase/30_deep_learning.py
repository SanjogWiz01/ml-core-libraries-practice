import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
torch.manual_seed(42); X=torch.randn(1000,10); y=(X[:,0]+X[:,1]-X[:,2]>0).long()
loader=DataLoader(TensorDataset(X[:800],y[:800]),32,shuffle=True)
m=nn.Sequential(nn.Linear(10,64),nn.ReLU(),nn.Dropout(.2),nn.Linear(64,2)); loss=nn.CrossEntropyLoss(); opt=torch.optim.AdamW(m.parameters(),.003)
for _ in range(10):
 for xb,yb in loader:
  l=loss(m(xb),yb); opt.zero_grad(); l.backward(); opt.step()
m.eval()
with torch.no_grad(): print("test accuracy",((m(X[800:]).argmax(1)==y[800:]).float().mean().item()))