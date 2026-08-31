import torch
from torch import nn
from torch.utils.data import TensorDataset,DataLoader
X=torch.randn(1000,6); y=(X[:,0]-X[:,1]>0).long(); loader=DataLoader(TensorDataset(X,y),32,shuffle=True)
m=nn.Sequential(nn.Linear(6,32),nn.ReLU(),nn.Linear(32,2)); loss=nn.CrossEntropyLoss(); opt=torch.optim.Adam(m.parameters(),lr=1e-3)
for epoch in range(5):
 total=0
 for xb,yb in loader:
  opt.zero_grad(); l=loss(m(xb),yb); l.backward(); opt.step(); total+=l.item()
 print(epoch+1,total/len(loader))
