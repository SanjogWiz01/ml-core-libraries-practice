import torch
from torch import nn
X=torch.randn(500,4); y=(X[:,0]>0).long(); m=nn.Sequential(nn.Linear(4,16),nn.ReLU(),nn.Linear(16,2)); opt=torch.optim.Adam(m.parameters(),.01); loss=nn.CrossEntropyLoss()
for _ in range(100):
 l=loss(m(X[:400]),y[:400]); opt.zero_grad(); l.backward(); opt.step()
m.eval()
with torch.no_grad(): print((m(X[400:]).argmax(1)==y[400:]).float().mean().item())