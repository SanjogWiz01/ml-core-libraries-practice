import torch
from torch import nn
m=nn.Sequential(nn.Linear(4,16),nn.ReLU(),nn.Linear(16,2)); X=torch.randn(20,4); m.eval()
with torch.no_grad(): pred=m(X).argmax(1)
print(pred)
