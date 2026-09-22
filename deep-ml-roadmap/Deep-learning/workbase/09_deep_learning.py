import torch
from torch import nn
torch.manual_seed(42); x=torch.randn(600,4); y=torch.argmax(torch.stack([x[:,0]+x[:,1],-x[:,0]+x[:,2],x[:,3]],1),1)
m=nn.Sequential(nn.Linear(4,32),nn.ReLU(),nn.Linear(32,3)); loss=nn.CrossEntropyLoss(); opt=torch.optim.Adam(m.parameters(),.01)
for _ in range(100):
 l=loss(m(x),y); opt.zero_grad(); l.backward(); opt.step()
print((m(x).argmax(1)==y).float().mean().item())