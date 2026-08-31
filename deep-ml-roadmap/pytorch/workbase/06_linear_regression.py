import torch
from torch import nn
torch.manual_seed(42); X=torch.randn(500,1); y=3*X+2+.2*torch.randn(500,1)
m=nn.Linear(1,1); loss=nn.MSELoss(); opt=torch.optim.SGD(m.parameters(),lr=.05)
for _ in range(100):
 p=m(X); l=loss(p,y); opt.zero_grad(); l.backward(); opt.step()
print(m.weight.item(),m.bias.item())
