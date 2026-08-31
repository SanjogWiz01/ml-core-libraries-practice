from torch import nn
class MLP(nn.Module):
 def __init__(self): super().__init__(); self.net=nn.Sequential(nn.Linear(10,32),nn.ReLU(),nn.Linear(32,2))
 def forward(self,x): return self.net(x)
print(MLP())
