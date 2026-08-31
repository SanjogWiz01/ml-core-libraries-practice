from torch import nn
m=nn.Sequential(nn.Linear(20,64),nn.ReLU(),nn.Linear(64,2))
for p in m[0].parameters(): p.requires_grad=False
print([(n,p.requires_grad) for n,p in m.named_parameters()])
