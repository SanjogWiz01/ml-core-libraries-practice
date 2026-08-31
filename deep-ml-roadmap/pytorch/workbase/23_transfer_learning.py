from torch import nn
from torchvision import models
m=models.resnet18(weights='DEFAULT')
for p in m.parameters(): p.requires_grad=False
m.fc=nn.Linear(m.fc.in_features,2); print(m.fc)
