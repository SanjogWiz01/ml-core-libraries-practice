import torch
from torchvision.models import resnet18,ResNet18_Weights
m=resnet18(weights=ResNet18_Weights.DEFAULT)
for p in m.parameters(): p.requires_grad=False
m.fc=torch.nn.Linear(m.fc.in_features,10); print(m.fc)