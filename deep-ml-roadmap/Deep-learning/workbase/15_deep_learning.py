import torch
from torch import nn
for cls in [torch.optim.SGD,torch.optim.Adam,torch.optim.AdamW]:
 m=nn.Linear(4,2); print(cls.__name__,cls(m.parameters(),.01))