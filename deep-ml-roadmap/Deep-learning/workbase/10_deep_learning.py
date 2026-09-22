import torch
from torch.utils.data import TensorDataset,DataLoader
loader=DataLoader(TensorDataset(torch.randn(100,4),torch.randint(0,2,(100,))),batch_size=16,shuffle=True)
for xb,yb in loader: print(xb.shape,yb.shape); break