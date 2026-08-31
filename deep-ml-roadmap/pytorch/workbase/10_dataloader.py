import torch
from torch.utils.data import TensorDataset,DataLoader
X=torch.randn(100,4); y=torch.randint(0,2,(100,)); loader=DataLoader(TensorDataset(X,y),batch_size=16,shuffle=True)
for xb,yb in loader: print(xb.shape,yb.shape); break
