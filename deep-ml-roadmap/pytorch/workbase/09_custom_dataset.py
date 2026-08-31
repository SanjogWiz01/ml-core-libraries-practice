import torch
from torch.utils.data import Dataset
class ToyDataset(Dataset):
 def __init__(self,n=100): self.X=torch.randn(n,5); self.y=(self.X[:,0]>0).long()
 def __len__(self): return len(self.X)
 def __getitem__(self,i): return self.X[i],self.y[i]
d=ToyDataset(); print(len(d),d[0])
