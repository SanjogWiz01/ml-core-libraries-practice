import torch
from torch import nn
class WeightedMSE(nn.Module):
 def forward(self,p,t): return ((1+t.abs())*(p-t)**2).mean()
print(WeightedMSE()(torch.tensor([[2.]]),torch.tensor([[1.]])))
