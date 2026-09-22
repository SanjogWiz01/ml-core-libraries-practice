import torch
from torch import nn
class TextClassifier(nn.Module):
 def __init__(self): super().__init__(); self.e=nn.Embedding(5000,64); self.r=nn.GRU(64,64,batch_first=True); self.fc=nn.Linear(64,2)
 def forward(self,x): _,h=self.r(self.e(x)); return self.fc(h[-1])
print(TextClassifier()(torch.randint(0,5000,(8,30))).shape)