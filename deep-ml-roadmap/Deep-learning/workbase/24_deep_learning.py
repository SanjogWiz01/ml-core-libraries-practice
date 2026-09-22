import torch
from torch import nn
m=nn.Sequential(nn.Linear(4,8),nn.ReLU(),nn.Linear(8,2)); torch.save(m.state_dict(),"model_weights.pth")
m2=nn.Sequential(nn.Linear(4,8),nn.ReLU(),nn.Linear(8,2)); m2.load_state_dict(torch.load("model_weights.pth",weights_only=True)); print("loaded")