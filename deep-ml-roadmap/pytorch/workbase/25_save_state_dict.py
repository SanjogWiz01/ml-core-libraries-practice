import torch
from torch import nn
m=nn.Linear(5,1); torch.save(m.state_dict(),'model_weights.pth'); n=nn.Linear(5,1); n.load_state_dict(torch.load('model_weights.pth',weights_only=True)); n.eval(); print('restored')
