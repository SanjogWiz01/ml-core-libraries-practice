import torch
from torch import nn
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); m=nn.Linear(10,1).to(device)
if device.type=='cuda':
 scaler=torch.amp.GradScaler('cuda'); X=torch.randn(32,10,device=device); y=torch.randn(32,1,device=device); opt=torch.optim.Adam(m.parameters())
 opt.zero_grad()
 with torch.autocast(device_type='cuda',dtype=torch.float16): l=((m(X)-y)**2).mean()
 scaler.scale(l).backward(); scaler.step(opt); scaler.update(); print('AMP step complete')
else: print('CUDA unavailable')
