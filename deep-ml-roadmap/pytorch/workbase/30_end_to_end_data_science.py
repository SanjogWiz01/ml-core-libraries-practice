import torch
from torch import nn
from torch.utils.data import TensorDataset,DataLoader
torch.manual_seed(42); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); X=torch.randn(2400,12); y=(X[:,0]+.7*X[:,1]-.4*X[:,2]>0).long(); Xtr,Xte=X[:2000],X[2000:]; ytr,yte=y[:2000],y[2000:]; loader=DataLoader(TensorDataset(Xtr,ytr),64,shuffle=True)
m=nn.Sequential(nn.Linear(12,64),nn.ReLU(),nn.Dropout(.2),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,2)).to(device); loss=nn.CrossEntropyLoss(); opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-4)
for e in range(20):
 total=0
 for xb,yb in loader:
  xb,yb=xb.to(device),yb.to(device); opt.zero_grad(); l=loss(m(xb),yb); l.backward(); opt.step(); total+=l.item()
 if (e+1)%5==0: print(e+1,total/len(loader))
m.eval()
with torch.no_grad(): pred=m(Xte.to(device)).argmax(1); acc=(pred.cpu()==yte).float().mean()
print('test accuracy:',acc.item()); torch.save(m.state_dict(),'final_model.pth')
