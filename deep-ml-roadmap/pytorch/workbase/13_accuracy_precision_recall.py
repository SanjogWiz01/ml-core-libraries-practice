import torch
y=torch.tensor([1,1,0,0,1,0]); p=torch.tensor([1,0,0,0,1,1]); tp=((y==1)&(p==1)).sum().item(); fp=((y==0)&(p==1)).sum().item(); fn=((y==1)&(p==0)).sum().item(); tn=((y==0)&(p==0)).sum().item()
print('accuracy',(tp+tn)/len(y),'precision',tp/(tp+fp),'recall',tp/(tp+fn))
