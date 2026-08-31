import torch
x,y=torch.randn(2,3),torch.randn(2,3)
print(x+y); print(x*y); print(x@y.T); print(x.reshape(3,2))
