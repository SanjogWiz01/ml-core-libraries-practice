import torch
x=torch.tensor(3.,requires_grad=True); y=x**2+2*x; y.backward(); print(y.item(),x.grad.item())
