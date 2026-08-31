def train_one_epoch(model,loader,loss_fn,optimizer,device):
 model.train(); total=0
 for X,y in loader:
  X,y=X.to(device),y.to(device); optimizer.zero_grad(); loss=loss_fn(model(X),y); loss.backward(); optimizer.step(); total+=loss.item()
 return total/len(loader)
print('training function defined')
