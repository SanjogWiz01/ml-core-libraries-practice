from torch import nn
m=nn.Sequential(nn.Linear(20,64),nn.BatchNorm1d(64),nn.ReLU(),nn.Dropout(.3),nn.Linear(64,1)); print(m)
