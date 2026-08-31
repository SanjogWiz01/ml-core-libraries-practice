# PyTorch Core — 80/20
## Mental model
Data -> Dataset/DataLoader -> Model -> Loss -> Optimizer -> Training loop -> Evaluation -> Save/Deploy

### Must know
- `torch.Tensor`: shape, dtype, device, indexing, reshape/view, squeeze/unsqueeze
- `torch.device` and `.to(device)`
- `torch.autograd`: `requires_grad`, `backward`
- `torch.nn.Module` and `forward`
- High-value layers: `Linear`, `ReLU`, `Dropout`, `BatchNorm`, `Conv2d`, `Embedding`, `LSTM`, `GRU`

Example:
```python
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 1)
    def forward(self, x):
        return self.layer(x)
```

### 80/20 priority
Become fluent with tensors, `nn.Module`, data loading, loss, optimization, training/evaluation, device management and model persistence before advanced internals.
