# Data, Autograd & Training

### Dataset / DataLoader
`Dataset` defines samples; `DataLoader` batches and iterates them.
Custom datasets normally implement `__len__` and `__getitem__`.

### Canonical training loop
```python
model.train()
for X, y in loader:
    optimizer.zero_grad()
    pred = model(X)
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()
```

### Evaluation
```python
model.eval()
with torch.no_grad():
    prediction = model(X)
```

### Losses
- Regression: `MSELoss`, `L1Loss`, `HuberLoss`
- Binary/multi-label: `BCEWithLogitsLoss`
- Multi-class: `CrossEntropyLoss`

`CrossEntropyLoss` expects raw logits, not softmax probabilities.

### Optimizers
Most useful: `Adam`, `AdamW`, `SGD`.
Key hyperparameters: learning rate, weight decay, momentum, batch size, epochs.

### Data-science discipline
Split before fitting learned preprocessing. Keep validation/test untouched during training and record experiment configuration.
