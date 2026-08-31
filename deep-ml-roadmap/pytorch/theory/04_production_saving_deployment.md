# Saving, Deployment & Production

### State dict
Recommended weight persistence:
```python
torch.save(model.state_dict(), "model.pth")
model.load_state_dict(torch.load("model.pth", weights_only=True))
model.eval()
```

### Checkpoint
Save epoch, model state, optimizer state and relevant metrics when resuming training.

### Inference
```python
model.eval()
with torch.no_grad():
    prediction = model(X)
```

### Production checklist
- version data and preprocessing
- preserve model artifacts
- pin dependencies
- validate shapes/dtypes
- test CPU/GPU behavior
- measure latency
- monitor drift and performance

### 80/20
`Tensor -> Dataset/DataLoader -> nn.Module -> forward -> loss -> backward -> optimizer.step -> eval/no_grad -> save/load -> transfer learning`
