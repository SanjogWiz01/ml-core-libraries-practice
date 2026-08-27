# pytorch — Research & Custom Loops

Stage 4 of the roadmap. PyTorch rewards understanding the mechanics — use it for research-style experiments.

## Topics

- **Tensors & Autograd**: dynamic graphs, backward passes, custom gradients
- **Datasets**: Dataset, DataLoader, custom collate functions
- **Training Loops**: manual forward/backward/optimize cycles
- **Architectures**: custom nn.Module subclasses, ResNet, Transformers
- **Experiments**: checkpointing, learning rate scheduling, mixed precision

## Structure

```
pytorch/
  tensors/          # tensor ops, autograd, device management
  datasets/         # Dataset and DataLoader patterns
  training/         # full training loop templates
  architectures/    # custom model definitions
  experiments/      # research-style experiment scripts
  exercises/        # practice problems
```

## Key Concepts to Lock In

1. Dynamic computation graph (define-by-run)
2. The training loop: zero_grad → forward → loss → backward → step
3. nn.Module as the unit of composable architecture
4. DataLoader workers and pin_memory for throughput
