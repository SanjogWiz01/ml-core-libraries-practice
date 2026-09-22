# Production and Experimentation
- Training updates parameters; inference only predicts.
- Save model weights/checkpoints plus configuration needed for reproducibility.
- GPU accelerates tensor operations; verify model/data device.
- Mixed precision can improve memory use and throughput on supported hardware.
- Batch inference prevents unnecessary memory usage.
- Track data version, seed, hyperparameters, architecture, metrics and checkpoint.
- Workflow: problem -> data -> baseline -> model -> validation -> error analysis -> tuning -> reproducibility -> inference.
