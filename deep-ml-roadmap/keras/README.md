# keras — Rapid Prototyping

Stage 2 of the roadmap. High-level API for building and validating ideas fast before going lower-level.

## Topics

- **Sequential API**: stacking layers, compile, fit, evaluate
- **Functional API**: multi-input/output models, shared layers, residual connections
- **Built-in layers**: Dense, Conv2D, LSTM, Dropout, BatchNormalization
- **Callbacks**: EarlyStopping, ModelCheckpoint, LearningRateScheduler, TensorBoard
- **Transfer learning**: loading pretrained models, fine-tuning

## Structure

```
keras/
  sequential/       # simple stacked models
  functional/       # complex graph-style models
  callbacks/        # training utilities
  transfer/         # pretrained model usage
  exercises/        # practice problems
```

## Key Concepts to Lock In

1. compile() defines the learning contract (optimizer, loss, metrics)
2. Functional API = DAG of layers, not a list
3. Callbacks are hooks — don't rewrite training logic to add features
4. fit() history object is your first debugging tool
