# Production, Saving, Transfer Learning & Deployment

## Save the complete model
Keras 3 uses the `.keras` format for whole-model saving.

```python
model.save("model.keras")
loaded = keras.models.load_model("model.keras")
```

A saved model can contain architecture, weights, compilation configuration, and optimizer state.

## Save weights
```python
model.save_weights("model.weights.h5")
loaded.load_weights("model.weights.h5")
```

## Export
For inference-oriented deployment, Keras provides `model.export()` with supported export targets depending on the installed ecosystem.

## Transfer learning workflow
1. Load pretrained backbone.
2. Freeze backbone.
3. Add task-specific head.
4. Train head.
5. Unfreeze selected backbone layers.
6. Fine-tune with a low learning rate.
7. Evaluate on untouched test data.

## Production checklist
- version datasets
- freeze preprocessing
- save model
- record dependencies
- test inference shape/dtype
- monitor latency
- monitor drift
- log predictions safely
- validate model after loading

## 80/20 rule
For most data-science projects, become excellent at:
`Sequential + Functional API + Dense/Conv/LSTM + compile/fit/evaluate/predict + callbacks + preprocessing + transfer learning + saving/loading`

Advanced topics such as custom layers, subclassing, custom loops, distributed training, quantization and specialized deployment should come after this foundation.
