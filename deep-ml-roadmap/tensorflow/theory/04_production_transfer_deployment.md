# Transfer Learning, Saving & Production

## Transfer learning
1. Load pretrained backbone.
2. Freeze backbone.
3. Add task head.
4. Train head.
5. Optionally unfreeze selected layers.
6. Fine-tune with a small learning rate.

## Saving
```python
model.save('model.keras')
loaded = tf.keras.models.load_model('model.keras')
model.save_weights('weights.weights.h5')
```

## Production checklist
Version data/preprocessing, keep test data untouched, save model artifacts, pin dependencies, test inference shapes/dtypes, measure latency, and monitor drift.

## 80/20 priority
`tensors -> tf.data -> Keras -> compile/fit -> callbacks -> evaluate/predict -> GradientTape -> save/load -> transfer learning`.