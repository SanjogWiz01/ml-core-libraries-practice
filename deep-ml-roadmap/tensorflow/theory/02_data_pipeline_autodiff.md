# Data Pipelines & Automatic Differentiation

## tf.data
Typical pipeline:
```python
ds = tf.data.Dataset.from_tensor_slices((X, y))
ds = ds.shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
```

Prioritize `from_tensor_slices`, `shuffle`, `batch`, `map`, `cache`, `prefetch`, and `take`.

## GradientTape
```python
with tf.GradientTape() as tape:
    prediction = model(x)
    loss = loss_fn(y, prediction)
grads = tape.gradient(loss, model.trainable_variables)
optimizer.apply_gradients(zip(grads, model.trainable_variables))
```

Use `GradientTape` for custom training logic; use Keras `fit()` for most standard workflows.