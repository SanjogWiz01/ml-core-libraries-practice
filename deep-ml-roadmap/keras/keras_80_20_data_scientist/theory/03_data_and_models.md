# Data Pipelines, Preprocessing & Model Design

## Data flow
Separate the data into:
- train
- validation
- test

Never fit preprocessing transformations on the test set.

### Keras preprocessing layers
Useful layers include:
- `Normalization`
- `Rescaling`
- `RandomFlip`
- `RandomRotation`
- `RandomZoom`
- `TextVectorization`
- categorical preprocessing layers

Example:
```python
normalizer = keras.layers.Normalization()
normalizer.adapt(X_train)
```

### Tabular deep learning
A practical pattern is:
`numeric features -> normalization -> Dense -> dropout -> Dense -> output`

For categorical features, use an appropriate encoding strategy. For many real-world tabular problems, also benchmark against tree-based models from scikit-learn/XGBoost rather than assuming neural networks will win.

### Images
Common pipeline:
`image -> resize/rescale -> augmentation -> CNN/transfer-learning backbone -> classifier`

### Transfer learning
Start with a pretrained model, freeze the backbone, train a new classification head, then optionally fine-tune selected layers using a small learning rate.

### Functional API
Use Functional API when the architecture is not a simple stack:
```python
inputs = keras.Input(shape=(20,))
x = keras.layers.Dense(64, activation="relu")(inputs)
x = keras.layers.Dense(32, activation="relu")(x)
outputs = keras.layers.Dense(1)(x)
model = keras.Model(inputs, outputs)
```

### Model selection
Choose based on:
- data type
- sample size
- compute
- latency
- interpretability
- deployment requirements
- baseline performance

Keras is a tool, not the objective. Always compare against a sensible baseline.
