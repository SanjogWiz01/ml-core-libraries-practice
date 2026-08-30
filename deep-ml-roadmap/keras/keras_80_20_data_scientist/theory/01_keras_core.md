# Keras 3 Core — 80/20 Theory

## Goal
Learn the small set of Keras concepts that covers most practical deep-learning work.

### 1. Keras 3
Keras is a high-level deep-learning API that can run with TensorFlow, JAX, or PyTorch backends.

### 2. The core mental model
`Data -> preprocessing -> model -> compile -> fit -> evaluate -> predict -> save/deploy`

The two most important model-building styles are:
- **Sequential**: a simple linear stack of layers.
- **Functional API**: graphs with multiple inputs/outputs, branches, skip connections, and shared layers.

### 3. High-value layers
For tabular data:
- `Dense`
- `Dropout`
- `BatchNormalization`

For images:
- `Conv2D`
- `MaxPooling2D`
- `GlobalAveragePooling2D`

For sequences:
- `Embedding`
- `LSTM`
- `GRU`
- `Conv1D`

### 4. Activation functions
- ReLU: common hidden-layer default.
- Sigmoid: binary probability output.
- Softmax: multi-class probability output.
- Linear: common regression output.

### 5. Build
```python
import keras
from keras import layers

model = keras.Sequential([
    keras.Input(shape=(10,)),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(1)
])
```

### 6. Data scientist priority
Master:
1. model construction
2. loss/optimizer/metrics
3. training
4. validation
5. callbacks
6. evaluation
7. saving/loading
8. transfer learning

Do not start with custom training loops or exotic architectures until the standard workflow is comfortable.
