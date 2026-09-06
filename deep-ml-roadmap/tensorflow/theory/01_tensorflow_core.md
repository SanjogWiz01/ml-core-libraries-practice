# TensorFlow Core — 80/20

Focus on `tf.Tensor`, `tf.Variable`, shapes/dtypes, `tf.data`, `GradientTape`, and `tf.keras`. Learn these before low-level graph/compiler APIs.

## Core flow
`Tensor -> data pipeline -> Keras model -> compile -> fit -> evaluate -> predict -> save/deploy`

High-value functions: `tf.constant`, `tf.Variable`, `tf.cast`, `tf.reshape`, `tf.squeeze`, `tf.expand_dims`, `tf.matmul`, `tf.math.*`.

## Keras model styles
- Sequential: simple layer stack
- Functional: multiple inputs/outputs and branching
- Subclassing: highly customized models

## High-value layers
`Dense`, `Dropout`, `BatchNormalization`, `Conv2D`, `MaxPooling2D`, `GlobalAveragePooling2D`, `Embedding`, `LSTM`, `GRU`.