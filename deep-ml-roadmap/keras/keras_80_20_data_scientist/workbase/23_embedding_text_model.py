import keras
from keras import layers

model = keras.Sequential([
    keras.Input(shape=(100,), dtype="int32"),
    layers.Embedding(input_dim=10000, output_dim=64),
    layers.GlobalAveragePooling1D(),
    layers.Dense(32, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])
model.summary()
