import keras
from keras import layers
import numpy as np

X = np.random.randn(500, 6).astype("float32") * 10
normalizer = layers.Normalization()
normalizer.adapt(X)

model = keras.Sequential([
    keras.Input(shape=(6,)),
    normalizer,
    layers.Dense(32, activation="relu"),
    layers.Dense(1)
])
model.summary()
