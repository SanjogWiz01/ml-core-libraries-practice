import keras
from keras import layers
import numpy as np

X = np.random.randn(1000, 10).astype("float32")
y = (X[:, 0] + X[:, 1] > 0).astype("float32")

model = keras.Sequential([
    keras.Input(shape=(10,)),
    layers.Dense(32, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X, y, epochs=5, validation_split=0.2, verbose=0)
print(model.evaluate(X, y, verbose=0))
