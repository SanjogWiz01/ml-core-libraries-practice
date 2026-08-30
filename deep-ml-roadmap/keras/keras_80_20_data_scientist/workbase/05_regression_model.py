import keras
from keras import layers
import numpy as np

X = np.random.randn(1000, 5).astype("float32")
y = (2*X[:,0] - 3*X[:,1] + 0.5).astype("float32")

model = keras.Sequential([
    keras.Input(shape=(5,)),
    layers.Dense(32, activation="relu"),
    layers.Dense(1)
])
model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model.fit(X, y, epochs=10, validation_split=0.2, verbose=0)
