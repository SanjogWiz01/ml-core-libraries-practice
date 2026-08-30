import keras
from keras import layers
import numpy as np

X = np.random.randn(1200, 8).astype("float32")
y = np.random.randint(0, 3, size=1200)

model = keras.Sequential([
    keras.Input(shape=(8,)),
    layers.Dense(64, activation="relu"),
    layers.Dense(3, activation="softmax")
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(X, y, epochs=5, validation_split=0.2, verbose=0)
