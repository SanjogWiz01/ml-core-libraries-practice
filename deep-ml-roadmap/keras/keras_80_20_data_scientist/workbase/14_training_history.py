import keras
import numpy as np

X = np.random.randn(500, 4).astype("float32")
y = (X[:,0] > 0).astype("float32")

model = keras.Sequential([
    keras.Input(shape=(4,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid")
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
history = model.fit(X, y, epochs=5, validation_split=0.2, verbose=0)
print(history.history.keys())
