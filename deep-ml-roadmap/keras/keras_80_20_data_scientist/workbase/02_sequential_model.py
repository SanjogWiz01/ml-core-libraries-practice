import keras
from keras import layers

model = keras.Sequential([
    keras.Input(shape=(10,)),
    layers.Dense(64, activation="relu"),
    layers.Dense(1)
])
model.summary()
