import keras
from keras import layers

model = keras.Sequential([
    keras.Input(shape=(20,)),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(1)
])
model.summary()
