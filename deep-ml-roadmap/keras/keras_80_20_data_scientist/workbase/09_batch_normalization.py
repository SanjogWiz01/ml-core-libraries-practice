import keras
from keras import layers

model = keras.Sequential([
    keras.Input(shape=(20,)),
    layers.Dense(64),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.Dense(1)
])
model.summary()
