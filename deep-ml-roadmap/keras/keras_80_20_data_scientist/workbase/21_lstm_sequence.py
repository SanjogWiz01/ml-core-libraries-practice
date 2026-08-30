import keras
from keras import layers

model = keras.Sequential([
    keras.Input(shape=(30, 8)),
    layers.LSTM(64),
    layers.Dense(1)
])
model.summary()
