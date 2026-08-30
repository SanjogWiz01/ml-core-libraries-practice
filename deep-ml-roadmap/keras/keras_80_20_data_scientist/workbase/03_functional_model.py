import keras
from keras import layers

inputs = keras.Input(shape=(10,))
x = layers.Dense(64, activation="relu")(inputs)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(1)(x)
model = keras.Model(inputs, outputs)
model.summary()
