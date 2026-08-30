import keras
from keras import layers

model = keras.Sequential([
    keras.Input(shape=(10,)),
    layers.Dense(32, activation="relu", name="hidden"),
    layers.Dense(1, name="output")
])
model.summary()
for layer in model.layers:
    print(layer.name, layer.count_params())
