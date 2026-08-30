import keras
from keras import layers

model = keras.Sequential([
    keras.Input(shape=(64,64,3)),
    layers.Rescaling(1./255),
    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation="relu"),
    layers.GlobalAveragePooling2D(),
    layers.Dense(10, activation="softmax")
])
model.summary()
