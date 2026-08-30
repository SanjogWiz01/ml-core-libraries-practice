import keras

model = keras.Sequential([
    keras.Input(shape=(5,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1)
])
model.save_weights("demo.weights.h5")

new_model = keras.Sequential([
    keras.Input(shape=(5,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1)
])
new_model.load_weights("demo.weights.h5")
