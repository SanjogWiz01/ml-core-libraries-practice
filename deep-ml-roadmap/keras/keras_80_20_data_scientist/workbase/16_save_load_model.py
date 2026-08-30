import keras

model = keras.Sequential([
    keras.Input(shape=(5,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1)
])
model.save("demo_model.keras")
loaded = keras.models.load_model("demo_model.keras")
loaded.summary()
