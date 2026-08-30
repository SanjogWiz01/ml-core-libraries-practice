import keras

model = keras.Sequential([
    keras.Input(shape=(4,)),
    keras.layers.Dense(8, activation="relu"),
    keras.layers.Dense(1)
])
# Depending on installed Keras backend/deployment target:
# model.export("exported_model", format="tf_saved_model")
print("Model ready for export.")
