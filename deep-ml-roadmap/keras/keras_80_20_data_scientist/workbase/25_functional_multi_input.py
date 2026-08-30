import keras
from keras import layers

age = keras.Input(shape=(1,), name="age")
income = keras.Input(shape=(1,), name="income")

x1 = layers.Dense(8, activation="relu")(age)
x2 = layers.Dense(8, activation="relu")(income)
x = layers.Concatenate()([x1, x2])
x = layers.Dense(16, activation="relu")(x)
output = layers.Dense(1)(x)

model = keras.Model(inputs=[age, income], outputs=output)
model.summary()
