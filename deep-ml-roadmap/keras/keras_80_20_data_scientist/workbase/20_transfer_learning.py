import keras
from keras import layers

base = keras.applications.MobileNetV2(
    include_top=False,
    weights="imagenet",
    input_shape=(160,160,3)
)
base.trainable = False

inputs = keras.Input(shape=(160,160,3))
x = keras.applications.mobilenet_v2.preprocess_input(inputs)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()
